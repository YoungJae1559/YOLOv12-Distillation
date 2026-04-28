import os
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
from .cankd_loss import FeatureAdapter, NLKD_IN_Loss
from .fd_cmkd_loss import FDFeatureAdapter, FD_NLKD_Loss
from .plain_kd_loss import PlainFeatureAdapter, PlainFeatureKDLoss
from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = ((F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") *
                     weight).mean(1).sum())
        return loss


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t)**gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    def __init__(self, reg_max=16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    def __init__(self, reg_max):
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)
        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    def __init__(self, sigmas) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    def __init__(self, model, tal_topk=10):
        device = next(model.parameters()).device
        h = model.args
        m = model.model[-1]
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()


def _get_detect(model):
    return model.model[-1]


def _kd_pre_hook(module, inputs, attr):
    feats = inputs[0]
    module.__dict__[attr] = tuple(feats) if isinstance(feats, (list, tuple)) else feats


def _register_kd_hook(det, attr):
    det.register_forward_pre_hook(functools.partial(_kd_pre_hook, attr=attr))


def _spatial_att_map(x, eps=1e-6):
    a = x.abs().mean(1, keepdim=True)
    a = a.flatten(1)
    a = F.normalize(a, p=2, dim=1, eps=eps)
    return a


def _channel_descriptor(x, bins=64, eps=1e-6):
    c = x.abs().mean((2, 3), keepdim=False)
    c = c.unsqueeze(1)
    c = F.interpolate(c, size=bins, mode="linear", align_corners=False)
    c = c.squeeze(1)
    c = F.normalize(c, p=2, dim=1, eps=eps)
    return c


def _batch_relation(z):
    return z @ z.t()


def _detect_in_channels(model):
    det = _get_detect(model)
    channels = []
    for branch in det.cv2:
        first = branch[0]
        if hasattr(first, "conv") and hasattr(first.conv, "in_channels"):
            channels.append(int(first.conv.in_channels))
        else:
            raise RuntimeError("Unable to infer Detect input channels for CanKD.")
    return tuple(channels)


def _ensure_cankd_modules(model, teacher):
    adapters = getattr(model, "cankd_adapters", None)
    blocks = getattr(model, "cankd_blocks", None)
    if (isinstance(adapters, nn.ModuleList) and isinstance(blocks, nn.ModuleList) and len(adapters) == len(blocks)
            and len(adapters) > 0):
        return adapters, blocks
    s_channels = _detect_in_channels(model)
    t_channels = _detect_in_channels(teacher)
    if len(s_channels) != len(t_channels):
        raise RuntimeError(f"CanKD feature count mismatch: {len(s_channels)} vs {len(t_channels)}")
    device = next(model.parameters()).device
    model.cankd_student_channels = s_channels
    model.cankd_teacher_channels = t_channels
    model.cankd_adapters = nn.ModuleList([FeatureAdapter(sc, tc) for sc, tc in zip(s_channels, t_channels)]).to(device)
    model.cankd_blocks = nn.ModuleList(
        [NLKD_IN_Loss(in_channels=tc, dimension=2, loss_weight=1.0) for tc in t_channels]).to(device)
    if int(os.environ.get("RANK", "0")) == 0:
        LOGGER.warning("CanKD modules were initialized lazily after model setup. "
                       "Prefer building them in DetectionTrainer.get_model() so the optimizer sees their parameters.")
    return model.cankd_adapters, model.cankd_blocks


def _ensure_fd_cmkd_modules(model, teacher):
    adapters = getattr(model, "fd_cmkd_adapters", None)
    blocks = getattr(model, "fd_cmkd_blocks", None)
    if (isinstance(adapters, nn.ModuleList) and isinstance(blocks, nn.ModuleList) and len(adapters) == len(blocks)
            and len(adapters) > 0):
        return adapters, blocks
    s_channels = _detect_in_channels(model)
    t_channels = _detect_in_channels(teacher)
    if len(s_channels) != len(t_channels):
        raise RuntimeError(f"FD-CMKD feature count mismatch: {len(s_channels)} vs {len(t_channels)}")
    low_keep_ratio = float(os.environ.get("YOLO_FD_LOW_KEEP_RATIO", "0.25"))
    low_freq_weight = float(os.environ.get("YOLO_FD_LOW_W", "1.0"))
    high_freq_weight = float(os.environ.get("YOLO_FD_HIGH_W", "1.0"))
    fd_loss_weight = float(os.environ.get("YOLO_FD_LOSS_W", "1.0"))
    device = next(model.parameters()).device
    model.fd_cmkd_student_channels = s_channels
    model.fd_cmkd_teacher_channels = t_channels
    model.fd_cmkd_low_keep_ratio = low_keep_ratio
    model.fd_cmkd_low_freq_weight = low_freq_weight
    model.fd_cmkd_high_freq_weight = high_freq_weight
    model.fd_cmkd_loss_weight = fd_loss_weight
    model.fd_cmkd_adapters = nn.ModuleList([FDFeatureAdapter(sc, tc) for sc, tc in zip(s_channels, t_channels)]).to(device)
    model.fd_cmkd_blocks = nn.ModuleList([
        FD_NLKD_Loss(
            in_channels=tc,
            low_keep_ratio=low_keep_ratio,
            low_freq_weight=low_freq_weight,
            high_freq_weight=high_freq_weight,
            loss_weight=fd_loss_weight,
        ) for tc in t_channels
    ]).to(device)
    if int(os.environ.get("RANK", "0")) == 0:
        LOGGER.warning("FD-CMKD modules were initialized lazily after model setup. "
                       "Prefer building them in DetectionTrainer.get_model() so the optimizer sees their parameters.")
    return model.fd_cmkd_adapters, model.fd_cmkd_blocks


def _ensure_plain_kd_modules(model, teacher):
    adapters = getattr(model, "plain_kd_adapters", None)
    blocks = getattr(model, "plain_kd_blocks", None)
    if (isinstance(adapters, nn.ModuleList) and isinstance(blocks, nn.ModuleList) and len(adapters) == len(blocks)
            and len(adapters) > 0):
        return adapters, blocks
    s_channels = _detect_in_channels(model)
    t_channels = _detect_in_channels(teacher)
    if len(s_channels) != len(t_channels):
        raise RuntimeError(f"Plain KD feature count mismatch: {len(s_channels)} vs {len(t_channels)}")
    feat_loss_weight = float(os.environ.get("YOLO_PLAIN_FEAT_W", "1.0"))
    device = next(model.parameters()).device
    model.plain_kd_student_channels = s_channels
    model.plain_kd_teacher_channels = t_channels
    model.plain_kd_loss_weight = feat_loss_weight
    model.plain_kd_adapters = nn.ModuleList(
        [PlainFeatureAdapter(sc, tc) for sc, tc in zip(s_channels, t_channels)]).to(device)
    model.plain_kd_blocks = nn.ModuleList([PlainFeatureKDLoss(loss_weight=feat_loss_weight) for _ in t_channels]).to(device)
    if int(os.environ.get("RANK", "0")) == 0:
        LOGGER.warning("Plain KD modules were initialized lazily after model setup. "
                       "Prefer building them in DetectionTrainer.get_model() so the optimizer sees their parameters.")
    return model.plain_kd_adapters, model.plain_kd_blocks


def _split_detect_logits(preds, no, nc, reg_max):
    feats = preds[1] if isinstance(preds, tuple) else preds
    if isinstance(feats, dict):
        feats = feats["one2many"]
    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], no, -1) for xi in feats], 2).split(
        (reg_max * 4, nc), 1)
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    return feats, pred_distri, pred_scores


def _teacher_anchor_weight(teacher_scores, min_conf=0.0):
    teacher_prob = teacher_scores.sigmoid()
    anchor_weight = teacher_prob.amax(dim=-1, keepdim=True)
    if min_conf > 0:
        anchor_weight = anchor_weight * anchor_weight.ge(min_conf).to(anchor_weight.dtype)
    return anchor_weight.detach()


class KDDetectionLoss(v8DetectionLoss):
    def __init__(
            self,
            model,
            teacher,
            kd_w=0.2,
            kd_scales=(1.0, 1.0, 1.0),
            kd_spatial=1.0,
            kd_channel=0.25,
            kd_relation=0.25,
            kd_bins=64,
            tal_topk=10,
    ):
        super().__init__(model, tal_topk=tal_topk)
        self.teacher = teacher
        self.kd_w = float(kd_w)
        self.kd_scales = kd_scales
        self.kd_spatial = float(kd_spatial)
        self.kd_channel = float(kd_channel)
        self.kd_relation = float(kd_relation)
        self.kd_bins = int(kd_bins)
        self.s_det = _get_detect(model)
        self.t_det = _get_detect(teacher)
        _register_kd_hook(self.s_det, "_kd_s")
        _register_kd_hook(self.t_det, "_kd_t")
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.last_kd = None
        self.last_kd_spatial = None
        self.last_kd_channel = None
        self.last_kd_relation = None

    def __call__(self, preds, batch):
        if not torch.is_grad_enabled():
            return super().__call__(preds, batch)
        with torch.no_grad():
            _ = self.teacher(batch["img"])
        det_loss, items = super().__call__(preds, batch)
        s = getattr(self.s_det, "_kd_s", None)
        t = getattr(self.t_det, "_kd_t", None)
        if s is None or t is None:
            self.last_kd = det_loss.new_zeros(())
            self.last_kd_spatial = det_loss.new_zeros(())
            self.last_kd_channel = det_loss.new_zeros(())
            self.last_kd_relation = det_loss.new_zeros(())
            return det_loss, items
        if len(s) != len(t):
            raise RuntimeError(f"KD feature count mismatch: {len(s)} vs {len(t)}")
        kd_total = det_loss.new_zeros(())
        kd_spatial_total = det_loss.new_zeros(())
        kd_channel_total = det_loss.new_zeros(())
        kd_relation_total = det_loss.new_zeros(())
        for w, fs, ft in zip(self.kd_scales, s, t):
            ft = ft.detach()
            s_sp = _spatial_att_map(fs)
            t_sp = _spatial_att_map(ft)
            kd_sp = F.smooth_l1_loss(s_sp, t_sp)
            s_ch = _channel_descriptor(fs, bins=self.kd_bins)
            t_ch = _channel_descriptor(ft, bins=self.kd_bins)
            kd_ch = F.smooth_l1_loss(s_ch, t_ch)
            s_rel = _batch_relation(s_sp)
            t_rel = _batch_relation(t_sp)
            kd_rel = F.smooth_l1_loss(s_rel, t_rel)
            kd_scale = self.kd_spatial * kd_sp + self.kd_channel * kd_ch + self.kd_relation * kd_rel
            kd_total = kd_total + float(w) * kd_scale
            kd_spatial_total = kd_spatial_total + float(w) * kd_sp
            kd_channel_total = kd_channel_total + float(w) * kd_ch
            kd_relation_total = kd_relation_total + float(w) * kd_rel
        self.last_kd = kd_total.detach()
        self.last_kd_spatial = kd_spatial_total.detach()
        self.last_kd_channel = kd_channel_total.detach()
        self.last_kd_relation = kd_relation_total.detach()
        if int(os.environ.get("RANK", "0")) == 0 and not hasattr(self, "_kd_printed"):
            print("KD loss active: spatial + channel + relation")
            self._kd_printed = True
        total_loss = det_loss + self.kd_w * kd_total
        return total_loss, items


class PlainKDDetectionLoss(v8DetectionLoss):
    def __init__(
            self,
            model,
            teacher,
            kd_w=0.2,
            kd_scales=(1.0, 1.0, 1.0),
            head_kd_w=0.1,
            head_cls_w=1.0,
            head_reg_w=1.0,
            head_tau=1.0,
            head_min_conf=0.0,
            feat_warmup_epochs=5.0,
            feat_decay_start=0.7,
            feat_min_ratio=0.4,
            head_warmup_epochs=3.0,
            head_decay_start=0.9,
            head_decay_min_ratio=0.8,
            tal_topk=10):
        super().__init__(model, tal_topk=tal_topk)
        self.model = model
        self.teacher = teacher
        self.kd_w = float(kd_w)
        self.kd_scales = tuple(float(x) for x in kd_scales)
        self.head_kd_w = float(head_kd_w)
        self.head_cls_w = float(head_cls_w)
        self.head_reg_w = float(head_reg_w)
        self.head_tau = float(head_tau)
        self.head_min_conf = float(head_min_conf)
        self.feat_warmup_epochs = float(feat_warmup_epochs)
        self.feat_decay_start = float(feat_decay_start)
        self.feat_min_ratio = float(feat_min_ratio)
        self.head_warmup_epochs = float(head_warmup_epochs)
        self.head_decay_start = float(head_decay_start)
        self.head_decay_min_ratio = float(head_decay_min_ratio)
        self.s_det = _get_detect(model)
        self.t_det = _get_detect(teacher)
        _register_kd_hook(self.s_det, "_plain_kd_s")
        _register_kd_hook(self.t_det, "_plain_kd_t")
        self.adapters, self.plain_kd_blocks = _ensure_plain_kd_modules(model, teacher)
        self.teacher_path = getattr(model, "plain_kd_teacher_path", "") or os.environ.get("YOLO_TEACHER", "")
        self.student_channels = getattr(model, "plain_kd_student_channels", _detect_in_channels(model))
        self.teacher_channels = getattr(model, "plain_kd_teacher_channels", _detect_in_channels(teacher))
        self.plain_feat_w = getattr(model, "plain_kd_loss_weight", float(os.environ.get("YOLO_PLAIN_FEAT_W", "1.0")))
        self.debug_steps = int(os.environ.get("YOLO_PLAIN_KD_DEBUG_STEPS", "3"))
        self.debug_seen = 0
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.last_kd = None
        self.last_kd_scales = None
        self.last_head_kd = None
        self.last_cls_kd = None
        self.last_reg_kd = None
        self.last_feat_scale = None
        self.last_head_scale = None

    @staticmethod
    def _schedule_ratio(epoch, total_epochs, warmup_epochs, decay_start, min_ratio):
        ratio = 1.0
        step = float(epoch) + 1.0
        if warmup_epochs > 0:
            ratio *= min(step / warmup_epochs, 1.0)
        if total_epochs and decay_start < 1.0:
            progress = step / max(float(total_epochs), 1.0)
            if progress > decay_start:
                decay_progress = min((progress - decay_start) / max(1.0 - decay_start, 1e-6), 1.0)
                ratio *= 1.0 - decay_progress * (1.0 - min_ratio)
        return float(max(ratio, 0.0))

    def _kd_schedule_scales(self, det_loss):
        epoch = getattr(self.model, "kd_epoch", 0)
        total_epochs = getattr(self.model, "kd_total_epochs", 0)
        feat_scale = self._schedule_ratio(
            epoch,
            total_epochs,
            self.feat_warmup_epochs,
            self.feat_decay_start,
            self.feat_min_ratio,
        )
        head_scale = self._schedule_ratio(
            epoch,
            total_epochs,
            self.head_warmup_epochs,
            self.head_decay_start,
            self.head_decay_min_ratio,
        )
        feat_scale = det_loss.new_tensor(feat_scale)
        head_scale = det_loss.new_tensor(head_scale)
        self.last_feat_scale = feat_scale.detach()
        self.last_head_scale = head_scale.detach()
        return feat_scale, head_scale

    def _head_output_kd(self, student_preds, teacher_preds, det_loss):
        _, s_distri, s_scores = _split_detect_logits(student_preds, self.no, self.nc, self.reg_max)
        _, t_distri, t_scores = _split_detect_logits(teacher_preds, self.no, self.nc, self.reg_max)
        tau = max(self.head_tau, 1e-6)
        anchor_weight = _teacher_anchor_weight(t_scores, min_conf=self.head_min_conf)
        anchor_norm = anchor_weight.sum().clamp_min(1.0)

        cls_target = torch.sigmoid(t_scores / tau)
        cls_loss = F.binary_cross_entropy_with_logits(s_scores / tau, cls_target, reduction="none")
        cls_loss = (cls_loss * anchor_weight).sum() / (anchor_norm * self.nc)
        cls_loss = cls_loss * (tau**2)

        if self.use_dfl:
            s_bins = s_distri.view(s_distri.shape[0], s_distri.shape[1], 4, self.reg_max)
            t_bins = t_distri.view(t_distri.shape[0], t_distri.shape[1], 4, self.reg_max)
            reg_loss = F.kl_div(
                F.log_softmax(s_bins / tau, dim=-1),
                F.softmax(t_bins / tau, dim=-1),
                reduction="none",
            ).sum(-1)
            reg_loss = (reg_loss * anchor_weight).sum() / (anchor_norm * 4)
            reg_loss = reg_loss * (tau**2)
        else:
            reg_loss = det_loss.new_zeros(())

        head_kd = self.head_cls_w * cls_loss + self.head_reg_w * reg_loss
        return head_kd, cls_loss.detach(), reg_loss.detach()

    def __call__(self, preds, batch):
        if not torch.is_grad_enabled():
            return super().__call__(preds, batch)
        with torch.no_grad():
            teacher_preds = self.teacher(batch["img"])
        det_loss, items = super().__call__(preds, batch)
        s = getattr(self.s_det, "_plain_kd_s", None)
        t = getattr(self.t_det, "_plain_kd_t", None)
        if s is None or t is None:
            self.last_kd = det_loss.new_zeros(())
            self.last_head_kd = det_loss.new_zeros(())
            self.last_cls_kd = det_loss.new_zeros(())
            self.last_reg_kd = det_loss.new_zeros(())
            return det_loss, items
        if len(s) != len(t):
            raise RuntimeError(f"Plain KD feature count mismatch: {len(s)} vs {len(t)}")
        if len(s) != len(self.adapters) or len(s) != len(self.plain_kd_blocks):
            raise RuntimeError(
                f"Plain KD module count mismatch: features={len(s)}, adapters={len(self.adapters)}, blocks={len(self.plain_kd_blocks)}"
            )
        kd_total = det_loss.new_zeros(())
        scale_logs = []
        scale_losses = []
        for idx, (w, fs, ft, adapter, plain_block) in enumerate(
                zip(self.kd_scales, s, t, self.adapters, self.plain_kd_blocks)):
            fs_raw_shape = tuple(fs.shape)
            ft = ft.detach()
            fs = adapter(fs)
            fs_adapted_shape = tuple(fs.shape)
            if fs.shape[-2:] != ft.shape[-2:]:
                fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)
            plain_loss = plain_block(fs, ft)
            kd_total = kd_total + float(w) * plain_loss
            scale_losses.append(plain_loss.detach())
            scale_logs.append(
                f"P{idx + 3}: raw_s={fs_raw_shape}, adapted_s={fs_adapted_shape}, teacher={tuple(ft.shape)}, "
                f"loss={float(plain_loss.detach()):.4f}, w={float(w):.2f}"
            )
        self.last_kd = kd_total.detach()
        self.last_kd_scales = torch.stack(scale_losses) if scale_losses else kd_total.new_zeros((0, ))
        head_kd = det_loss.new_zeros(())
        cls_kd = det_loss.new_zeros(())
        reg_kd = det_loss.new_zeros(())
        feat_scale, head_scale = self._kd_schedule_scales(det_loss)
        if self.head_kd_w > 0:
            head_kd, cls_kd, reg_kd = self._head_output_kd(preds, teacher_preds, det_loss)
        self.last_head_kd = head_kd.detach()
        self.last_cls_kd = cls_kd.detach()
        self.last_reg_kd = reg_kd.detach()
        if int(os.environ.get("RANK", "0")) == 0 and not hasattr(self, "_plain_kd_printed"):
            LOGGER.info("Plain KD loss active: adapter + feature MSE + head output KD")
            self._plain_kd_printed = True
        total_loss = det_loss + (self.kd_w * feat_scale) * kd_total + (self.head_kd_w * head_scale) * head_kd
        if int(os.environ.get("RANK", "0")) == 0 and self.debug_seen < self.debug_steps:
            LOGGER.info("Plain KD debug "
                        f"[{self.debug_seen + 1}/{self.debug_steps}] "
                        f"teacher={self.teacher_path or 'loaded_from_args'} "
                        f"student_channels={self.student_channels} teacher_channels={self.teacher_channels} "
                        f"feat_loss_w={self.plain_feat_w} feat_scale={float(feat_scale):.3f} "
                        f"head_scale={float(head_scale):.3f} det_loss={float(det_loss.detach()):.4f} "
                        f"feature_kd={float(kd_total.detach()):.4f} head_kd={float(head_kd.detach()):.4f} "
                        f"cls_kd={float(cls_kd):.4f} reg_kd={float(reg_kd):.4f} "
                        f"total_loss={float(total_loss.detach()):.4f}")
            for msg in scale_logs:
                LOGGER.info(f"Plain KD debug    {msg}")
            self.debug_seen += 1
        return total_loss, items


class CanKDDetectionLoss(v8DetectionLoss):
    def __init__(
            self,
            model,
            teacher,
            kd_w=0.2,
            kd_scales=(1.0, 1.0, 1.0),
            head_kd_w=0.1,
            head_cls_w=1.0,
            head_reg_w=1.0,
            head_tau=1.0,
            head_min_conf=0.0,
            feat_warmup_epochs=5.0,
            feat_decay_start=0.7,
            feat_min_ratio=0.4,
            head_warmup_epochs=3.0,
            head_decay_start=0.9,
            head_decay_min_ratio=0.8,
            tal_topk=10):
        super().__init__(model, tal_topk=tal_topk)
        self.model = model
        self.teacher = teacher
        self.kd_w = float(kd_w)
        self.kd_scales = tuple(float(x) for x in kd_scales)
        self.head_kd_w = float(head_kd_w)
        self.head_cls_w = float(head_cls_w)
        self.head_reg_w = float(head_reg_w)
        self.head_tau = float(head_tau)
        self.head_min_conf = float(head_min_conf)
        self.feat_warmup_epochs = float(feat_warmup_epochs)
        self.feat_decay_start = float(feat_decay_start)
        self.feat_min_ratio = float(feat_min_ratio)
        self.head_warmup_epochs = float(head_warmup_epochs)
        self.head_decay_start = float(head_decay_start)
        self.head_decay_min_ratio = float(head_decay_min_ratio)
        self.s_det = _get_detect(model)
        self.t_det = _get_detect(teacher)
        _register_kd_hook(self.s_det, "_kd_s")
        _register_kd_hook(self.t_det, "_kd_t")
        self.adapters, self.cankd_blocks = _ensure_cankd_modules(model, teacher)
        self.teacher_path = getattr(model, "cankd_teacher_path", "") or os.environ.get("YOLO_TEACHER", "")
        self.student_channels = getattr(model, "cankd_student_channels", _detect_in_channels(model))
        self.teacher_channels = getattr(model, "cankd_teacher_channels", _detect_in_channels(teacher))
        self.debug_steps = int(os.environ.get("YOLO_CANKD_DEBUG_STEPS", "3"))
        self.debug_seen = 0
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.last_kd = None
        self.last_kd_scales = None
        self.last_head_kd = None
        self.last_cls_kd = None
        self.last_reg_kd = None
        self.last_feat_scale = None
        self.last_head_scale = None

    @staticmethod
    def _schedule_ratio(epoch, total_epochs, warmup_epochs, decay_start, min_ratio):
        ratio = 1.0
        step = float(epoch) + 1.0
        if warmup_epochs > 0:
            ratio *= min(step / warmup_epochs, 1.0)
        if total_epochs and decay_start < 1.0:
            progress = step / max(float(total_epochs), 1.0)
            if progress > decay_start:
                decay_progress = min((progress - decay_start) / max(1.0 - decay_start, 1e-6), 1.0)
                ratio *= 1.0 - decay_progress * (1.0 - min_ratio)
        return float(max(ratio, 0.0))

    def _kd_schedule_scales(self, det_loss):
        epoch = getattr(self.model, "kd_epoch", 0)
        total_epochs = getattr(self.model, "kd_total_epochs", 0)
        feat_scale = self._schedule_ratio(
            epoch,
            total_epochs,
            self.feat_warmup_epochs,
            self.feat_decay_start,
            self.feat_min_ratio,
        )
        head_scale = self._schedule_ratio(
            epoch,
            total_epochs,
            self.head_warmup_epochs,
            self.head_decay_start,
            self.head_decay_min_ratio,
        )
        feat_scale = det_loss.new_tensor(feat_scale)
        head_scale = det_loss.new_tensor(head_scale)
        self.last_feat_scale = feat_scale.detach()
        self.last_head_scale = head_scale.detach()
        return feat_scale, head_scale

    def _head_output_kd(self, student_preds, teacher_preds, det_loss):
        _, s_distri, s_scores = _split_detect_logits(student_preds, self.no, self.nc, self.reg_max)
        _, t_distri, t_scores = _split_detect_logits(teacher_preds, self.no, self.nc, self.reg_max)
        tau = max(self.head_tau, 1e-6)
        anchor_weight = _teacher_anchor_weight(t_scores, min_conf=self.head_min_conf)
        anchor_norm = anchor_weight.sum().clamp_min(1.0)

        cls_target = torch.sigmoid(t_scores / tau)
        cls_loss = F.binary_cross_entropy_with_logits(s_scores / tau, cls_target, reduction="none")
        cls_loss = (cls_loss * anchor_weight).sum() / (anchor_norm * self.nc)
        cls_loss = cls_loss * (tau**2)

        if self.use_dfl:
            s_bins = s_distri.view(s_distri.shape[0], s_distri.shape[1], 4, self.reg_max)
            t_bins = t_distri.view(t_distri.shape[0], t_distri.shape[1], 4, self.reg_max)
            reg_loss = F.kl_div(
                F.log_softmax(s_bins / tau, dim=-1),
                F.softmax(t_bins / tau, dim=-1),
                reduction="none",
            ).sum(-1)
            reg_loss = (reg_loss * anchor_weight).sum() / (anchor_norm * 4)
            reg_loss = reg_loss * (tau**2)
        else:
            reg_loss = det_loss.new_zeros(())

        head_kd = self.head_cls_w * cls_loss + self.head_reg_w * reg_loss
        return head_kd, cls_loss.detach(), reg_loss.detach()

    def __call__(self, preds, batch):
        if not torch.is_grad_enabled():
            return super().__call__(preds, batch)
        with torch.no_grad():
            teacher_preds = self.teacher(batch["img"])
        det_loss, items = super().__call__(preds, batch)
        s = getattr(self.s_det, "_kd_s", None)
        t = getattr(self.t_det, "_kd_t", None)
        if s is None or t is None:
            self.last_kd = det_loss.new_zeros(())
            self.last_head_kd = det_loss.new_zeros(())
            self.last_cls_kd = det_loss.new_zeros(())
            self.last_reg_kd = det_loss.new_zeros(())
            return det_loss, items
        if len(s) != len(t):
            raise RuntimeError(f"CanKD feature count mismatch: {len(s)} vs {len(t)}")
        if len(s) != len(self.adapters) or len(s) != len(self.cankd_blocks):
            raise RuntimeError(
                f"CanKD module count mismatch: features={len(s)}, adapters={len(self.adapters)}, blocks={len(self.cankd_blocks)}"
            )
        kd_total = det_loss.new_zeros(())
        scale_logs = []
        scale_losses = []
        for idx, (w, fs, ft, adapter,
                  can_block) in enumerate(zip(self.kd_scales, s, t, self.adapters, self.cankd_blocks)):
            fs_raw_shape = tuple(fs.shape)
            ft = ft.detach()
            fs = adapter(fs)
            fs_adapted_shape = tuple(fs.shape)
            if fs.shape[-2:] != ft.shape[-2:]:
                fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)
            can_loss = can_block(fs, ft)
            kd_total = kd_total + float(w) * can_loss
            scale_losses.append(can_loss.detach())
            scale_logs.append(f"P{idx + 3}: raw_s={fs_raw_shape}, adapted_s={fs_adapted_shape}, "
                              f"teacher={tuple(ft.shape)}, loss={float(can_loss.detach()):.4f}, w={float(w):.2f}")
        self.last_kd = kd_total.detach()
        self.last_kd_scales = torch.stack(scale_losses) if scale_losses else kd_total.new_zeros((0, ))
        head_kd = det_loss.new_zeros(())
        cls_kd = det_loss.new_zeros(())
        reg_kd = det_loss.new_zeros(())
        feat_scale, head_scale = self._kd_schedule_scales(det_loss)
        if self.head_kd_w > 0:
            head_kd, cls_kd, reg_kd = self._head_output_kd(preds, teacher_preds, det_loss)
        self.last_head_kd = head_kd.detach()
        self.last_cls_kd = cls_kd.detach()
        self.last_reg_kd = reg_kd.detach()
        if int(os.environ.get("RANK", "0")) == 0 and not hasattr(self, "_cankd_printed"):
            LOGGER.info("CanKD loss active: feature CanKD + teacher cls/regression output distillation")
            self._cankd_printed = True
        total_loss = det_loss + (self.kd_w * feat_scale) * kd_total + (self.head_kd_w * head_scale) * head_kd
        if int(os.environ.get("RANK", "0")) == 0 and self.debug_seen < self.debug_steps:
            LOGGER.info("CanKD debug "
                        f"[{self.debug_seen + 1}/{self.debug_steps}] "
                        f"teacher={self.teacher_path or 'loaded_from_args'} "
                        f"student_channels={self.student_channels} teacher_channels={self.teacher_channels} "
                        f"feat_scale={float(feat_scale):.3f} head_scale={float(head_scale):.3f} "
                        f"det_loss={float(det_loss.detach()):.4f} feature_kd={float(kd_total.detach()):.4f} "
                        f"head_kd={float(head_kd.detach()):.4f} cls_kd={float(cls_kd):.4f} "
                        f"reg_kd={float(reg_kd):.4f} "
                        f"total_loss={float(total_loss.detach()):.4f}")
            for msg in scale_logs:
                LOGGER.info(f"CanKD debug    {msg}")
            self.debug_seen += 1
        return total_loss, items


class FDCMKDDetectionLoss(v8DetectionLoss):
    def __init__(
            self,
            model,
            teacher,
            kd_w=0.2,
            kd_scales=(1.0, 1.0, 1.0),
            head_kd_w=0.1,
            head_cls_w=1.0,
            head_reg_w=1.0,
            head_tau=1.0,
            head_min_conf=0.0,
            feat_warmup_epochs=5.0,
            feat_decay_start=0.7,
            feat_min_ratio=0.4,
            head_warmup_epochs=3.0,
            head_decay_start=0.9,
            head_decay_min_ratio=0.8,
            tal_topk=10):
        super().__init__(model, tal_topk=tal_topk)
        self.model = model
        self.teacher = teacher
        self.kd_w = float(kd_w)
        self.kd_scales = tuple(float(x) for x in kd_scales)
        self.head_kd_w = float(head_kd_w)
        self.head_cls_w = float(head_cls_w)
        self.head_reg_w = float(head_reg_w)
        self.head_tau = float(head_tau)
        self.head_min_conf = float(head_min_conf)
        self.feat_warmup_epochs = float(feat_warmup_epochs)
        self.feat_decay_start = float(feat_decay_start)
        self.feat_min_ratio = float(feat_min_ratio)
        self.head_warmup_epochs = float(head_warmup_epochs)
        self.head_decay_start = float(head_decay_start)
        self.head_decay_min_ratio = float(head_decay_min_ratio)
        self.s_det = _get_detect(model)
        self.t_det = _get_detect(teacher)
        _register_kd_hook(self.s_det, "_fd_kd_s")
        _register_kd_hook(self.t_det, "_fd_kd_t")
        self.adapters, self.fd_cmkd_blocks = _ensure_fd_cmkd_modules(model, teacher)
        self.teacher_path = getattr(model, "fd_cmkd_teacher_path", "") or os.environ.get("YOLO_TEACHER", "")
        self.student_channels = getattr(model, "fd_cmkd_student_channels", _detect_in_channels(model))
        self.teacher_channels = getattr(model, "fd_cmkd_teacher_channels", _detect_in_channels(teacher))
        self.low_keep_ratio = getattr(model, "fd_cmkd_low_keep_ratio", float(os.environ.get("YOLO_FD_LOW_KEEP_RATIO", "0.25")))
        self.low_freq_weight = getattr(model, "fd_cmkd_low_freq_weight", float(os.environ.get("YOLO_FD_LOW_W", "1.0")))
        self.high_freq_weight = getattr(model, "fd_cmkd_high_freq_weight", float(os.environ.get("YOLO_FD_HIGH_W", "1.0")))
        self.fd_loss_weight = getattr(model, "fd_cmkd_loss_weight", float(os.environ.get("YOLO_FD_LOSS_W", "1.0")))
        self.debug_steps = int(os.environ.get("YOLO_FD_CMKD_DEBUG_STEPS", "3"))
        self.debug_seen = 0
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.last_kd = None
        self.last_kd_scales = None
        self.last_head_kd = None
        self.last_cls_kd = None
        self.last_reg_kd = None
        self.last_feat_scale = None
        self.last_head_scale = None

    @staticmethod
    def _schedule_ratio(epoch, total_epochs, warmup_epochs, decay_start, min_ratio):
        ratio = 1.0
        step = float(epoch) + 1.0
        if warmup_epochs > 0:
            ratio *= min(step / warmup_epochs, 1.0)
        if total_epochs and decay_start < 1.0:
            progress = step / max(float(total_epochs), 1.0)
            if progress > decay_start:
                decay_progress = min((progress - decay_start) / max(1.0 - decay_start, 1e-6), 1.0)
                ratio *= 1.0 - decay_progress * (1.0 - min_ratio)
        return float(max(ratio, 0.0))

    def _kd_schedule_scales(self, det_loss):
        epoch = getattr(self.model, "kd_epoch", 0)
        total_epochs = getattr(self.model, "kd_total_epochs", 0)
        feat_scale = self._schedule_ratio(
            epoch,
            total_epochs,
            self.feat_warmup_epochs,
            self.feat_decay_start,
            self.feat_min_ratio,
        )
        head_scale = self._schedule_ratio(
            epoch,
            total_epochs,
            self.head_warmup_epochs,
            self.head_decay_start,
            self.head_decay_min_ratio,
        )
        feat_scale = det_loss.new_tensor(feat_scale)
        head_scale = det_loss.new_tensor(head_scale)
        self.last_feat_scale = feat_scale.detach()
        self.last_head_scale = head_scale.detach()
        return feat_scale, head_scale

    def _head_output_kd(self, student_preds, teacher_preds, det_loss):
        _, s_distri, s_scores = _split_detect_logits(student_preds, self.no, self.nc, self.reg_max)
        _, t_distri, t_scores = _split_detect_logits(teacher_preds, self.no, self.nc, self.reg_max)
        tau = max(self.head_tau, 1e-6)
        anchor_weight = _teacher_anchor_weight(t_scores, min_conf=self.head_min_conf)
        anchor_norm = anchor_weight.sum().clamp_min(1.0)
        cls_target = torch.sigmoid(t_scores / tau)
        cls_loss = F.binary_cross_entropy_with_logits(s_scores / tau, cls_target, reduction="none")
        cls_loss = (cls_loss * anchor_weight).sum() / (anchor_norm * self.nc)
        cls_loss = cls_loss * (tau**2)
        if self.use_dfl:
            s_bins = s_distri.view(s_distri.shape[0], s_distri.shape[1], 4, self.reg_max)
            t_bins = t_distri.view(t_distri.shape[0], t_distri.shape[1], 4, self.reg_max)
            reg_loss = F.kl_div(
                F.log_softmax(s_bins / tau, dim=-1),
                F.softmax(t_bins / tau, dim=-1),
                reduction="none",
            ).sum(-1)
            reg_loss = (reg_loss * anchor_weight).sum() / (anchor_norm * 4)
            reg_loss = reg_loss * (tau**2)
        else:
            reg_loss = det_loss.new_zeros(())
        head_kd = self.head_cls_w * cls_loss + self.head_reg_w * reg_loss
        return head_kd, cls_loss.detach(), reg_loss.detach()

    def __call__(self, preds, batch):
        if not torch.is_grad_enabled():
            return super().__call__(preds, batch)
        with torch.no_grad():
            teacher_preds = self.teacher(batch["img"])
        det_loss, items = super().__call__(preds, batch)
        s = getattr(self.s_det, "_fd_kd_s", None)
        t = getattr(self.t_det, "_fd_kd_t", None)
        if s is None or t is None:
            self.last_kd = det_loss.new_zeros(())
            self.last_head_kd = det_loss.new_zeros(())
            self.last_cls_kd = det_loss.new_zeros(())
            self.last_reg_kd = det_loss.new_zeros(())
            return det_loss, items
        if len(s) != len(t):
            raise RuntimeError(f"FD-CMKD feature count mismatch: {len(s)} vs {len(t)}")
        if len(s) != len(self.adapters) or len(s) != len(self.fd_cmkd_blocks):
            raise RuntimeError(
                f"FD-CMKD module count mismatch: features={len(s)}, adapters={len(self.adapters)}, blocks={len(self.fd_cmkd_blocks)}"
            )
        kd_total = det_loss.new_zeros(())
        scale_logs = []
        scale_losses = []
        for idx, (w, fs, ft, adapter, fd_block) in enumerate(
                zip(self.kd_scales, s, t, self.adapters, self.fd_cmkd_blocks)):
            fs_raw_shape = tuple(fs.shape)
            ft = ft.detach()
            fs = adapter(fs)
            fs_adapted_shape = tuple(fs.shape)
            if fs.shape[-2:] != ft.shape[-2:]:
                fs = F.interpolate(fs, size=ft.shape[-2:], mode="bilinear", align_corners=False)
            fd_loss = fd_block(fs, ft)
            kd_total = kd_total + float(w) * fd_loss
            scale_losses.append(fd_loss.detach())
            low_loss = getattr(fd_block, "last_low_loss", None)
            high_loss = getattr(fd_block, "last_high_loss", None)
            low_text = f"{float(low_loss):.4f}" if low_loss is not None else "n/a"
            high_text = f"{float(high_loss):.4f}" if high_loss is not None else "n/a"
            scale_logs.append(
                f"P{idx + 3}: raw_s={fs_raw_shape}, adapted_s={fs_adapted_shape}, teacher={tuple(ft.shape)}, "
                f"loss={float(fd_loss.detach()):.4f}, low={low_text}, high={high_text}, w={float(w):.2f}"
            )
        self.last_kd = kd_total.detach()
        self.last_kd_scales = torch.stack(scale_losses) if scale_losses else kd_total.new_zeros((0, ))
        head_kd = det_loss.new_zeros(())
        cls_kd = det_loss.new_zeros(())
        reg_kd = det_loss.new_zeros(())
        feat_scale, head_scale = self._kd_schedule_scales(det_loss)
        if self.head_kd_w > 0:
            head_kd, cls_kd, reg_kd = self._head_output_kd(preds, teacher_preds, det_loss)
        self.last_head_kd = head_kd.detach()
        self.last_cls_kd = cls_kd.detach()
        self.last_reg_kd = reg_kd.detach()
        if int(os.environ.get("RANK", "0")) == 0 and not hasattr(self, "_fd_cmkd_printed"):
            LOGGER.info("FD-CMKD loss active: non-local feature KD + DC filter + low/high-pass + MSE/LogMSE")
            self._fd_cmkd_printed = True
        total_loss = det_loss + (self.kd_w * feat_scale) * kd_total + (self.head_kd_w * head_scale) * head_kd
        if int(os.environ.get("RANK", "0")) == 0 and self.debug_seen < self.debug_steps:
            LOGGER.info("FD-CMKD debug "
                        f"[{self.debug_seen + 1}/{self.debug_steps}] "
                        f"teacher={self.teacher_path or 'loaded_from_args'} "
                        f"student_channels={self.student_channels} teacher_channels={self.teacher_channels} "
                        f"low_keep_ratio={self.low_keep_ratio} low_w={self.low_freq_weight} "
                        f"high_w={self.high_freq_weight} fd_loss_w={self.fd_loss_weight} "
                        f"feat_scale={float(feat_scale):.3f} head_scale={float(head_scale):.3f} "
                        f"det_loss={float(det_loss.detach()):.4f} feature_kd={float(kd_total.detach()):.4f} "
                        f"head_kd={float(head_kd.detach()):.4f} cls_kd={float(cls_kd):.4f} "
                        f"reg_kd={float(reg_kd):.4f} total_loss={float(total_loss.detach()):.4f}")
            for msg in scale_logs:
                LOGGER.info(f"FD-CMKD debug    {msg}")
            self.debug_seen += 1
        return total_loss, items


class v8SegmentationLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        loss = torch.zeros(4, device=self.device)
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                            "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help.") from e
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
            loss[1] = self.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto,
                                                       pred_masks, imgsz, self.overlap)
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.box
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def single_mask_loss(gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor,
                         area: torch.Tensor) -> torch.Tensor:
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        _, _, mask_h, mask_w = proto.shape
        loss = 0
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                loss += self.single_mask_loss(gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i],
                                              marea_i[fg_mask_i])
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()
        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        loss = torch.zeros(5, device=self.device)
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            loss[1], loss[2] = self.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx,
                                                             stride_tensor, target_bboxes, pred_kpts)
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        loss[3] *= self.hyp.cls
        loss[4] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes,
                                 pred_kpts):
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()
        batched_keypoints = torch.zeros((batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]),
                                        device=keypoints.device)
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, :keypoints_i.shape[0]] = keypoints_i
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2]))
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)
        kpts_loss = 0
        kpts_obj_loss = 0
        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()
        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                            "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                            "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help.") from e
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)
        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
        else:
            loss[0] += (pred_angle * 0).sum()
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return loss.sum() * batch_size, loss.detach()

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    def __init__(self, model):
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
