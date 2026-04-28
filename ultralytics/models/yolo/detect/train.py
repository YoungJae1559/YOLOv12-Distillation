import math
import os
import random
from copy import copy
import numpy as np
import torch.nn as nn
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.cankd_loss import FeatureAdapter, NLKD_IN_Loss
from ultralytics.utils.fd_cmkd_loss import FDFeatureAdapter, FD_NLKD_Loss
from ultralytics.utils.plain_kd_loss import PlainFeatureAdapter, PlainFeatureKDLoss
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


def _detect_input_channels(model):
    det = model.model[-1]
    channels = []
    for branch in det.cv2:
        first = branch[0]
        if hasattr(first, "conv") and hasattr(first.conv, "in_channels"):
            channels.append(int(first.conv.in_channels))
        else:
            raise RuntimeError("Unable to infer Detect input channels for CanKD setup.")
    return tuple(channels)


def _build_cankd_modules(model, teacher_w):
    teacher, _ = attempt_load_one_weight(teacher_w, device="cpu", fuse=False)
    s_channels = _detect_input_channels(model)
    t_channels = _detect_input_channels(teacher)
    if len(s_channels) != len(t_channels):
        raise RuntimeError(f"CanKD feature count mismatch: {len(s_channels)} vs {len(t_channels)}")
    model.cankd_adapters = nn.ModuleList([FeatureAdapter(sc, tc) for sc, tc in zip(s_channels, t_channels)])
    model.cankd_blocks = nn.ModuleList(
        [NLKD_IN_Loss(in_channels=tc, dimension=2, loss_weight=1.0) for tc in t_channels])
    model.cankd_student_channels = s_channels
    model.cankd_teacher_channels = t_channels
    model.cankd_teacher_path = str(teacher_w)


def _build_fd_cmkd_modules(model, teacher_w):
    teacher, _ = attempt_load_one_weight(teacher_w, device="cpu", fuse=False)
    s_channels = _detect_input_channels(model)
    t_channels = _detect_input_channels(teacher)
    if len(s_channels) != len(t_channels):
        raise RuntimeError(f"FD-CMKD feature count mismatch: {len(s_channels)} vs {len(t_channels)}")
    low_keep_ratio = float(os.environ.get("YOLO_FD_LOW_KEEP_RATIO", "0.25"))
    low_freq_weight = float(os.environ.get("YOLO_FD_LOW_W", "1.0"))
    high_freq_weight = float(os.environ.get("YOLO_FD_HIGH_W", "1.0"))
    fd_loss_weight = float(os.environ.get("YOLO_FD_LOSS_W", "1.0"))
    model.fd_cmkd_adapters = nn.ModuleList([FDFeatureAdapter(sc, tc) for sc, tc in zip(s_channels, t_channels)])
    model.fd_cmkd_blocks = nn.ModuleList([
        FD_NLKD_Loss(
            in_channels=tc,
            low_keep_ratio=low_keep_ratio,
            low_freq_weight=low_freq_weight,
            high_freq_weight=high_freq_weight,
            loss_weight=fd_loss_weight,
        ) for tc in t_channels
    ])
    model.fd_cmkd_student_channels = s_channels
    model.fd_cmkd_teacher_channels = t_channels
    model.fd_cmkd_teacher_path = str(teacher_w)
    model.fd_cmkd_low_keep_ratio = low_keep_ratio
    model.fd_cmkd_low_freq_weight = low_freq_weight
    model.fd_cmkd_high_freq_weight = high_freq_weight
    model.fd_cmkd_loss_weight = fd_loss_weight


def _build_plain_kd_modules(model, teacher_w):
    teacher, _ = attempt_load_one_weight(teacher_w, device="cpu", fuse=False)
    s_channels = _detect_input_channels(model)
    t_channels = _detect_input_channels(teacher)
    if len(s_channels) != len(t_channels):
        raise RuntimeError(f"Plain KD feature count mismatch: {len(s_channels)} vs {len(t_channels)}")
    feat_loss_weight = float(os.environ.get("YOLO_PLAIN_FEAT_W", "1.0"))
    model.plain_kd_adapters = nn.ModuleList([PlainFeatureAdapter(sc, tc) for sc, tc in zip(s_channels, t_channels)])
    model.plain_kd_blocks = nn.ModuleList([PlainFeatureKDLoss(loss_weight=feat_loss_weight) for _ in t_channels])
    model.plain_kd_student_channels = s_channels
    model.plain_kd_teacher_channels = t_channels
    model.plain_kd_teacher_path = str(teacher_w)
    model.plain_kd_loss_weight = feat_loss_weight


class DetectionTrainer(BaseTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch["img"]
            sz = (random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride)) //
                  self.stride * self.stride)
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        self.model.nc = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args = self.args

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        kd_type = (getattr(self.args, "kd_type", "") or os.environ.get("YOLO_KD_TYPE", "")).lower()
        teacher_w = getattr(self.args, "teacher", "") or os.environ.get("YOLO_TEACHER", "")
        if kd_type == "cankd" and teacher_w:
            _build_cankd_modules(model, teacher_w)
            if RANK in {-1, 0}:
                LOGGER.info("CanKD setup ready: "
                            f"teacher_path={model.cankd_teacher_path}, "
                            f"student_detect_channels={model.cankd_student_channels}, "
                            f"teacher_detect_channels={model.cankd_teacher_channels}, "
                            f"num_scales={len(model.cankd_blocks)}")
        elif kd_type == "fd_cmkd" and teacher_w:
            _build_fd_cmkd_modules(model, teacher_w)
            if RANK in {-1, 0}:
                LOGGER.info("FD-CMKD setup ready: "
                            f"teacher_path={model.fd_cmkd_teacher_path}, "
                            f"student_detect_channels={model.fd_cmkd_student_channels}, "
                            f"teacher_detect_channels={model.fd_cmkd_teacher_channels}, "
                            f"num_scales={len(model.fd_cmkd_blocks)}, "
                            f"low_keep_ratio={model.fd_cmkd_low_keep_ratio}, "
                            f"low_w={model.fd_cmkd_low_freq_weight}, "
                            f"high_w={model.fd_cmkd_high_freq_weight}, "
                            f"loss_w={model.fd_cmkd_loss_weight}")
        elif kd_type == "plain_kd" and teacher_w:
            _build_plain_kd_modules(model, teacher_w)
            if RANK in {-1, 0}:
                LOGGER.info("Plain KD setup ready: "
                            f"teacher_path={model.plain_kd_teacher_path}, "
                            f"student_detect_channels={model.plain_kd_student_channels}, "
                            f"teacher_detect_channels={model.plain_kd_teacher_channels}, "
                            f"num_scales={len(model.plain_kd_blocks)}, "
                            f"feat_loss_w={model.plain_kd_loss_weight}")
        return model

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(self.test_loader,
                                              save_dir=self.save_dir,
                                              args=copy(self.args),
                                              _callbacks=self.callbacks)

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ("\n" + "%11s" * (5 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
            "ETA",
        )

    def plot_training_samples(self, batch, ni):
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        plot_results(file=self.csv, on_plot=self.on_plot)

    def plot_training_labels(self):
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def auto_batch(self):
        train_dataset = self.build_dataset(self.trainset, mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        return super().auto_batch(max_num_obj)
