import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import custom_fwd, custom_bwd
except Exception:
    from torch.cuda.amp import custom_fwd, custom_bwd


def _make_amp_decorator(deco, device_type: str = "cuda"):
    try:
        return deco(device_type=device_type)
    except TypeError:
        try:
            return deco()
        except TypeError:
            return deco


_CUSTOM_FWD = _make_amp_decorator(custom_fwd)
_CUSTOM_BWD = _make_amp_decorator(custom_bwd)

try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:
    triton, tl = None, None
    _TRITON_OK = False


class Conv2d_BN(nn.Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
    ):
        super().__init__()
        self.add_module("c", nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module("bn", nn.BatchNorm2d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class LKP(nn.Module):
    def __init__(self, dim, lks=7, sks=3, group_size=8):
        super().__init__()
        assert dim % 2 == 0
        assert dim % group_size == 0
        wc = dim // group_size
        self.cv1 = Conv2d_BN(dim, dim // 2)
        self.act = nn.ReLU(inplace=True)
        self.cv2 = Conv2d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv2d_BN(dim // 2, dim // 2)
        self.cv4 = nn.Conv2d(dim // 2, (sks * sks) * wc, kernel_size=1, bias=True)
        self.norm = nn.GroupNorm(num_groups=wc, num_channels=(sks * sks) * wc)
        self.sks = sks
        self.wc = wc

    def forward(self, x):
        x = self.act(self.cv1(x))
        x = self.act(self.cv2(x))
        x = self.act(self.cv3(x))
        w = self.norm(self.cv4(x))
        b, _, h, w_ = w.shape
        w = w.view(b, self.wc, self.sks * self.sks, h, w_)
        return w


def _ska_torch(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    ks2 = int(w.shape[2])
    ks = int(math.isqrt(ks2))
    assert ks * ks == ks2
    pad = (ks - 1) // 2
    b, c, h, wi = x.shape
    wc = w.shape[1]
    assert c % wc == 0
    g = c // wc

    xp = F.pad(x, (pad, pad, pad, pad))
    xp = xp.view(b, g, wc, h + 2 * pad, wi + 2 * pad)
    out = torch.zeros((b, g, wc, h, wi), device=x.device, dtype=x.dtype)

    kidx = 0
    for kh in range(ks):
        hs = kh
        he = kh + h
        for kw in range(ks):
            ws = kw
            we = kw + wi
            xs = xp[:, :, :, hs:he, ws:we]
            ww = w[:, None, :, kidx, :, :]
            out = out + xs * ww
            kidx += 1

    return out.view(b, c, h, wi)


if _TRITON_OK:

    def _grid(numel: int, bs: int) -> tuple:
        return (triton.cdiv(numel, bs),)


    @triton.jit
    def _idx(i, n: int, c: int, h: int, w: int):
        ni = i // (c * h * w)
        ci = (i // (h * w)) % c
        hi = (i // w) % h
        wi = i % w
        m = i < (n * c * h * w)
        return ni, ci, hi, wi, m


    @triton.jit
    def ska_fwd(
        x_ptr,
        w_ptr,
        o_ptr,
        n,
        ic,
        h,
        w,
        ks: tl.constexpr,
        pad: tl.constexpr,
        wc,
        BS: tl.constexpr,
        CT: tl.constexpr,
        AT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
        val = tl.zeros((BS,), dtype=AT)

        for kh in tl.static_range(0, ks):
            hin = hi - pad + kh
            hb = (hin >= 0) & (hin < h)
            for kw in tl.static_range(0, ks):
                win = wi - pad + kw
                b = hb & (win >= 0) & (win < w)

                x_off = ((ni * ic + ci) * h + hin) * w + win
                kidx = kh * ks + kw
                w_off = ((ni * wc + (ci % wc)) * (ks * ks) + kidx) * h * w + hi * w + wi

                x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(CT)
                w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
                val += tl.where(b & m, x_val * w_val, 0.0).to(AT)

        tl.store(o_ptr + offs, val.to(CT), mask=m)


    @triton.jit
    def ska_bwd_x(
        go_ptr,
        w_ptr,
        gi_ptr,
        n,
        ic,
        h,
        w,
        ks: tl.constexpr,
        pad: tl.constexpr,
        wc,
        BS: tl.constexpr,
        CT: tl.constexpr,
        AT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
        val = tl.zeros((BS,), dtype=AT)

        for kh in tl.static_range(0, ks):
            ho = hi + pad - kh
            hb = (ho >= 0) & (ho < h)
            for kw in tl.static_range(0, ks):
                wo = wi + pad - kw
                b = hb & (wo >= 0) & (wo < w)

                go_off = ((ni * ic + ci) * h + ho) * w + wo
                kidx = kh * ks + kw
                w_off = ((ni * wc + (ci % wc)) * (ks * ks) + kidx) * h * w + ho * w + wo

                go_val = tl.load(go_ptr + go_off, mask=m & b, other=0.0).to(CT)
                w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
                val += tl.where(b & m, go_val * w_val, 0.0).to(AT)

        tl.store(gi_ptr + offs, val.to(CT), mask=m)


    @triton.jit
    def ska_bwd_w(
        go_ptr,
        x_ptr,
        gw_ptr,
        n,
        wc,
        h,
        w,
        ic,
        ks: tl.constexpr,
        pad: tl.constexpr,
        G: tl.constexpr,
        BS: tl.constexpr,
        CT: tl.constexpr,
        AT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, wc, h, w)

        for kh in tl.static_range(0, ks):
            hin = hi - pad + kh
            hb = (hin >= 0) & (hin < h)
            for kw in tl.static_range(0, ks):
                win = wi - pad + kw
                b = hb & (win >= 0) & (win < w)

                kidx = kh * ks + kw
                w_off = ((ni * wc + ci) * (ks * ks) + kidx) * h * w + hi * w + wi

                acc = tl.zeros((BS,), dtype=AT)
                for s in tl.static_range(0, G):
                    cc = ci + s * wc
                    x_off = ((ni * ic + cc) * h + hin) * w + win
                    go_off = ((ni * ic + cc) * h + hi) * w + wi

                    x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(CT)
                    go_val = tl.load(go_ptr + go_off, mask=m & b, other=0.0).to(CT)
                    acc += (x_val * go_val).to(AT)

                tl.store(gw_ptr + w_off, acc.to(CT), mask=m)


    class _SkaFn(torch.autograd.Function):
        @staticmethod
        @_CUSTOM_FWD
        def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            ks2 = int(w.shape[2])
            ks = int(math.isqrt(ks2))
            assert ks * ks == ks2
            pad = (ks - 1) // 2
            n, ic, h, wi = x.shape
            wc = w.shape[1]
            assert ic % wc == 0
            g = ic // wc

            o = torch.empty((n, ic, h, wi), device=x.device, dtype=x.dtype)
            x = x.contiguous()
            w = w.contiguous()

            numel = o.numel()
            grid = lambda meta: _grid(numel, meta["BS"])

            ct = tl.float16 if x.dtype == torch.float16 else (tl.float32 if x.dtype == torch.float32 else tl.float64)
            at = tl.float32 if x.dtype == torch.float16 else ct

            ska_fwd[grid](x, w, o, n, ic, h, wi, ks=ks, pad=pad, wc=wc, BS=1024, CT=ct, AT=at)

            ctx.save_for_backward(x, w)
            ctx.ks, ctx.pad, ctx.wc, ctx.ic, ctx.g = ks, pad, wc, ic, g
            ctx.ct, ctx.at = ct, at
            return o

        @staticmethod
        @_CUSTOM_BWD
        def backward(ctx, go: torch.Tensor):
            x, w = ctx.saved_tensors
            ks, pad, wc, ic, g = ctx.ks, ctx.pad, ctx.wc, ctx.ic, ctx.g
            n, _, h, wi = x.shape
            go = go.contiguous()

            gx = gw = None
            ct, at = ctx.ct, ctx.at

            if ctx.needs_input_grad[0]:
                gx = torch.empty_like(x)
                numel = gx.numel()
                ska_bwd_x[lambda meta: _grid(numel, meta["BS"]) ](
                    go, w, gx, n, ic, h, wi, ks=ks, pad=pad, wc=wc, BS=1024, CT=ct, AT=at
                )

            if ctx.needs_input_grad[1]:
                gw = torch.empty_like(w)
                numel = n * wc * h * wi
                ska_bwd_w[lambda meta: _grid(numel, meta["BS"]) ](
                    go, x, gw, n, wc, h, wi, ic, ks=ks, pad=pad, G=g, BS=1024, CT=ct, AT=at
                )

            return gx, gw


class SKA(nn.Module):
    def __init__(self, prefer_triton: bool = True):
        super().__init__()
        self.prefer_triton = prefer_triton

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if self.prefer_triton and _TRITON_OK and x.is_cuda and w.is_cuda:
            return _SkaFn.apply(x, w)
        return _ska_torch(x, w)


class LSConv(nn.Module):
    def __init__(self, c1, c2=None, lks=7, sks=3, group_size=8, prefer_triton: bool = True):
        super().__init__()
        c2 = c1 if c2 is None else c2
        assert c2 % group_size == 0
        self.short = nn.Identity() if c1 == c2 else Conv2d_BN(c1, c2, ks=1)
        self.lkp = LKP(c2, lks=lks, sks=sks, group_size=group_size)
        self.ska = SKA(prefer_triton=prefer_triton)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        y = self.short(x)
        return self.bn(self.ska(y, self.lkp(y))) + y


__all__ = ("Conv2d_BN", "LKP", "SKA", "LSConv")
