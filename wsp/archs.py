"""Small CIFAR-10 classifiers defined as functional forwards over a flat weight dict.

mlp / cnn / vit are the standard architectures. roles / basis / rule are designed to be written by
hand: they provide the colour -> local filter -> named-region pooling -> score-table pipeline that
writers built inside the standard architectures, with progressively fewer numbers to write.
ratio is roles with the score reading log ratios between regions instead of region means.
white is ratio with the score given as one prototype feature vector per class, read through a
covariance that the forward measures on fixed unlabelled training images.

The same forward is used for gradient-trained baselines and for writer-built weights,
so a weights.npz from either source is evaluated by identical code.
Input: float32 images in [0, 1], shape (N, 3, 32, 32), no normalization. Output: (N, 10) logits.
"""
import importlib
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from wsp import free_model as free
from wsp import distill, given_archs, lineage, lm_start, olmo2_1b, olmo_start
from wsp import memory as mem
from wsp import text_archs as ta
from wsp import vision_tokens as vt
from wsp import vision_words as vw

VIT_DIM = 64
VIT_HEADS = 4
VIT_DEPTH = 2
VIT_PATCH = 4
VIT_MLP = 128


def _mlp_shapes():
    return {
        "fc1.weight": (64, 3072), "fc1.bias": (64,),
        "fc2.weight": (64, 64), "fc2.bias": (64,),
        "fc3.weight": (10, 64), "fc3.bias": (10,),
    }


def _mlp_forward(w, x):
    h = F.relu(F.linear(x.flatten(1), w["fc1.weight"], w["fc1.bias"]))
    h = F.relu(F.linear(h, w["fc2.weight"], w["fc2.bias"]))
    return F.linear(h, w["fc3.weight"], w["fc3.bias"])


def _cnn_shapes():
    return {
        "conv1.weight": (16, 3, 3, 3), "conv1.bias": (16,),
        "conv2.weight": (32, 16, 3, 3), "conv2.bias": (32,),
        "conv3.weight": (64, 32, 3, 3), "conv3.bias": (64,),
        "fc.weight": (10, 64), "fc.bias": (10,),
    }


def _cnn_forward(w, x):
    h = F.max_pool2d(F.relu(F.conv2d(x, w["conv1.weight"], w["conv1.bias"], padding=1)), 2)
    h = F.max_pool2d(F.relu(F.conv2d(h, w["conv2.weight"], w["conv2.bias"], padding=1)), 2)
    h = F.relu(F.conv2d(h, w["conv3.weight"], w["conv3.bias"], padding=1))
    return F.linear(h.mean(dim=(2, 3)), w["fc.weight"], w["fc.bias"])


def _vit_shapes():
    d, n_tok = VIT_DIM, (32 // VIT_PATCH) ** 2 + 1
    shapes = {
        "patch.weight": (d, 3 * VIT_PATCH * VIT_PATCH), "patch.bias": (d,),
        "cls": (d,), "pos": (n_tok, d),
    }
    for i in range(VIT_DEPTH):
        p = f"block{i}."
        shapes.update({
            p + "ln1.weight": (d,), p + "ln1.bias": (d,),
            p + "q.weight": (d, d), p + "q.bias": (d,),
            p + "k.weight": (d, d), p + "k.bias": (d,),
            p + "v.weight": (d, d), p + "v.bias": (d,),
            p + "o.weight": (d, d), p + "o.bias": (d,),
            p + "ln2.weight": (d,), p + "ln2.bias": (d,),
            p + "fc1.weight": (VIT_MLP, d), p + "fc1.bias": (VIT_MLP,),
            p + "fc2.weight": (d, VIT_MLP), p + "fc2.bias": (d,),
        })
    shapes.update({"ln_f.weight": (d,), "ln_f.bias": (d,),
                   "head.weight": (10, d), "head.bias": (10,)})
    return shapes


def _patchify(x):
    # (N, 3, 32, 32) -> (N, 64, 48); patch row-major over the 8x8 grid, features ordered (c, dy, dx)
    n, p = x.shape[0], VIT_PATCH
    x = x.reshape(n, 3, 32 // p, p, 32 // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(n, (32 // p) ** 2, 3 * p * p)


def _vit_forward(w, x):
    d, hd = VIT_DIM, VIT_DIM // VIT_HEADS
    t = F.linear(_patchify(x), w["patch.weight"], w["patch.bias"])
    t = torch.cat([w["cls"].expand(t.shape[0], 1, d), t], dim=1) + w["pos"]
    for i in range(VIT_DEPTH):
        p = f"block{i}."
        h = F.layer_norm(t, (d,), w[p + "ln1.weight"], w[p + "ln1.bias"])
        q, k, v = (F.linear(h, w[p + n + ".weight"], w[p + n + ".bias"])
                   .reshape(h.shape[0], h.shape[1], VIT_HEADS, hd).transpose(1, 2) for n in "qkv")
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), dim=-1) @ v
        t = t + F.linear(a.transpose(1, 2).reshape(t.shape), w[p + "o.weight"], w[p + "o.bias"])
        h = F.layer_norm(t, (d,), w[p + "ln2.weight"], w[p + "ln2.bias"])
        t = t + F.linear(F.gelu(F.linear(h, w[p + "fc1.weight"], w[p + "fc1.bias"])),
                         w[p + "fc2.weight"], w[p + "fc2.bias"])
    cls = F.layer_norm(t[:, 0], (d,), w["ln_f.weight"], w["ln_f.bias"])
    return F.linear(cls, w["head.weight"], w["head.bias"])


REGIONS = ("all", "top", "bottom", "middle", "centre", "border")


def region_masks(size: int) -> torch.Tensor:
    """(6, size, size) averaging masks, each summing to 1, in the order of REGIONS.

    top / bottom: first / last quarter of rows; middle: the rows between them;
    centre: the central half in both directions; border: everything outside the centre.
    """
    q = size // 4
    m = torch.zeros(len(REGIONS), size, size)
    m[0] = 1
    m[1, :q] = 1
    m[2, size - q:] = 1
    m[3, q:size - q] = 1
    m[4, q:size - q, q:size - q] = 1
    m[5] = 1 - m[4]
    return m / m.sum(dim=(1, 2), keepdim=True)


def region_pool(h: torch.Tensor) -> torch.Tensor:
    """(N, C, S, S) -> (N, C * 6): feature c in region r lands at index c * 6 + r."""
    return torch.einsum("ncij,rij->ncr", h, region_masks(h.shape[-1]).to(h)).flatten(1)


def _roles_shapes():
    return {
        "colour.weight": (8, 3, 1, 1), "colour.bias": (8,),
        "local1.weight": (24, 8, 3, 3), "local1.bias": (24,),
        "local2.weight": (32, 24, 3, 3), "local2.bias": (32,),
        "score.weight": (10, 32 * len(REGIONS)), "score.bias": (10,),
    }


def pad_edge(h: torch.Tensor, p: int) -> torch.Tensor:
    """Pads by repeating edge pixels, so filters see no artificial edge at the image border."""
    return F.pad(h, (p, p, p, p), mode="replicate")


def _roles_forward(w, x):
    h = F.conv2d(x, w["colour.weight"], w["colour.bias"])          # linear colour transform
    h = F.relu(F.conv2d(pad_edge(h, 1), w["local1.weight"], w["local1.bias"]))
    h = F.relu(F.conv2d(pad_edge(F.avg_pool2d(h, 2), 1), w["local2.weight"], w["local2.bias"]))
    return F.linear(region_pool(h), w["score.weight"], w["score.bias"])


LOG_FLOOR = 0.01
CONTRASTS = ("all", "top-bottom", "centre-border", "middle-ends")


def log_contrast_pool(h: torch.Tensor) -> torch.Tensor:
    """(N, C, S, S) non-negative maps -> (N, C * 4): feature c, contrast k lands at index c * 4 + k.

    g[c, r] = log(mean of map c over region r + LOG_FLOOR), natural log, regions as in REGIONS. Per channel:
    all = g[c, all] minus the mean of g[., all] over the C channels of the same image (every channel,
    including ones that are always zero);
    top-bottom = g[c, top] - g[c, bottom]; centre-border = g[c, centre] - g[c, border];
    middle-ends = g[c, middle] - (g[c, top] + g[c, bottom]) / 2.
    Where region means are well above LOG_FLOOR the three contrasts are log ratios, which a constant
    factor on the map barely changes; where they are near or below LOG_FLOOR, the floor dominates.
    """
    g = torch.log(region_pool(h).reshape(h.shape[0], h.shape[1], len(REGIONS)) + LOG_FLOOR)
    a, t, b, m, c, o = g.unbind(-1)
    return torch.stack([a - a.mean(1, keepdim=True), t - b, c - o, m - (t + b) / 2], -1).flatten(1)


def _ratio_shapes():
    return _roles_shapes() | {"score.weight": (10, 32 * len(CONTRASTS))}


def _ratio_forward(w, x):
    h = F.conv2d(x, w["colour.weight"], w["colour.bias"])          # linear colour transform
    h = F.relu(F.conv2d(pad_edge(h, 1), w["local1.weight"], w["local1.bias"]))
    h = F.relu(F.conv2d(pad_edge(F.avg_pool2d(h, 2), 1), w["local2.weight"], w["local2.bias"]))
    return F.linear(log_contrast_pool(h), w["score.weight"], w["score.bias"])


CALIB_STEP = 50
SHRINK = 0.7
RIDGE = 3e-2
_CALIB: dict = {}


def calibration_images(device: torch.device) -> torch.Tensor:
    """Every CALIB_STEP-th training image (1,000 images), labels unused; the same for every network."""
    if device not in _CALIB:
        from wsp import data  # writer sandboxes import this module without the data stack
        with np.load(data.CIFAR_DIR / "train.npz") as z:
            x = np.ascontiguousarray(z["x"][::CALIB_STEP]).astype(np.float32) / 255.0
        _CALIB[device] = torch.from_numpy(x).to(device)
    return _CALIB[device]


def prototype_readout(f: torch.Tensor, f_calib: torch.Tensor, proto: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Linear discriminant scores of features f (N, F) against class prototypes proto (10, F).

    proto[c] is the typical feature vector of class c, in the units of the feature output (here the
    output of _ratio_features); only proto[c] - m0 matters. Features with a small spread over the
    calibration images get large weights, so an error in proto there moves the scores most.

    m0 and cov are the mean and covariance of the features over the calibration images. cov is shrunk
    towards its diagonal, S = (1 - SHRINK) * cov + SHRINK * diag(cov) + RIDGE * mean(diag(cov)) * I, and
    score_c = (f - m0) . S^-1 (p_c - m0) - (p_c - m0) . S^-1 (p_c - m0) / 2 + bias_c.
    Features that are constant over the calibration images are ignored.
    """
    m0 = f_calib.mean(0)
    d = f_calib - m0
    cov = d.T @ d / (len(f_calib) - 1)
    var = cov.diagonal()
    keep = (var > 1e-10).to(f.dtype)
    eye = torch.eye(len(var), dtype=f.dtype, device=f.device)
    s = (1 - SHRINK) * cov + SHRINK * torch.diag(var) + RIDGE * var.mean() * eye
    s = s * keep[:, None] * keep[None, :] + torch.diag(1 - keep)
    q = (proto - m0) * keep
    a = torch.linalg.solve(s, q.T)
    return (f - m0) @ a - 0.5 * (q * a.T).sum(1) + bias


def _ratio_features(w, x):
    h = F.conv2d(x, w["colour.weight"], w["colour.bias"])          # linear colour transform
    h = F.relu(F.conv2d(pad_edge(h, 1), w["local1.weight"], w["local1.bias"]))
    h = F.relu(F.conv2d(pad_edge(F.avg_pool2d(h, 2), 1), w["local2.weight"], w["local2.bias"]))
    return log_contrast_pool(h)


def _white_shapes():
    shapes = _ratio_shapes()
    del shapes["score.weight"]
    return shapes | {"prototype": (10, 32 * len(CONTRASTS))}


def _white_init(g: torch.Generator) -> dict[str, torch.Tensor]:
    """Equal prototypes so that gradient training starts from uniform logits, not whitened random ones."""
    return {"prototype": torch.zeros(10, 32 * len(CONTRASTS))}


def _white_forward(w, x):
    f = _ratio_features(w, x)
    f_calib = _ratio_features(w, calibration_images(x.device))
    return prototype_readout(f, f_calib, w["prototype"], w["score.bias"])


def basis_bank() -> torch.Tensor:
    """Fixed (27, 3, 5, 5) filter bank: 3 colour channels x 9 spatial filters.

    Colour channels: L = (R+G+B)/3, RG = R-G, BY = B-(R+G)/2.
    Spatial filters, in order: Gaussian blur (sigma 1, sums to 1); then for angles 0, 45, 90, 135 degrees
    an even and an odd Gabor (sigma 1.5, wavelength 4; each has zero sum and absolute values summing to 1).
    Coordinates: x is the column (to the right), y is the row (downward). The angle is the direction in
    which intensity changes: 0 degrees responds to vertical edges / stripes, 90 degrees to horizontal ones,
    45 degrees to edges running from bottom-left to top-right (a forward slash) and 135 degrees to edges
    running from top-left to bottom-right (a backslash). Orientation tuning is broad: a diagonal edge also
    drives the 0 and 90 degree filters about as strongly; the 45 / 135 pair tells the two diagonals apart.
    The odd Gabor at angle t is positive where intensity increases along direction t.
    Output map k = colour k // 9, filter k % 9. Filters are applied by cross-correlation (no flip).
    """
    colours = torch.tensor([[1 / 3, 1 / 3, 1 / 3], [1.0, -1.0, 0.0], [-0.5, -0.5, 1.0]])
    r = torch.arange(-2, 3, dtype=torch.float32)
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    blur = torch.exp(-(xx ** 2 + yy ** 2) / 2)
    spatial = [blur / blur.sum()]
    for deg in (0, 45, 90, 135):
        t = math.radians(deg)
        u = xx * math.cos(t) + yy * math.sin(t)
        env = torch.exp(-(xx ** 2 + yy ** 2) / (2 * 1.5 ** 2))
        even = env * torch.cos(2 * math.pi * u / 4)
        even = even - even.mean()
        odd = env * torch.sin(2 * math.pi * u / 4)
        spatial += [even / even.abs().sum(), odd / odd.abs().sum()]
    return torch.stack([c[:, None, None] * s for c in colours for s in spatial])


def _basis_shapes():
    return {"mix.weight": (32, 27, 1, 1), "mix.bias": (32,),
            "score.weight": (10, 32 * len(REGIONS)), "score.bias": (10,)}


def _basis_forward(w, x):
    r = F.conv2d(pad_edge(x, 2), basis_bank().to(x))
    blur = torch.arange(27, device=x.device) % 9 == 0
    r = torch.where(blur[None, :, None, None], r, r.abs())   # Gabor responses rectified (abs), blur keeps sign
    h = F.relu(F.conv2d(F.avg_pool2d(r, 2), w["mix.weight"], w["mix.bias"]))
    return F.linear(region_pool(h), w["score.weight"], w["score.bias"])


RULE_FILTERS = 24
RULE_SIZE = 7


def rule_kernels(p: torch.Tensor) -> torch.Tensor:
    """Expands (24, 9) filter rules into (24, 3, 7, 7) kernels.

    Row = [theta, sigma, wavelength, phase, colour_R, colour_G, colour_B, gain, bias]; the kernel is
    gain * colour (outer) exp(-(x^2+y^2) / (2 sigma^2)) * cos(2 pi u / wavelength + phase),
    u = x cos(theta) + y sin(theta), x to the right (column), y downward (row), pixel offsets -3..3.
    theta is the direction in which the grating varies: theta 0 responds to vertical edges / stripes.
    |sigma| is clamped to >= 0.3 and |wavelength| to >= 2. Kernels are applied by cross-correlation
    (no flip) on the image padded by repeating its edge pixels.
    The bias column is applied after the convolution.
    """
    r = torch.arange(RULE_SIZE, dtype=p.dtype, device=p.device) - RULE_SIZE // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    theta, sigma, lam, phase = (p[:, i, None, None] for i in range(4))
    sigma, lam = sigma.abs().clamp(min=0.3), lam.abs().clamp(min=2.0)
    u = xx * torch.cos(theta) + yy * torch.sin(theta)
    spatial = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)) * torch.cos(2 * math.pi * u / lam + phase)
    return p[:, 7, None, None, None] * p[:, 4:7, None, None] * spatial[:, None]


def _rule_shapes():
    return {"rules": (RULE_FILTERS, 9), "score.weight": (10, RULE_FILTERS * len(REGIONS)), "score.bias": (10,)}


def _rule_forward(w, x):
    h = F.conv2d(pad_edge(x, RULE_SIZE // 2), rule_kernels(w["rules"]), w["rules"][:, 8])
    h = F.avg_pool2d(F.relu(h), 2)
    return F.linear(region_pool(h), w["score.weight"], w["score.bias"])


def _rule_init(g: torch.Generator) -> dict[str, torch.Tensor]:
    n = RULE_FILTERS
    rules = torch.stack([
        torch.rand(n, generator=g) * math.pi, torch.full((n,), 1.5), torch.full((n,), 4.0),
        torch.rand(n, generator=g) * math.pi, *(torch.randn(3, n, generator=g) * 0.5),
        torch.ones(n), torch.zeros(n)], dim=1)
    return {"rules": rules}


@dataclass(frozen=True)
class Arch:
    shapes: Callable[[], dict]
    forward: Callable
    shown: tuple = ()  # helpers whose source the writer sees with the forward
    init: Callable | None = field(default=None)  # custom init for gradient baselines
    consts: tuple = ()  # module constants the shown source refers to
    task: str = "cifar"  # key in wsp.tasks.TASKS: data, classes and input format
    deliverable: str = "build_weights.py"  # what the writer leaves in /srv/work


ARCHS = {
    "mlp": Arch(_mlp_shapes, _mlp_forward),
    "cnn": Arch(_cnn_shapes, _cnn_forward),
    "vit": Arch(_vit_shapes, _vit_forward, shown=(_patchify,),
                consts=("VIT_DIM", "VIT_HEADS", "VIT_DEPTH", "VIT_PATCH", "VIT_MLP")),
    "roles": Arch(_roles_shapes, _roles_forward, shown=(pad_edge, region_masks, region_pool), consts=("REGIONS",)),
    "ratio": Arch(_ratio_shapes, _ratio_forward, shown=(pad_edge, region_masks, region_pool, log_contrast_pool),
                  consts=("REGIONS", "LOG_FLOOR", "CONTRASTS")),
    "white": Arch(_white_shapes, _white_forward, init=_white_init,
                  shown=(_ratio_features, pad_edge, region_masks, region_pool, log_contrast_pool, calibration_images,
                         prototype_readout),
                  consts=("REGIONS", "LOG_FLOOR", "CONTRASTS", "CALIB_STEP", "SHRINK", "RIDGE")),
    "bow": Arch(ta._bow_shapes, ta._bow_forward, shown=(ta.token_counts,), consts=("TEXT_VOCAB",), task="agnews"),
    "fasttext": Arch(ta._fasttext_shapes, ta._fasttext_forward, task="agnews"),
    "cnn1d": Arch(ta._cnn1d_shapes, ta._cnn1d_forward, task="agnews"),
    "transformer": Arch(ta._transformer_shapes, ta._transformer_forward,
                        consts=("TF_DIM", "TF_HEADS"), task="agnews"),
    "cbow": Arch(ta._bow_shapes, ta._bow_forward, shown=(ta.token_counts,), consts=("TEXT_VOCAB",),
                 task="agnews_cipher"),
    "cfasttext": Arch(ta._fasttext_shapes, ta._fasttext_forward, task="agnews_cipher"),
    "ccnn1d": Arch(ta._cnn1d_shapes, ta._cnn1d_forward, task="agnews_cipher"),
    "ctransformer": Arch(ta._transformer_shapes, ta._transformer_forward, consts=("TF_DIM", "TF_HEADS"),
                         task="agnews_cipher"),
    "vwords": Arch(vw._vwords_shapes, vw._vwords_forward,
                   shown=(vw.word_counts, vw.patch_rows, vw.quadrant_of_patch, vw.codebook),
                   consts=("VW_PATCH", "VW_STRIDE", "VW_SIDE", "VW_WORDS", "VW_QUADRANTS", "NORM_EPS", "SOFT_TEMP"),
                   task="cifar_words"),
    "vtok": Arch(vt._vtok_shapes, vt._vtok_forward, shown=(vt.token_counts,), consts=("VT_VOCAB",),
                 task="cifar_tokens"),
    "vemb": Arch(vt._vemb_shapes, vt._vemb_forward, task="cifar_emb"),
    "vwords_gd": Arch(vw._vwords_gd_shapes, vw._vwords_gd_forward, task="cifar_words"),
    "basis": Arch(_basis_shapes, _basis_forward, shown=(basis_bank, pad_edge, region_masks, region_pool),
                  consts=("REGIONS",)),
    "rule": Arch(_rule_shapes, _rule_forward, shown=(rule_kernels, pad_edge, region_masks, region_pool),
                 init=_rule_init,
                 consts=("RULE_FILTERS", "RULE_SIZE", "REGIONS")),
}
ARCHS.update({f"mem-{s}": Arch(mem._mem_shapes, mem.forward, shown=(mem.build, mem.bags, mem.parse, mem.words),
                               consts=("THETA", "MAX_LINES", "MAX_LINE_CHARS"), task=f"mmlu-{s}",
                               deliverable=mem.DELIVERABLE) for s in mem.SUBJECTS})
# shapes and forward of a free-design model are the writer's own; free.run builds and runs its folder
ARCHS.update({f"free-{s}": Arch(dict, free.run, task=f"mmlu-{s}", deliverable=free.DELIVERABLE) for s in mem.SUBJECTS})
# ARC: the writer designs the model (free-arc and its variants, wsp.lineage.ARC_VARIANTS) or writes the weights of a
# fixed standard arch.py (<name>-arc)
ARCHS.update({f"{n}-arc": Arch(dict, free.run, task="arc", deliverable=free.DELIVERABLE)
              for n in ("free", *given_archs.NAMES, *given_archs.MEMORY, *lineage.ARC_VARIANTS)})
ARCHS.update({lineage.variant_arch(v, g): Arch(dict, free.run, task="arc", deliverable=free.DELIVERABLE)
              for v in lineage.LINEAGE_VARIANTS for g in range(2, lineage.MAX_GENERATION + 1)})
ARCHS.update({lineage.given_arch(n, g): Arch(dict, free.run, task="arc", deliverable=free.DELIVERABLE)
              for n in lineage.GIVEN_LINEAGES for g in range(2, lineage.MAX_GENERATION + 1)})
# MathBench: the writer designs the model (free-mb) or writes the weights of a given arch.py (<name>-mb)
ARCHS.update({f"{n}-mb": Arch(dict, free.run, task="mathbench", deliverable=free.DELIVERABLE)
              for n in ("free", "mem_assoc2", "transformer", "transformer_d8", "transformer_d24")})
# MMLU over all subjects: the writer designs the model (free-mm)
ARCHS["free-mm"] = Arch(dict, free.run, task="mmlu_all", deliverable=free.DELIVERABLE)
# ARC distillation baseline: the writer writes training data (wsp.distill), students are trained on it by GD
ARCHS["distill-arc"] = ARCHS["distill100-arc"] = Arch(dict, None, task="arc", deliverable=distill.DELIVERABLE)
# SQuAD 2.0: the writer designs the model (free-sq) or writes the weights of the given transformer (transformer-sq);
# wsp.free_model.run_spans, forward(w, q, c)
ARCHS.update({a: Arch(dict, free.run, task="squad", deliverable=free.DELIVERABLE) for a in ("free-sq", "transformer-sq")})
# SQuAD 1.1 (every item has an answer; column 0 of the output is ignored): free-sq1, transformer-sq1
ARCHS.update({a: Arch(dict, free.run, task="squad1", deliverable=free.DELIVERABLE) for a in ("free-sq1", "transformer-sq1")})
# SciQ answered in words, generated one word at a time (wsp.free_model.run_gen, forward(w, q, a)): free-gen,
# transformer-gen
ARCHS.update({a: Arch(dict, free.run, task="sciq_gen", deliverable=free.DELIVERABLE) for a in ("free-gen", "transformer-gen")})
# the general task (wsp.general): text in, words out, one word at a time (wsp.free_model.run_all, forward(w, x, a))
ARCHS["design-all"] = Arch(dict, free.run, task="general", deliverable=free.DELIVERABLE)
# a language model written by hand (wsp.textlm): a text in pieces, the next piece out (wsp.free_model.run_lm, forward(w, x))
ARCHS["design-text"] = Arch(dict, free.run, task="text", deliverable=free.DELIVERABLE)
# CIFAR-10 images: the writer designs the model (wsp.free_model.run_images, forward(w, x))
ARCHS["design-cifar"] = Arch(dict, free.run, task="cifar", deliverable=free.DELIVERABLE)
# later generations of a free-design model start from the previous generation's model (wsp.lineage)
ARCHS.update({lineage.arch(g, t): Arch(dict, free.run, task="arc" if t == "arc" else f"mmlu-{t}",
                                       deliverable=free.DELIVERABLE)
              for g in range(2, lineage.MAX_GENERATION + 1) for t in (*mem.SUBJECTS, "arc")})
# ARC through a released small LM (lm-<name>-<start>): the writer edits the weights of the fixed wsp/lm_archs/<name>.py,
# starting from the base checkpoint or a random initialisation (wsp.lm_start)
ARCHS.update({a: Arch(dict, importlib.import_module(f"wsp.lm_archs.{lm_start.parse(a)[0]}").forward, task="arc",
                      deliverable=free.DELIVERABLE) for a in lm_start.ARCHS})
# OLMo 2 1B edited by hand as the start of pretraining or of SFT (wsp.olmo_start); scored by harness.olmo_score
ARCHS.update({a: Arch(dict, olmo2_1b.logits, task="arc", deliverable=free.DELIVERABLE) for a in olmo_start.ARCHS})
# A written network re-expressed in a larger architecture that computes the same function, for fine-tuning.
LIFTS = {"vwords": ("vwords_gd", vw.with_codebook)}
STANDARD = ("mlp", "cnn", "vit")
WRITABLE = ("roles", "ratio", "white", "basis", "rule")


def shapes(arch: str) -> dict[str, tuple[int, ...]]:
    return ARCHS[arch].shapes()


def task_of(arch: str):
    from wsp import tasks  # tasks loads the data stack, which writer sandboxes do not need
    return tasks.TASKS[ARCHS[arch].task]


def n_params(arch: str) -> int:
    return sum(math.prod(s) for s in shapes(arch).values())


def forward(arch: str, weights: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    return ARCHS[arch].forward(weights, x)


def validate(arch: str, arrays: dict[str, np.ndarray]) -> list[str]:
    """Returns a list of problems; empty means the weights match the spec exactly."""
    expected = shapes(arch)
    problems = [f"missing: {k}" for k in expected if k not in arrays]
    problems += [f"unexpected: {k}" for k in arrays if k not in expected]
    for k, s in expected.items():
        if k not in arrays:
            continue
        a = arrays[k]
        if tuple(a.shape) != s:
            problems.append(f"{k}: shape {tuple(a.shape)} != {s}")
        elif not np.issubdtype(a.dtype, np.floating) and not np.issubdtype(a.dtype, np.integer):
            problems.append(f"{k}: non-numeric dtype {a.dtype}")
        elif not np.all(np.isfinite(a)):
            problems.append(f"{k}: non-finite values")
    return problems


def init_weights(arch: str, seed: int) -> dict[str, torch.Tensor]:
    """LeCun-uniform init (variance 1/fan_in) for gradient baselines."""
    g = torch.Generator().manual_seed(seed)
    out = ARCHS[arch].init(g) if ARCHS[arch].init else {}
    for k, s in shapes(arch).items():
        if k in out:
            continue
        if k.endswith("ln1.weight") or k.endswith("ln2.weight") or k == "ln_f.weight":
            out[k] = torch.ones(s)
        elif len(s) == 1 and k not in ("cls",):
            out[k] = torch.zeros(s)
        elif k in ("cls", "pos"):
            out[k] = torch.randn(s, generator=g) * 0.02
        else:
            fan_in = math.prod(s[1:])
            bound = 1.0 / math.sqrt(fan_in)
            out[k] = (torch.rand(s, generator=g) * 2 - 1) * bound * math.sqrt(3)
    return out
