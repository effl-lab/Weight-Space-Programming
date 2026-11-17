import torch


def bag(w, ids):
    """(..., L) word ids -> (..., D): mean of the word embeddings, padding (id 0) left out."""
    real = (ids != 0).float()[..., None]
    return (w["embed"][ids] * real).sum(-2) / real.sum(-2).clamp(min=1)


def forward(w, q, o):
    """Bilinear match between the question's bag and each option's bag, plus a score of the option alone."""
    qe, oe = bag(w, q), bag(w, o)
    return torch.einsum("nd,de,nke->nk", qe, w["W"], oe) + oe @ w["u"]
