import torch

TOPK = 5
COV_P = 4.0
PRF_K = 3
WH_SPAN = 7


def support(M, norm, qw, ow, n):
    """Per fact row: (question idf mass matched) x (new option idf mass matched). Returns (n, 4, rows)."""
    Qf = torch.sparse.mm(M, qw.t()).t() * norm[None, :]                    # (n, R)
    Of = torch.sparse.mm(M, ow.t()).t().reshape(n, 4, -1)                  # (n, 4, R)
    return Qf[:, None, :] * Of


def forward(w, q, o):
    n = q.shape[0]
    idf = w["idf"]
    C = idf.shape[0]

    # ---- question bag: concept -> weight (idf, boosted for the last tokens of the question)
    qc = w["concept"][q]                                   # (n, 128)
    real = (q != 0).float()
    L = real.sum(1, keepdim=True)
    pos = torch.arange(q.shape[1], dtype=torch.float32)[None, :]
    late = (pos >= L - w["late_n"]).float() * real
    tokw = real * (1.0 + w["late_boost"] * late)
    qbag = torch.zeros(n, C).scatter_reduce(1, qc, tokw, reduce="amax", include_self=True)
    qpres = (qbag > 0).float()
    qexp = torch.sparse.mm(w["Et"], qpres.t()).t().clamp(max=1.0) * (1.0 - qpres)
    qw = (qbag + w["qisa"] * qexp) * idf ** w["q_idf_pow"]

    # ---- option bags: concepts not already in the question, plus their categories
    oc = w["concept"][o].reshape(n * 4, o.shape[2])
    obag = torch.zeros(n * 4, C).scatter_(1, oc, 1.0)
    qrep = qpres.repeat_interleave(4, 0)
    obag = obag * (1.0 - qrep)
    oexp = torch.sparse.mm(w["Et"], obag.t()).t().clamp(max=1.0) * (1.0 - obag) * (1.0 - qrep)
    otot = (obag * idf).sum(1).reshape(n, 4)
    ow = obag * idf
    owe = oexp * idf
    scale = (otot[:, :, None] + 1.0) ** w["opt_len_pow"]

    # ---- order of directional concepts (energy forms, states of matter): pairs (a before b) in the
    # question, in each option, and question->option; facts that state the same order gain, reversed lose
    ND = w["dir_eye"].shape[0]
    ohq = torch.nn.functional.one_hot(w["dir_of"][qc], ND + 1)[..., :ND].float() * real[:, :, None]
    beforeq = torch.cumsum(ohq, 1) - ohq
    Pq = torch.einsum("nla,nlb->nab", beforeq, ohq).clamp(max=1.0)                  # (n, ND, ND)
    oreal = (oc != 0).float()
    oho = torch.nn.functional.one_hot(w["dir_of"][w["concept"][o].reshape(n * 4, -1)], ND + 1)[..., :ND].float()
    oho = oho * oreal[:, :, None]
    beforeo = torch.cumsum(oho, 1) - oho
    Po = torch.einsum("nla,nlb->nab", beforeo, oho).clamp(max=1.0).reshape(n, 4, ND, ND)
    qhas = ohq.amax(1)                                                               # (n, ND)
    ohas = oho.amax(1).reshape(n, 4, ND)
    Pqo = qhas[:, None, :, None] * ohas[:, :, None, :]
    qcue = (w["qcue"][q].sum(1) > 0).float()[:, None, None, None]                       # (n,1,1,1)
    ocue = (w["ocue"][o].sum(2) > 0).float()[:, :, None, None]                          # (n,4,1,1)
    Pq = Pq[:, None] * qcue
    Pqo = Pqo * qcue
    Po = Po * ocue
    eye = w["dir_eye"][None, None]
    P = ((Pq + Po + Pqo).clamp(max=1.0) * (1.0 - eye)).reshape(n * 4, ND * ND)
    Prev = P.reshape(n * 4, ND, ND).transpose(1, 2).reshape(n * 4, ND * ND)
    agree = torch.sparse.mm(w["FP"], P.t()).t() - torch.sparse.mm(w["FP"], Prev.t()).t()   # (4n, Nf)
    dirf = torch.exp(w["dir"] * agree.clamp(-2.0, 2.0)).reshape(n, 4, -1)

    # ---- fact retrieval: a fact supports an option when it shares words with both question and option;
    # the option's categories (isa.txt) give a weaker second route
    gate = (w["isa"] * torch.exp(-otot))[:, :, None]                      # categories matter for unknown words

    def score_with(qv):
        total = torch.zeros(n, 4)
        for ob, wt in ((ow, 1.0), (owe, gate[:, :, 0])):
            s = support(w["F"], w["F_norm"], qv, ob, n) / scale * dirf
            top = s.topk(TOPK, dim=2).values
            sw = support(w["Fw"], w["Fw_norm"], qv, ob, n) / scale
            sp = support(w["Fp"], w["Fp_norm"], qv, ob, n) / scale
            total = total + wt * (top[:, :, 0] + w["second"] * top[:, :, 1:].sum(2) + w["win"] * sw.amax(2)
                                  + w["par"] * sp.amax(2))
        return total

    # ---- question words that link equally to all four options cannot tell them apart: weight each question
    # word by how unevenly it links to the options (soft best fact per word and option)
    OfA = torch.sparse.mm(w["F"], ow.t()).t() * w["F_norm"][None, :]                   # (4n, Nf)
    link = (torch.sparse.mm(w["F"].t(), (OfA ** COV_P).t()).t() ** (1.0 / COV_P)).reshape(n, 4, C)
    spread = 1.0 - link.mean(1) / (link.amax(1) + 1e-6)                                 # (n, C) in [0, 3/4]
    disc = w["disc_floor"] + (1.0 - w["disc_floor"]) * (spread / 0.75)
    qw = qw * disc

    score = score_with(qw)

    # ---- evidence collected across facts: for each question word, the (soft) best fact linking it to the
    # option; an option tied to more of the question's words through different facts gains
    def coverage(qv):
        Of = torch.sparse.mm(w["F"], ow.t()).t() * w["F_norm"][None, :]          # (4n, Nf)
        link = torch.sparse.mm(w["F"].t(), (Of ** COV_P).t()).t() ** (1.0 / COV_P)  # (4n, C)
        qrep4 = qv.repeat_interleave(4, 0)
        return ((link * qrep4).sum(1).reshape(n, 4)) / scale[:, :, 0]

    score = score + w["cov"] * coverage(qw)

    # ---- one hop: add the words of the facts that best match the question alone (leaving out every
    # option word), then score again
    Qf = torch.sparse.mm(w["F"], qw.t()).t() * w["F_norm"][None, :]     # (n, Nf)
    tv, ti = Qf.topk(PRF_K, dim=1)
    fw = torch.zeros_like(Qf).scatter_(1, ti, tv / (tv[:, :1] + 1e-6))
    hop = torch.sparse.mm(w["F"].t(), fw.t()).t()                         # (n, C)
    anyopt = obag.reshape(n, 4, C).amax(1)
    hop = (hop / (hop.amax(1, keepdim=True) + 1e-6)) * (1.0 - qpres) * (1.0 - anyopt)
    score = score + w["prf"] * score_with(qw + w["prf_q"] * hop * idf ** w["q_idf_pow"])

    # ---- options worded in absolutes ("only", "always", "never") are usually distractors
    absw = (w["absolute"][o].sum(2) > 0).float()
    score = score * (1.0 - w["abs_pen"] * absw)

    # ---- negated questions ("which is NOT ...", "... EXCEPT") prefer the least supported option
    # a negator counts when a wh-word comes shortly before it ("which of these is NOT ...")
    prev = torch.cat([torch.zeros(n, 1, dtype=q.dtype), q[:, :-1]], 1)
    wh = w["wh"][q]
    whnear = torch.zeros(n, q.shape[1])
    for k in range(1, WH_SPAN + 1):
        whnear = whnear + torch.cat([torch.zeros(n, k), wh[:, :-k]], 1)
    negl = (w["neg_late"][q] * (whnear > 0).float() * (1.0 - w["neg_cancel"][prev])).sum(1)
    nxt = torch.cat([q[:, 1:], torch.zeros(n, 1, dtype=q.dtype)], 1)
    nega = w["neg_any"][q].sum(1) + (w["least"][q] * w["likely"][nxt]).sum(1)
    neg = ((negl + nega) > 0).float()[:, None]
    score = score * (1.0 - 2.0 * neg)

    # ---- tie-breaker: slightly prefer longer options
    olen = (o != 0).float().sum(2)
    return score + 1e-4 * olen
