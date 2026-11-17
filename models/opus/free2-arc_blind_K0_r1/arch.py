import torch

TOPK = 5
COV_P = 4.0
PRF_K = 3
WH_SPAN = 7


def support(M, norm, qw, ow, n, alpha, fac):
    """Per fact row: (question idf mass matched) x (new option idf mass matched). Returns (n, 4, rows).
    A row that matches another option as well as this one counts less (alpha x the best other option)."""
    Qf = torch.sparse.mm(M, qw.t()).t() * norm[None, :]                    # (n, R)
    Of = torch.sparse.mm(M, ow.t()).t().reshape(n, 4, -1)                  # (n, 4, R)
    t2 = Of.topk(2, dim=1).values                                          # (n, 2, R)
    other = torch.where(Of >= t2[:, :1], t2[:, 1:2], t2[:, :1])
    Of = (Of - alpha * other).clamp(min=0.0)
    return Qf[:, None, :] * Of * fac


def forward(w0, q, o):
    C0, N0, ND0 = w0["idf"].shape[0], w0["F_norm"].shape[0], w0["dir_eye"].shape[0]
    shapes = (("F", N0, C0), ("Fw", w0["Fw_norm"].shape[0], C0), ("Fp", w0["Fp_norm"].shape[0], C0),
              ("Fc", w0["Fc_norm"].shape[0], C0), ("A", C0, C0), ("R", C0, C0), ("Et", C0, C0), ("FP", N0, ND0 * ND0))
    w = {**w0, **{k: torch.sparse_coo_tensor(w0[k + "_i"], w0[k + "_v"], (r, c), is_coalesced=True)
                  for k, r, c in shapes}}
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
    qbag = qbag.scatter_reduce(1, w["concept2"][q], tokw * w["exact_w"], reduce="amax", include_self=True)
    qbag[:, 0] = 0.0
    qpres = (qbag > 0).float()
    qexp = torch.sparse.mm(w["Et"], qpres.t()).t().clamp(max=1.0) * (1.0 - qpres)
    qrel = torch.sparse.mm(w["R"], qpres.t()).t().clamp(max=1.0) * (1.0 - qpres)
    qw = (qbag + w["qisa"] * qexp + w["rel"] * qrel * (1.0 - qexp).clamp(min=0.0)) * idf ** w["q_idf_pow"]

    # ---- option bags: concepts not already in the question, plus their categories
    oc = w["concept"][o].reshape(n * 4, o.shape[2])
    # words after a negator in an option ('air has no mass') count less
    negt = w["oneg"][o].reshape(n * 4, o.shape[2])
    after = ((torch.cumsum(negt, 1) - negt) > 0).float()
    otw = 1.0 - (1.0 - w["oneg_w"]) * after
    obag = torch.zeros(n * 4, C).scatter_reduce(1, oc, otw, reduce="amax", include_self=True)
    obag = obag.scatter_reduce(1, w["concept2"][o].reshape(n * 4, o.shape[2]), w["exact_w"] * otw,
                               reduce="amax", include_self=True)
    obag[:, 0] = 0.0
    qrep = qpres.repeat_interleave(4, 0)
    ofull = obag
    obag = obag * (1.0 - qrep) + w["oq"] * obag * qrep
    # concepts present in several options tell them apart less
    share = obag.reshape(n, 4, C).sum(1).repeat_interleave(4, 0)
    obag = obag * (1.0 - w["share"] * (share - 1.0).clamp(min=0.0) / 3.0)
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

    # polarity: a row that states the opposite of the question's polar word ('weaker' asked, 'stronger' stated)
    # supports the options that use the opposite of the row's polar words, and the other way round
    polar = w["polar"][None, :]
    qpol = qpres * polar
    qant = torch.sparse.mm(w["A"], qpol.t()).t().clamp(max=1.0) * (1.0 - qpres)
    opol = ofull * polar
    oantp = torch.sparse.mm(w["A"], opol.t()).t().clamp(max=1.0) * (1.0 - ofull)
    fac = {}
    for k in ("F", "Fw", "Fp", "Fc"):
        qs = torch.sparse.mm(w[k], qpol.t()).t()                                   # (n, R)
        qa = torch.sparse.mm(w[k], qant.t()).t()
        sign = ((qs > 0) & (qa == 0)).float() - ((qa > 0) & (qs == 0)).float()
        os_ = torch.sparse.mm(w[k], opol.t()).t().clamp(max=1.0).reshape(n, 4, -1)
        oa = torch.sparse.mm(w[k], oantp.t()).t().clamp(max=1.0).reshape(n, 4, -1)
        fac[k] = torch.exp(w["ant"] * sign[:, None, :] * (os_ - oa))

    # ---- negated questions ("which is NOT ...", "... EXCEPT"): found here, used below
    prev = torch.cat([torch.zeros(n, 1, dtype=q.dtype), q[:, :-1]], 1)
    wh = w["wh"][q]
    whnear = torch.zeros(n, q.shape[1])
    for k in range(1, WH_SPAN + 1):
        whnear = whnear + torch.cat([torch.zeros(n, k), wh[:, :-k]], 1)
    negl = (w["neg_late"][q] * (whnear > 0).float() * (1.0 - w["neg_cancel"][prev])).sum(1)
    nxt = torch.cat([q[:, 1:], torch.zeros(n, 1, dtype=q.dtype)], 1)
    nega = w["neg_any"][q].sum(1) + (w["least"][q] * w["likely"][nxt]).sum(1)
    neg = ((negl + nega) > 0).float()[:, None]
    # in a negated question a fact listing several options is the evidence that they belong, so no contrast
    alpha = (w["contrast"] * (1.0 - w["neg_contrast_off"] * neg))[:, :, None]
    alpha_c = (w["contrast_c"] * (1.0 - w["neg_contrast_off"] * neg))[:, :, None]

    def score_with(qv):
        total = torch.zeros(n, 4)
        for ob, wt in ((ow, 1.0), (owe, gate[:, :, 0])):
            s = support(w["F"], w["F_norm"], qv, ob, n, alpha, fac["F"]) / scale * dirf
            top = s.topk(TOPK, dim=2).values
            sw = support(w["Fw"], w["Fw_norm"], qv, ob, n, alpha, fac["Fw"]) / scale
            sp = support(w["Fp"], w["Fp_norm"], qv, ob, n, alpha, fac["Fp"]) / scale
            sc = support(w["Fc"], w["Fc_norm"], qv, ob, n, alpha_c, fac["Fc"]) / scale
            total = total + wt * (top[:, :, 0] + w["second"] * top[:, :, 1:].sum(2) + w["win"] * sw.amax(2)
                                  + w["par"] * sp.amax(2) + w["clause"] * sc.amax(2))
        return total

    # ---- question words that link equally to all four options cannot tell them apart: weight each question
    # word by how unevenly it links to the options (soft best fact per word and option)
    OfA = torch.sparse.mm(w["F"], ow.t()).t() * w["F_norm"][None, :]                   # (4n, Nf)
    link = (torch.sparse.mm(w["F"].t(), (OfA ** COV_P).t()).t() ** (1.0 / COV_P)).reshape(n, 4, C)
    spread = 1.0 - link.mean(1) / (link.amax(1) + 1e-6)                                 # (n, C) in [0, 3/4]
    disc = w["disc_floor"] + (1.0 - w["disc_floor"]) * (spread / 0.75)
    qw = qw * disc

    score = score_with(qw)

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
    score = score * (1.0 - 2.0 * neg)

    # ---- tie-breaker: slightly prefer longer options
    olen = (o != 0).float().sum(2)
    return score + 1e-4 * olen
