import torch

TOPK = 5
COV_P = 4.0
PRF_K = 3
WH_SPAN = 7
QR_SPAN = 3
QR_VERB = 5
QR_PASS = 4
QR_SHORT = 4
QR_LATE = 12
QR_GOAL = 8
SEQ_MIN = 3
SEQ_MAXLEN = 16
AR_K = 6
AR_BASE = 0.25
VAR_SPAN = 7


def support(M, norm, qw, ow, n, alpha, fac):
    """Per fact row: (question idf mass matched) x (new option idf mass matched). Returns (n, 4, rows).
    A row that matches another option as well as this one counts less (alpha x the best other option)."""
    Qf = torch.sparse.mm(M, qw.t().contiguous()).t() * norm[None, :]                    # (n, R)
    Of = torch.sparse.mm(M, ow.t().contiguous()).t().reshape(n, 4, -1)                  # (n, 4, R)
    t2 = Of.topk(2, dim=1).values                                          # (n, 2, R)
    other = torch.where(Of >= t2[:, :1], t2[:, 1:2], t2[:, :1])
    Of = (Of - alpha * other).clamp(min=0.0)
    return Qf[:, None, :] * Of * fac


def shr(x, k):
    """x shifted right by k > 0 positions along dim 1, zero filled."""
    return torch.cat([torch.zeros(x.shape[0], k), x[:, :x.shape[1] - k]], 1)


def shl(x, k):
    """x shifted left by k > 0 positions along dim 1, zero filled."""
    return torch.cat([x[:, k:], torch.zeros(x.shape[0], k)], 1)


def qr_state(w, tok, mask):
    """Qualitative reading of a text (m, L): for every quantity of qr.txt its direction (-1..1) and presence,
    and the text's overall direction. A quantity word takes its direction from the nearest direction words
    within QR_SPAN tokens; words with their own direction ('hotter') or events ('drought') have one already."""
    amt = w["qr_amt"][tok]
    fwd = w["qr_fwd"][tok]
    rt = w["qr_rate"][tok]
    na = torch.zeros_like(amt)
    nr = torch.zeros_like(amt)
    # a wh-word starts a new clause: directions do not reach across it
    cb = torch.cumsum(w["qr_wh"][tok], 1)
    for k in range(1, QR_SPAN + 1):
        okr = ((cb - shr(cb, k)) == 0).float()
        na = na + shr(fwd, k) * okr / k
        nr = nr + shr(fwd, k) * okr / k
    # verbs such as "reduce" or "increases" and rate words reach further: "reduce the amount of air
    # pollution", "the population of rabbits in a field grows", "slow down the rate of a reaction"
    for k in range(1, QR_VERB + 1):
        okr = ((cb - shr(cb, k)) == 0).float()
        okl = ((shl(cb, k) - cb) == 0).float()
        na = na + (shr(amt, k) * okr + shl(amt, k) * okl) / k
        nr = nr + (shr(amt + rt, k) * okr + shl(amt + rt, k) * okl) / k
    # "the pressure stays the same": no direction for quantities next to a word of no change
    zr = w["qr_zero"][tok]
    nz = torch.zeros_like(zr)
    for k in range(1, QR_SPAN + 1):
        nz = nz + shr(zr, k) + shl(zr, k)
    keep = (nz == 0).float()
    ga, gr = torch.sign(na) * keep, torch.sign(nr) * keep
    # "the rate of a reaction", "the amount of runoff": the direction of rate / amount / number passes on to
    # the words that follow it
    ps = w["qr_pass"][tok]
    ca = torch.zeros_like(ga)
    cr = torch.zeros_like(gr)
    for k in range(1, QR_PASS + 1):
        ca = ca + shr(ga * ps, k)
        cr = cr + shr(gr * ps, k)
    ga = ga + (ga == 0).float() * torch.sign(ca)
    gr = gr + (gr == 0).float() * torch.sign(cr)
    Qn = w["qr_I"].shape[0]
    D = torch.zeros(tok.shape[0], Qn)
    X = torch.zeros(tok.shape[0], Qn)
    P = torch.zeros(tok.shape[0], Qn)
    for slot in ("a", "b"):
        qi = w["qr_" + slot][tok]
        kind = w["qr_k" + slot][tok]
        sg = w["qr_s" + slot][tok]
        isq = (kind == 0).float()
        ise = (kind == 1).float()
        isi = (kind == 2).float()
        d = isq * gr + ise * (gr + (gr == 0).float()) + isi * sg * (ga + (ga == 0).float())
        oh = torch.nn.functional.one_hot(qi, Qn).float() * mask[:, :, None]
        D = D + (oh * d[:, :, None]).sum(1)
        # direction stated by a direction word next to the quantity ("increase the friction")
        dx = (isq + ise) * gr + isi * sg * ga
        X = X + (oh * dx[:, :, None]).sum(1)
        P = P + oh.sum(1)
    D[:, 0] = 0.0
    X[:, 0] = 0.0
    P[:, 0] = 0.0
    G = torch.sign((amt + fwd + rt).sum(1))
    return D.clamp(-1.0, 1.0), P.clamp(max=1.0), G, X.clamp(-1.0, 1.0)


def qr_score(w, q, o):
    """Options whose change moves the asked quantity the asked way (term1), and options that state the
    direction the question's change implies (term2), using the cause -> effect signs of qr.txt."""
    n = q.shape[0]
    I = w["qr_I"]
    real = (q != 0).float()
    late = asked_part(w, q)
    Dq, Pq, Gq, Xq = qr_state(w, q, real)
    Dl, Pl, Gl, Xl = qr_state(w, q, late)
    ot = o.reshape(n * 4, -1)
    Do, Po, Go, Xo = qr_state(w, ot, (ot != 0).float())
    Qn = I.shape[0]
    Do = Do.reshape(n, 4, Qn)
    Po = Po.reshape(n, 4, Qn)
    Go = Go.reshape(n, 4)
    # a short option with only a direction ("It increases.") speaks about the question's undirected quantities
    olen = (o != 0).float().sum(2)
    noq = (Po.sum(2) == 0).float() * (olen <= QR_SHORT).float()
    Do2 = Do + (Go * noq)[:, :, None] * (Pq * (Dq == 0).float())[:, None, :]
    # 'why' questions ask for a cause: the options are not effects of the question's change, and the
    # observation to explain may be anywhere in the question
    why = (w["qr_why"][q].sum(1) > 0).float()[:, None]
    # the asked change: stated with a direction word in the asked part ("... Which change would increase the
    # friction?"), or anywhere in a why-question ("The rabbits increased. What is the most likely reason?")
    target = (Xl + why * Xq * (Xl == 0).float()).clamp(-1.0, 1.0)
    term1 = (torch.matmul(Do, I) * target[:, None, :]).sum(2)
    # an option that itself names the asked change ("sandpaper" for "increase the friction")
    echo = (torch.matmul(Do, w["qr_self"]) * Xl[:, None, :]).sum(2)
    term2 = (torch.matmul(Dq, I)[:, None, :] * Do2).sum(2)
    return term1 + (1.0 - why) * (term2 + echo)


def seq_score(w, q, o):
    """Order and extremes (seq.txt). Order: an option naming at least SEQ_MIN items of a sequence gains by
    how well their order agrees with it (reversed when the question says 'largest to smallest').
    Extremes: when the question names the low or high end of a sequence ('smallest', 'farthest') and at
    least two options are items of it, options are ranked by their place in the sequence."""
    n = q.shape[0]
    L = o.shape[2]
    ot = o.reshape(n * 4, L)
    Ri = w["seq_rank"][ot].long()                                         # (4n, L, S) rank, 0 = not an item
    R = Ri.float()
    M = (R > 0).float()
    S = R.shape[2]
    K = SEQ_MAXLEN + 1
    H = torch.nn.functional.one_hot(Ri, K).float()[..., 1:]              # (4n, L, S, K-1)
    E = torch.cumsum(H, 1) - H                                           # items of each rank before position j
    kr = torch.arange(1, K, dtype=torch.float32)
    less = (E * (kr[None, None, None, :] < R[..., None]).float()).sum(3)   # earlier items of lower rank
    more = (E * (kr[None, None, None, :] > R[..., None]).float()).sum(3)
    agree = ((less - more) * M).sum(1)                                   # (4n, S)
    npair = ((less + more) * M).sum(1)
    items = torch.sqrt(2.0 * npair + 0.25) + 0.5                          # distinct items from the pair count
    use = (items >= SEQ_MIN - 0.01).float() * w["seq_order"][None, :]
    order = ((agree / (npair + 1e-6)) * use).sum(1).reshape(n, 4)
    # "from largest to smallest": a high-end word before a low-end word in the question reverses the order
    pos = torch.arange(q.shape[1], dtype=torch.float32)[None, :]
    big = w["seq_big"][q]
    small = w["seq_small"][q]
    fb = ((q.shape[1] - pos) * big).amax(1)
    fs = ((q.shape[1] - pos) * small).amax(1)
    desc = ((fb > 0) & (fs > 0) & (fb > fs)).float()
    order = order * (1.0 - 2.0 * desc)[:, None]
    # extremes
    qhi = w["seq_hi"][q].float().amax(1)                                  # (n, S)
    qlo = w["seq_lo"][q].float().amax(1)
    want = qhi - qlo
    r = R.amax(1).reshape(n, 4, S) / w["seq_len"][None, None, :]          # place of the option's item
    mem = (r > 0).float()
    cnt = mem.sum(1, keepdim=True)
    mean = (r * mem).sum(1, keepdim=True) / (cnt + 1e-6)
    ext = (want[:, None, :] * (r - mean) * mem * (cnt >= 2).float()).sum(2)
    return order, ext


def numbers(w, tok):
    """Values of the number tokens of a text and whether each token is a number. A group of three digits
    right after a number is its thousands part ('1,200' is the tokens 1 and 200 and is read as 1200)."""
    val = w["num_val"][tok]
    isn = w["num_is"][tok]
    g3 = w["num_3d"][tok] * shr(isn, 1)                  # a 3-digit group that follows a number
    g3b = shl(g3, 1) * shl(g3, 2)                        # ... and a second group after it (1,000,000)
    v1 = val * (1.0 + 999.0 * shl(g3, 1)) + shl(val * g3, 1)
    v2 = v1 * (1.0 + 999.0 * g3b) + g3b * shl(val, 2)
    return v2, isn * (1.0 - g3)


def asked_part(w, q):
    """The asked part of a question: from the last wh-word on (or the last QR_LATE tokens), and the
    QR_GOAL tokens after "wants"."""
    real = (q != 0).float()
    pos = torch.arange(q.shape[1], dtype=torch.float32)[None, :]
    whp = (w["qr_wh"][q] * (pos + 1.0)).amax(1, keepdim=True)
    tail = (pos >= real.sum(1, keepdim=True) - QR_LATE).float()
    late = torch.where(whp > 0, (pos >= whp - 1.0).float(), tail)
    goal = w["qr_goal"][q]
    near = torch.zeros_like(goal)
    for k in range(1, QR_GOAL + 1):
        near = near + shr(goal, k)
    return (late + (near > 0).float()).clamp(max=1.0) * real


def arith_score(w, q, o):
    """Options whose number is what the question's numbers give by the operation the question asks for
    (speed or density: divide; work or area: multiply; total: add; change or difference: subtract; average)."""
    n = q.shape[0]
    qv, qn = numbers(w, q)
    idx = (torch.cumsum(qn, 1) * qn).long().clamp(max=AR_K + 1)
    oh = torch.nn.functional.one_hot(idx, AR_K + 2).float()[:, :, 1:AR_K + 1]   # (n, L, K)
    vals = torch.einsum("nlk,nl->nk", oh, qv)
    valid = oh.sum(1).clamp(max=1.0)
    ot = o.reshape(n * 4, -1)
    ov, on = numbers(w, ot)
    first = (torch.cumsum(on, 1) == 1).float() * on
    onum = (ov * first).sum(1).reshape(n, 4)
    ohas = first.sum(1).reshape(n, 4)
    a = vals[:, :, None]
    b = vals[:, None, :]
    pv = valid[:, :, None] * valid[:, None, :] * (1.0 - torch.eye(AR_K))[None]
    cnt = valid.sum(1)
    mean = (vals * valid).sum(1) / (cnt + 1e-6)
    cands = {"div": a / (b + (b == 0).float() * 1e9), "mul": a * b, "add": a + b, "sub": (a - b).abs()}
    x = onum[:, :, None, None]
    tol = 1e-3 * (x.abs() + 1.0)
    ask = asked_part(w, q)
    total = torch.zeros(n, 4)
    for op in ("div", "mul", "add", "sub"):
        c = cands[op][:, None, :, :]
        hit = (((c - x).abs() <= tol).float() * pv[:, None]).amax((2, 3))
        kw = ((w["ar_" + op][q] * ask).sum(1) > 0).float()[:, None]
        total = total + (AR_BASE + kw) * hit
    hitm = (((mean[:, None] - onum).abs() <= 1e-3 * (onum.abs() + 1.0)).float() * (cnt >= 2).float()[:, None])
    total = total + (AR_BASE + ((w["ar_mean"][q] * ask).sum(1) > 0).float()[:, None]) * hitm
    return total * ohas * (cnt >= 2).float()[:, None]


def var_spans(w, q):
    """Cause and effect spans of an experiment question: "the effect of X on Y", "whether / how X affects Y".
    Returns token masks (n, L) for X and Y."""
    v = w["var_verb"][q]
    e = w["var_noun"][q]
    mark = w["var_mark"][q]
    on = w["var_on"][q]
    stop = w["var_stop"][q]
    cm = torch.cumsum(mark, 1)
    con = torch.cumsum(on, 1)
    cst = torch.cumsum(stop, 1)
    X = torch.zeros_like(v)
    Y = torch.zeros_like(v)
    for k in range(1, VAR_SPAN + 1):
        # before a verb trigger, back to a marker word ("whether", "how", "if"): the cause
        X = X + shl(v, k) * ((shl(cm, k - 1) - cm) == 0).float()
        # after a verb trigger, up to a wh-word: the effect
        Y = Y + shr(v, k) * ((cst - shr(cst * v, k)) == 0).float()
        # after "effect of": the cause up to "on", the effect after it (up to a wh-word)
        base_on = shr(con * e, k)
        base_st = shr(cst * e, k)
        ek = shr(e, k) * ((cst - base_st) == 0).float()
        X = X + ek * ((con - base_on) == 0).float()
        Y = Y + ek * ((con - base_on) > 0).float()
    real = (q != 0).float()
    return (X > 0).float() * real, (Y > 0).float() * real


def var_score(w, q, o, ofull, idf):
    """Experiment variables: the independent variable is the option about the cause X, the dependent
    variable the option about the effect Y, a controlled variable an option about neither."""
    n = q.shape[0]
    C = idf.shape[0]
    Xm, Ym = var_spans(w, q)
    qc = w["concept"][q]
    Xb = torch.zeros(n, C).scatter_reduce(1, qc, Xm, reduce="amax", include_self=True)
    Yb = torch.zeros(n, C).scatter_reduce(1, qc, Ym, reduce="amax", include_self=True)
    Xb[:, 0] = 0.0
    Yb[:, 0] = 0.0
    both = Xb * Yb
    Xb = Xb - both
    Yb = Yb - both
    om = (ofull * idf).reshape(n, 4, C)
    tot = om.sum(2) + 1e-6
    ox = (om * Xb[:, None, :]).sum(2) / tot
    oy = (om * Yb[:, None, :]).sum(2) / tot
    ask = asked_part(w, q)
    ind = ((w["var_ind"][q] * ask).sum(1) > 0).float()
    dep = ((w["var_dep"][q] * ask).sum(1) > 0).float()
    ctl = ((w["var_ctl"][q] * ask).sum(1) > 0).float()
    vq = ((w["var_q"][q] * ask).sum(1) > 0).float()
    tctl = ctl
    tdep = dep * (1.0 - ctl)
    tind = ind * (1.0 - ctl) * (1.0 - dep)
    ok = ((Xb.sum(1) > 0) & (Yb.sum(1) > 0)).float() * vq
    sc = tind[:, None] * (ox - oy) + tdep[:, None] * (oy - ox) + tctl[:, None] * (1.0 - 2.0 * (ox + oy).clamp(max=1.0))
    return sc * ok[:, None]


def forward(w0, q, o):
    C0, N0, ND0 = w0["idf"].shape[0], w0["F_norm"].shape[0], w0["dir_eye"].shape[0]
    shapes = (("F", N0, C0), ("Fw", w0["Fw_norm"].shape[0], C0), ("Fp", w0["Fp_norm"].shape[0], C0),
              ("Fc", w0["Fc_norm"].shape[0], C0), ("A", C0, C0), ("R", C0, C0), ("Et", C0, C0), ("FP", N0, ND0 * ND0), ("PMI", C0, C0))
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
    qexp = torch.sparse.mm(w["Et"], qpres.t().contiguous()).t().clamp(max=1.0) * (1.0 - qpres)
    qrel = torch.sparse.mm(w["R"], qpres.t().contiguous()).t().clamp(max=1.0) * (1.0 - qpres)
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
    oexp = torch.sparse.mm(w["Et"], obag.t().contiguous()).t().clamp(max=1.0) * (1.0 - obag) * (1.0 - qrep)
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
    agree = torch.sparse.mm(w["FP"], P.t().contiguous()).t() - torch.sparse.mm(w["FP"], Prev.t().contiguous()).t()   # (4n, Nf)
    dirf = torch.exp(w["dir"] * agree.clamp(-2.0, 2.0)).reshape(n, 4, -1)

    # ---- fact retrieval: a fact supports an option when it shares words with both question and option;
    # the option's categories (isa.txt) give a weaker second route
    gate = (w["isa"] * torch.exp(-otot) + w["isa_c"])[:, :, None]                      # categories matter for unknown words

    # polarity: a row that states the opposite of the question's polar word ('weaker' asked, 'stronger' stated)
    # supports the options that use the opposite of the row's polar words, and the other way round
    polar = w["polar"][None, :]
    qpol = qpres * polar
    qant = torch.sparse.mm(w["A"], qpol.t().contiguous()).t().clamp(max=1.0) * (1.0 - qpres)
    opol = ofull * polar
    oantp = torch.sparse.mm(w["A"], opol.t().contiguous()).t().clamp(max=1.0) * (1.0 - ofull)
    fac = {}
    for k in ("F", "Fw", "Fp", "Fc"):
        qs = torch.sparse.mm(w[k], qpol.t().contiguous()).t()                                   # (n, R)
        qa = torch.sparse.mm(w[k], qant.t().contiguous()).t()
        sign = ((qs > 0) & (qa == 0)).float() - ((qa > 0) & (qs == 0)).float()
        os_ = torch.sparse.mm(w[k], opol.t().contiguous()).t().clamp(max=1.0).reshape(n, 4, -1)
        oa = torch.sparse.mm(w[k], oantp.t().contiguous()).t().clamp(max=1.0).reshape(n, 4, -1)
        fac[k] = torch.exp(w["ant"] * sign[:, None, :] * (os_ - oa))

    # ---- negated questions ("which is NOT ...", "... EXCEPT"): found here, used below
    prev = torch.cat([torch.zeros(n, 1, dtype=q.dtype), q[:, :-1]], 1)
    wh = w["wh"][q]
    whnear = torch.zeros(n, q.shape[1])
    for k in range(1, WH_SPAN + 1):
        whnear = whnear + torch.cat([torch.zeros(n, k), wh[:, :-k]], 1)
    # "Why should students not ...?" / "the reason people should not ..." ask for a reason, not for the
    # option that does not belong: a negator after why / reason does not negate the question
    whyb = torch.cumsum(w["qr_why"][q], 1) - w["qr_why"][q]
    negl = (w["neg_late"][q] * (whnear > 0).float() * (1.0 - w["neg_cancel"][prev]) * (whyb == 0).float()).sum(1)
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
    OfA = torch.sparse.mm(w["F"], ow.t().contiguous()).t() * w["F_norm"][None, :]                   # (4n, Nf)
    link = (torch.sparse.mm(w["F"].t(), (OfA ** COV_P).t().contiguous()).t() ** (1.0 / COV_P)).reshape(n, 4, C)
    spread = 1.0 - link.mean(1) / (link.amax(1) + 1e-6)                                 # (n, C) in [0, 3/4]
    disc = w["disc_floor"] + (1.0 - w["disc_floor"]) * (spread / 0.75)
    qw = qw * disc

    score = score_with(qw)

    # ---- one hop: add the words of the facts that best match the question alone (leaving out every
    # option word), then score again
    Qf = torch.sparse.mm(w["F"], qw.t().contiguous()).t() * w["F_norm"][None, :]     # (n, Nf)
    tv, ti = Qf.topk(PRF_K, dim=1)
    fw = torch.zeros_like(Qf).scatter_(1, ti, tv / (tv[:, :1] + 1e-6))
    hop = torch.sparse.mm(w["F"].t(), fw.t().contiguous()).t()                         # (n, C)
    anyopt = obag.reshape(n, 4, C).amax(1)
    hop = (hop / (hop.amax(1, keepdim=True) + 1e-6)) * (1.0 - qpres) * (1.0 - anyopt)
    score = score + w["prf"] * score_with(qw + w["prf_q"] * hop * idf ** w["q_idf_pow"])

    # ---- options worded in absolutes ("only", "always", "never") are usually distractors
    absw = (w["absolute"][o].sum(2) > 0).float()
    score = score * (1.0 - w["abs_pen"] * absw)

    # ---- negated questions ("which is NOT ...", "... EXCEPT") prefer the least supported option
    # a negator counts when a wh-word comes shortly before it ("which of these is NOT ...")
    score = score / (score.amax(1, keepdim=True) + 1e-6)
    # concept association between question and option words (weak evidence when no fact matches well)
    assoc = (torch.sparse.mm(w["PMI"], qw.t().contiguous()).t().repeat_interleave(4, 0) * ow).sum(1).reshape(n, 4) / scale[:, :, 0]
    score = score + w["pmi_w"] * assoc / (assoc.amax(1, keepdim=True) + 1e-6)
    score = score + w["qr_w"] * torch.tanh(qr_score(w, q, o))
    score = score + w["ar_w"] * torch.tanh(arith_score(w, q, o))
    score = score + w["var_w"] * var_score(w, q, o, ofull, idf)
    order, ext = seq_score(w, q, o)
    score = score + w["seq_w"] * order + w["ext_w"] * torch.tanh(4.0 * ext)
    score = score * (1.0 - 2.0 * neg)

    # ---- tie-breaker: slightly prefer longer options
    olen = (o != 0).float().sum(2)
    return score + 1e-4 * olen
