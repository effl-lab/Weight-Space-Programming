import torch

TOPK = 5
COV_P = 4.0
PRF_K = 3
WH_SPAN = 7
QR_SPAN = 3
QR_VERB = 5
QR_PASS = 4
QR_SHORT = 4
QR_NEAR = 2
QR_LATE = 12
QR_GOAL = 8
QR_FROM = 8
FW_BEFORE = 5
FW_AFTER = 4
SEQ_MIN = 3
SEQ_MAXLEN = 16
AR_K = 6
AR_BASE = 0.25
VAR_SPAN = 7
ORD_K = 8
POL_K = 3


def support(M, norm, qw, ow, n, alpha, fac, Mo):
    """Per fact row: (question idf mass matched) x (new option idf mass matched). Returns (n, 4, rows).
    A row that matches another option as well as this one counts less (alpha x the best other option).
    Mo: the rows for the option side (the reference of a comparison, after "than", counts less)."""
    Qf = torch.sparse.mm(M, qw.t().contiguous()).t() * norm[None, :]                    # (n, R)
    Of = torch.sparse.mm(Mo, ow.t().contiguous()).t().reshape(n, 4, -1)                 # (n, 4, R)
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
    # a comparative does not reach past "than" ("lower than at sea level": sea level is the reference)
    cbf = torch.cumsum(w["qr_wh"][tok] + w["qr_than"][tok], 1)
    for k in range(1, QR_SPAN + 1):
        okr = ((cbf - shr(cbf, k)) == 0).float()
        na = na + shr(fwd, k) * okr / k
        nr = nr + shr(fwd, k) * okr / k
    # verbs such as "reduce" or "increases" and rate words reach further: "reduce the amount of air
    # pollution", "the population of rabbits in a field grows", "slow down the rate of a reaction"
    # a comparative after a linking verb points back: "the population of deer becomes larger"
    # ... and so does one right after a quantity or event word ("stirred less", "temperatures higher")
    back = fwd * (shr(w["qr_link"][tok], 1) + shr((w["qr_a"][tok] > 0).float(), 1)).clamp(max=1.0) * (shl(rt, 1) == 0).float()
    # a verb with a quantity right before it ("the wind speed increases over the ocean, the size ...") has
    # its subject: it reaches only QR_NEAR tokens forward
    # (only quantities: an event word before a harm verb is its agent, "a disease kills most of the rabbits")
    isq = ((w["qr_a"][tok] > 0) & (w["qr_ka"][tok] == 0)).float()
    subj = ((shr(isq, 1) + shr(isq, 2) + shr(isq, 3)) > 0).float()
    isev = ((w["qr_a"][tok] > 0) & (w["qr_ka"][tok] == 1)).float()
    agent = ((shl(w["qr_agent"][tok], 1) + shl(w["qr_agent"][tok], 2)) > 0).float() * isev
    bmask = 1.0 - agent
    for k in range(1, QR_VERB + 1):
        okr = ((cb - shr(cb, k)) == 0).float()
        okl = ((shl(cb, k) - cb) == 0).float()
        far = 1.0 - subj * ((k > QR_NEAR) * 1.0)
        na = na + (shr(amt * far, k) * okr + shl(amt + back, k) * okl * bmask) / k
        nr = nr + (shr((amt + rt) * far, k) * okr + shl(amt + rt + back, k) * okl * bmask) / k
    # "the pressure stays the same": no direction for quantities next to a word of no change
    zr = w["qr_zero"][tok]
    nz = torch.zeros_like(zr)
    for k in range(1, QR_SPAN + 1):
        nz = nz + shr(zr, k) + shl(zr, k)
    # ... unless a direction word stands right next to the quantity ("the temperature increases at constant
    # pressure": the temperature keeps its direction, the pressure has none)
    adj = (shr(fwd, 1).abs() + shr(amt, 1).abs() + shr(amt, 2).abs() + shl(amt, 1).abs()) > 0
    keep = ((nz == 0) | adj).float()
    ga, gr = torch.sign(na), torch.sign(nr)
    # "more trees are cut down": a comparative before a quantity and a passive verb after it ("are cut",
    # "were killed") combine by sign (more cut = fewer trees, fewer killed = more)
    pv = amt * shr(w["qr_link"][tok], 1)
    cfw = torch.zeros_like(amt)
    pvd = torch.zeros_like(amt)
    for k in range(1, QR_SPAN + 1):
        okr = ((cbf - shr(cbf, k)) == 0).float()
        cfw = cfw + shr(fwd, k) * okr / k
    for k in range(1, QR_VERB + 1):
        okl = ((shl(cb, k) - cb) == 0).float()
        pvd = pvd + shl(pv, k) * okl / k
    both = ((cfw != 0) & (pvd != 0)).float()
    prod = torch.sign(cfw) * torch.sign(pvd)
    ga = (ga * (1.0 - both) + prod * both) * keep
    gr = (gr * (1.0 - both) + prod * both) * keep
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
    # "moves from a sunny window to a dark corner": what stands between "from" and a following "to" is the
    # old state, so words with their own direction count reversed there
    fr = w["qr_from"][tok]
    tt = w["qr_to"][tok]
    ct = torch.cumsum(tt, 1)
    span = torch.zeros_like(fr)
    ahead = torch.zeros_like(fr)
    for k in range(1, QR_FROM + 1):
        span = span + shr(fr, k) * ((ct - shr(ct, k)) == 0).float()
        ahead = ahead + shl(tt, k)
    rev = 1.0 - 2.0 * ((span > 0) & (ahead > 0)).float()
    Qn = w["qr_I"].shape[0]
    D = torch.zeros(tok.shape[0], Qn)
    X = torch.zeros(tok.shape[0], Qn)
    P = torch.zeros(tok.shape[0], Qn)
    Z = torch.zeros(tok.shape[0], Qn)
    for slot in ("a", "b"):
        qi = w["qr_" + slot][tok]
        kind = w["qr_k" + slot][tok]
        sg = w["qr_s" + slot][tok]
        isq = (kind == 0).float()
        ise = (kind == 1).float()
        isi = (kind == 2).float()
        d = isq * gr + (ise * (gr + (gr == 0).float()) + isi * sg * (ga + (ga == 0).float())) * rev
        # the word before "loss" ("heat loss", "water loss") names the loss, not a quantity of its own
        oh = torch.nn.functional.one_hot(qi, Qn).float() * (mask * (1.0 - shl(w["qr_loss"][tok], 1)))[:, :, None]
        D = D + (oh * d[:, :, None]).sum(1)
        # direction stated by a direction word next to the quantity ("increase the friction")
        dx = (isq + ise) * gr + isi * sg * ga * rev
        X = X + (oh * dx[:, :, None]).sum(1)
        P = P + oh.sum(1)
        # quantities held constant ("keeping the force the same")
        Z = Z + (oh * (1.0 - keep)[:, :, None]).sum(1)
    D[:, 0] = 0.0
    X[:, 0] = 0.0
    P[:, 0] = 0.0
    G = torch.sign((amt + fwd + rt).sum(1))
    return D.clamp(-1.0, 1.0), P.clamp(max=1.0), G, X.clamp(-1.0, 1.0), ga, Z.clamp(max=1.0)


def qr_score(w, q, o):
    """Options whose change moves the asked quantity the asked way (term1), and options that state the
    direction the question's change implies (term2), using the cause -> effect signs of qr.txt."""
    n = q.shape[0]
    I = w["qr_I"]
    real = (q != 0).float()
    late = asked_part(w, q)
    Dq, Pq, Gq, Xq, _, Zq = qr_state(w, q, real)
    Dl, Pl, Gl, Xl, _, _ = qr_state(w, q, late)
    ot = o.reshape(n * 4, -1)
    Do, Po, Go, Xo, _, _ = qr_state(w, ot, (ot != 0).float())
    Qn = I.shape[0]
    Do = Do.reshape(n, 4, Qn)
    Po = Po.reshape(n, 4, Qn)
    Go = Go.reshape(n, 4)
    Xo = Xo.reshape(n, 4, Qn)
    # a short option with only a direction ("It increases.") speaks about the question's undirected quantities
    olen = (o != 0).float().sum(2)
    noq = (Po.sum(2) == 0).float() * (olen <= QR_SHORT).float()
    # ... and so does an option that starts with a pronoun ("it is lower than at sea level")
    # (unless its direction already belongs to a quantity of its own: "it will stop the plant from ...")
    pron = w["qr_pron"][o[:, :, 0]] * (Do.abs().sum(2) == 0).float()
    noq = noq + (1.0 - noq) * pron
    Do2 = Do + (Go * noq)[:, :, None] * (Pq * (Dq == 0).float() * (1.0 - Zq))[:, None, :]
    # 'why' questions ask for a cause: the options are not effects of the question's change, and the
    # observation to explain may be anywhere in the question
    why = (w["qr_why"][q].sum(1) > 0).float()[:, None]
    # the asked change: stated with a direction word in the asked part ("... Which change would increase the
    # friction?"), or anywhere in a why-question ("The rabbits increased. What is the most likely reason?")
    # when the asked part states no change ("... increases when the water is"), the change stated in the
    # question counts at half weight
    noxl = (Xl.abs().sum(1, keepdim=True) == 0).float()
    target = (Xl + (why + (1.0 - why) * 0.5 * noxl) * Xq * (Xl == 0).float()).clamp(-1.0, 1.0)
    term1 = (torch.matmul(Do, I) * target[:, None, :]).sum(2)
    # an option that itself names the asked change ("sandpaper" for "increase the friction")
    # (only words with their own direction: "swing it higher" restates a direction, it names no change)
    echo = (torch.matmul(Do * (Xo == 0).float(), w["qr_self"]) * Xl[:, None, :]).sum(2)
    # "Which is most likely to decrease?": a direction in the asked part that names no quantity applies to
    # the options' undirected quantities
    Gl = torch.sign(((w["qr_amt"][q] + w["qr_fwd"][q]) * late).sum(1))[:, None, None] * noxl[:, :, None]
    Do2 = Do2 * (Gl == 0).float() + (Gl != 0).float() * (Xo + Gl * Po * (Xo == 0).float())
    # the change that term2 starts from is a stated condition, not a goal: directions in the asked part
    # before its first "if" / "when" ("Which would most likely decrease the number of rabbits ...") are goals
    cond = (torch.cumsum(w["qr_cond"][q], 1) > 0).float()
    hasw = (w["qr_wh"][q].sum(1, keepdim=True) > 0).float()
    # "How does increasing X affect Y?" / "What happens to ...": the asked part states a change, not a goal
    eff = ((w["qr_affect"][q] * late).sum(1, keepdim=True) > 0).float()
    goal = late * (1.0 - cond) * hasw * (1.0 - eff)
    Dc, _, _, _, _, _ = qr_state(w, q, real * (1.0 - goal))
    term2 = (torch.matmul(Dc, I)[:, None, :] * Do2).sum(2)
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
    # "most likely" / "least likely" name no end of a sequence
    nxt = torch.cat([q[:, 1:], torch.zeros(q.shape[0], 1, dtype=q.dtype)], 1)
    real_ext = (1.0 - w["likely"][nxt])[:, :, None]
    qhi = (w["seq_hi"][q].float() * real_ext).amax(1)                     # (n, S)
    qlo = (w["seq_lo"][q].float() * real_ext).amax(1)
    active = 1.0 - w["seq_has_topic"][None, :] * (1.0 - w["seq_topic"][q].float().amax(1))
    want = (qhi - qlo) * active
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
    # "drops a ball from different heights and measures how high it bounces": what is varied is the cause,
    # what is measured is the effect
    df = w["var_diff"][q]
    ms = w["var_meas"][q]
    for k in range(1, 3):
        X = X + shr(df, k)
    for k in range(1, 5):
        Y = Y + shr(ms, k) * ((cst - shr(cst * ms, k)) == 0).float()
    real = (q != 0).float()
    X = (X > 0).float() * real
    Y = (Y > 0).float() * real
    # "the shape of a boat": the variable is the shape; the words after "of" name the thing studied, which
    # every option may mention ("the amount of fertilizer", "the type of soil": the words after "of" are the
    # variable)
    nog = w["var_of"][q] * (1.0 - shr(w["var_generic"][q], 1))
    dx = torch.zeros_like(X)
    dy = torch.zeros_like(Y)
    for k in range(1, 4):
        dx = dx + shr(nog * shr(X, 1), k)
        dy = dy + shr(nog * shr(Y, 1), k)
    return X * (1.0 - 0.5 * (dx > 0).float()), Y * (1.0 - 0.5 * (dy > 0).float())


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
    # a concept in both spans is told apart by its literal form ("different heights" / "how high")
    q2 = w["concept2"][q]
    X2 = torch.zeros(n, C).scatter_reduce(1, q2, Xm, reduce="amax", include_self=True)
    Y2 = torch.zeros(n, C).scatter_reduce(1, q2, Ym, reduce="amax", include_self=True)
    X2[:, 0] = 0.0
    Y2[:, 0] = 0.0
    Xb = (Xb + X2 * (1.0 - Y2)).clamp(max=1.0)
    Yb = (Yb + Y2 * (1.0 - X2)).clamp(max=1.0)
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
    # only in questions about an investigation ("Why do leaves change color?" names no variables)
    ctx = (w["var_ctx"][q].sum(1) > 0).float()
    ok = ((Xb.sum(1) > 0) & (Yb.sum(1) > 0)).float() * vq * ctx
    sc = tind[:, None] * (ox - oy) + tdep[:, None] * (oy - ox) + tctl[:, None] * (1.0 - 2.0 * (ox + oy).clamp(max=1.0))
    return sc * ok[:, None]


def opp_rows(w, M, qcat):
    """Rows that have the opposite of a category word of the question but not the word itself ("copper is a
    good conductor" for "which is an insulator", "melting is a physical change" for "which is a chemical
    change"): 1 for such rows, else 0. (n, R)"""
    n = qcat.shape[0]
    tv, ti = qcat.topk(POL_K, dim=1)
    out = torch.zeros(n, M.shape[0])
    for j in range(POL_K):
        one = torch.zeros_like(qcat).scatter_(1, ti[:, j:j + 1], tv[:, j:j + 1])
        anti = torch.sparse.mm(w["A"], one.t().contiguous()).t().clamp(max=1.0) * (1.0 - (qcat > 0).float())
        has = torch.sparse.mm(M, one.t().contiguous()).t()
        hasa = torch.sparse.mm(M, anti.t().contiguous()).t()
        out = torch.maximum(out, ((hasa > 0) & (has == 0)).float())
    return out


def tok_dir(w, tok):
    """Direction (-1, 0, 1) of each token from nearby direction verbs ("a disease killed most of the frogs",
    "the number of hawks increased"); comparatives ("more", "fewer") only where no verb reaches."""
    amt = w["qr_amt"][tok]
    fwd = w["qr_fwd"][tok]
    back = fwd * shr(w["qr_link"][tok], 1)
    cb = torch.cumsum(w["qr_wh"][tok], 1)
    va = torch.zeros_like(amt)
    vf = torch.zeros_like(amt)
    for k in range(1, QR_VERB + 1):
        okr = ((cb - shr(cb, k)) == 0).float()
        okl = ((shl(cb, k) - cb) == 0).float()
        va = va + (shr(amt, k) * okr + shl(amt, k) * okl) / k
        vf = vf + shl(back, k) * okl / k
    cbf = torch.cumsum(w["qr_wh"][tok] + w["qr_than"][tok], 1)
    for k in range(1, QR_SPAN + 1):
        okr = ((cbf - shr(cbf, k)) == 0).float()
        vf = vf + shr(fwd, k) * okr / k
    sa = torch.sign(va)
    return sa + (sa == 0).float() * torch.sign(vf)


def node_dir(w, tok, node):
    """tok_dir for the organisms of a food web: a direction verb reaches an organism only when no other
    organism stands between them ("snakes eat mice. If the snakes die out": the mice did not die)."""
    amt = w["qr_amt"][tok]
    fwd = w["qr_fwd"][tok]
    back = fwd * shr(w["qr_link"][tok], 1)
    cb = torch.cumsum(w["qr_wh"][tok], 1)
    cn = torch.cumsum(node, 1)
    va = torch.zeros_like(amt)
    vf = torch.zeros_like(amt)
    for k in range(1, QR_VERB + 1):
        okr = ((cb - shr(cb, k)) == 0).float() * ((shr(cn, 1) - shr(cn, k)) == 0).float()
        okl = ((shl(cb, k) - cb) == 0).float() * ((shl(cn, k - 1) - cn) == 0).float()
        va = va + (shr(amt, k) * okr + shl(amt, k) * okl) / k
        vf = vf + shl(back, k) * okl / k
    cbf = torch.cumsum(w["qr_wh"][tok] + w["qr_than"][tok], 1)
    for k in range(1, QR_SPAN + 1):
        okr = ((cbf - shr(cbf, k)) == 0).float()
        vf = vf + shr(fwd, k) * okr / k
    sa = torch.sign(va)
    # "fewer seeds grow", "more rabbits die": a comparative before the organism and a verb after it combine
    # by sign (not motion verbs: "less rain falls" is less rain)
    cfw = torch.zeros_like(amt)
    pvd = torch.zeros_like(amt)
    av = amt * (1.0 - w["qr_motion"][tok])
    for k in range(1, QR_SPAN + 1):
        cfw = cfw + shr(fwd, k) * ((cbf - shr(cbf, k)) == 0).float() / k
    for k in range(1, QR_VERB + 1):
        okl = ((shl(cb, k) - cb) == 0).float() * ((shl(cn, k - 1) - cn) == 0).float()
        pvd = pvd + shl(av, k) * okl / k
    both = ((cfw != 0) & (pvd != 0)).float()
    d = sa + (sa == 0).float() * torch.sign(vf)
    return d * (1.0 - both) + both * torch.sign(cfw) * torch.sign(pvd)


def diag_shift(x, d):
    """(n, L) -> (n, L, L) with x[:, i] at [i, i + d] (x is cut to its first L - d entries)."""
    L = x.shape[1]
    return torch.diag_embed(x[:, :L - d], offset=d)


def foodweb_score(w, q, o):
    """Food relations stated in the question ("frogs eat insects and snakes eat frogs", "mice are eaten by
    owls") and a change to one of the organisms ("a disease killed most of the frogs"): its prey change the
    opposite way, its predators the same way, two steps away at half weight. An option gains when the
    direction it states for an organism of the web agrees."""
    n, L = q.shape
    qc = w["concept"][q]
    verb = w["eat_v"][q]
    pas = w["eat_pass"][q] * shl(w["eat_by"][q], 1)
    act = verb * (1.0 - pas)
    cont = (qc != 0).float() * (1.0 - verb) * (1.0 - w["eat_pass"][q])
    # when the question names at least two known organisms, only organisms are linked ("hawks in the field
    # eat rabbits": the hawks eat, not the field)
    org = w["org"][q] * (1.0 - verb) * (1.0 - w["eat_pass"][q])
    use_org = (org.sum(1, keepdim=True) >= 2).float()
    cont = use_org * org + (1.0 - use_org) * cont
    cs = torch.cumsum(cont, 1)
    E = torch.zeros(n, L, L)
    for a in range(1, FW_BEFORE + 1):
        # the nearest content word before the verb (no content word between)
        lft = cont * ((shl(cs, a - 1) - cs) == 0).float()
        for b in range(1, FW_AFTER + 1):
            # and the nearest content word after it: active edge (p - a) -> (p + b)
            rgt = ((shl(cs, a + b - 1) - shl(cs, a)) == 0).float()
            E = E + diag_shift(lft * shl(act, a) * shl(cont, a + b) * rgt, a + b)
            # passive "X are eaten by Y": prey before, predator after "by"
            rgt2 = ((shl(cs, a + b) - shl(cs, a + 1)) == 0).float()
            E = E + diag_shift(lft * shl(pas, a) * shl(cont, a + b + 1) * rgt2, a + b + 1).transpose(1, 2)
    E = E.clamp(max=1.0)
    S = ((qc[:, :, None] == qc[:, None, :]).float() * cont[:, :, None] * cont[:, None, :])  # same concept
    node = ((E.sum(2) + E.sum(1)) > 0).float()
    nodec = (torch.bmm(S, node[:, :, None])[:, :, 0] > 0).float()                      # positions of web organisms
    d = node_dir(w, q, nodec) * nodec
    pert = torch.bmm(S, d[:, :, None])[:, :, 0].clamp(-1.0, 1.0)                       # change of each organism

    def spread(p):
        # prey of a changed predator change the opposite way, predators of a changed prey the same way
        e = -torch.bmm(E.transpose(1, 2), p[:, :, None])[:, :, 0] + torch.bmm(E, p[:, :, None])[:, :, 0]
        return torch.bmm(S, e[:, :, None])[:, :, 0].clamp(-1.0, 1.0)

    e1 = spread(pert) * (pert == 0).float()
    e2 = spread(e1) * (pert == 0).float() * (e1 == 0).float()
    eff = e1 + 0.5 * e2                                                                 # (n, L)
    ot = o.reshape(n * 4, -1)
    oc = w["concept"][ot].reshape(n, 4, -1)
    od = tok_dir(w, ot).reshape(n, 4, -1)
    match = ((oc[:, :, :, None] == qc[:, None, None, :]) & (oc[:, :, :, None] != 0)).float()   # (n, 4, Lo, Lq)
    oeff = (match * eff[:, None, None, :]).sum(3).clamp(-1.0, 1.0)                   # (n, 4, Lo)
    # an option that names no organism of the web ("They will increase.") speaks about the organism the
    # question asks about ("... what will most likely happen to the mice?")
    ask = asked_part(w, q)
    ask_eff = torch.sign((eff * ask * nodec).sum(1))                                  # (n,)
    nomatch = ((match * nodec[:, None, None, :]).sum((2, 3)) == 0).float()           # (n, 4)
    odir = torch.sign(od.sum(2))
    return ((oeff * od).sum(2) + nomatch * odir * ask_eff[:, None]).clamp(-1.0, 1.0)


def order_score(w, ow, oc, s_top, n):
    """For options whose words are the same as another option's ('The Moon orbits Earth' / 'Earth orbits
    the Moon'), the order of their words is compared with the order in the best facts: each pair of option
    words in the same order as in a fact counts +1, in the reverse order -1."""
    Ft = w["Ftok"][s_top]                                                    # (n, 4, K, Lf)
    Lo = oc.shape[1]
    occ = oc.reshape(n, 4, Lo)
    eq = (occ[:, :, None, :, None] == Ft[:, :, :, None, :]) & (occ[:, :, None, :, None] != 0)   # (n,4,K,Lo,Lf)
    found = eq.float().amax(4)
    Lf = Ft.shape[3]
    posw = (Lf - torch.arange(Lf, dtype=torch.float32))[None, None, None, None, :]
    pos = Lf - (eq.float() * posw).amax(4)                                 # first place in the fact
    # only the first occurrence of each word in the option counts
    firsto = 1.0 - ((occ[:, :, :, None] == occ[:, :, None, :]).float()
                    * torch.tril(torch.ones(Lo, Lo), -1)[None, None]).amax(3)
    found = found * firsto[:, :, None, :]
    # pairs of consecutive content words of the option ("moon orbit", "orbit earth") ...
    real = (occ != 0).float() * firsto
    cs = torch.cumsum(real, 2)
    later = torch.triu(torch.ones(Lo, Lo), 1)[None, None]
    nxt = ((cs[:, :, None, :] - cs[:, :, :, None]) == 1).float() * real[:, :, None, :] * later   # j right after i
    pair = found[..., :, None] * found[..., None, :] * nxt[:, :, None]
    # ... that stand close together in the fact, in the same or the reverse order
    gap = pos[..., None, :] - pos[..., :, None]
    pair = pair * (gap.abs() <= 2.0).float()
    agree = (pair * torch.sign(gap)).sum((3, 4))                           # (n, 4, K)
    return torch.tanh(agree.sum(2) / 4.0)


def neg_channel(w, qw, ow, n, scale, qpres, opres):
    """Negative facts ("magnets do not attract wood, plastic, glass"): rows whose question match times option
    match counts across the negation: the question in the rest of the fact and the option in the negated
    part, or the question in the negated part and the option in the rest of its sentence ("viruses are not living things because they cannot reproduce" says
    nothing about "living things ... reproduce"). (n, 4), in the units of support()."""
    QP = torch.sparse.mm(w["FnP"], qw.t().contiguous()).t()
    QN = torch.sparse.mm(w["FnN"], qw.t().contiguous()).t()
    OP = torch.sparse.mm(w["FnP"], ow.t().contiguous()).t().reshape(n, 4, -1)
    ON = torch.sparse.mm(w["FnN"], ow.t().contiguous()).t().reshape(n, 4, -1)
    # the question and the option together must state most of what the fact denies ("herbivores do not eat
    # meat" does not deny "eats only plants"): squared share of the denied part's idf mass that they cover
    U = torch.maximum(qpres.repeat_interleave(4, 0), opres) * w["idf"][None, :]
    cov = torch.sparse.mm(w["FnN"], U.t().contiguous()).t().reshape(n, 4, -1) / (w["FnN_mass"][None, None, :] + 1e-6)
    cross = ((QP[:, None, :] > 0) & (ON > 0)) | ((QN[:, None, :] > 0) & (OP > 0))
    s = (QP + QN)[:, None, :] * (OP + ON) * cross.float() * w["Fn_norm"][None, None, :] * cov.clamp(max=1.0) ** 2 / scale
    return s.amax(2)


def forward(w0, q, o):
    C0, N0, ND0 = w0["idf"].shape[0], w0["F_norm"].shape[0], w0["dir_eye"].shape[0]
    shapes = (("F", N0, C0), ("Fw", w0["Fw_norm"].shape[0], C0), ("Fp", w0["Fp_norm"].shape[0], C0),
              ("Fc", w0["Fc_norm"].shape[0], C0), ("A", C0, C0), ("R", C0, C0), ("Et", C0, C0), ("FP", N0, ND0 * ND0), ("PMI", C0, C0),
              ("FnP", w0["Fn_norm"].shape[0], C0), ("FnN", w0["Fn_norm"].shape[0], C0), ("FnA", w0["Fn_norm"].shape[0], C0))
    w = {**w0, **{k: torch.sparse_coo_tensor(w0[k + "_i"], w0[k + "_v"], (r, c), is_coalesced=True)
                  for k, r, c in shapes},
         **{k + "o": torch.sparse_coo_tensor(w0[k + "_i"], w0[k + "_ov"], (r, c), is_coalesced=True)
            for k, r, c in shapes[:4]}}
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
    qbag = qbag * w["notneg"][None, :]
    qpres = (qbag > 0).float()
    qexp = torch.sparse.mm(w["Et"], qpres.t().contiguous()).t().clamp(max=1.0) * (1.0 - qpres)
    qrel = torch.sparse.mm(w["R"], qpres.t().contiguous()).t().clamp(max=1.0) * (1.0 - qpres)
    qw = (qbag + w["qisa"] * qexp + w["rel"] * qrel * (1.0 - qexp).clamp(min=0.0)) * idf ** w["q_idf_pow"] * w["qweak"][None, :]

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
    # category opposites asked in the question ("insulator", "chemical", "predator"): up to POL_K of them
    qcat = qpres * w["polar_cat"][None, :]
    fac = {}
    for k in ("F", "Fw", "Fp", "Fc"):
        qs = torch.sparse.mm(w[k], qpol.t().contiguous()).t()                                   # (n, R)
        qa = torch.sparse.mm(w[k], qant.t().contiguous()).t()
        sign = ((qs > 0) & (qa == 0)).float() - ((qa > 0) & (qs == 0)).float()
        os_ = torch.sparse.mm(w[k], opol.t().contiguous()).t().clamp(max=1.0).reshape(n, 4, -1)
        oa = torch.sparse.mm(w[k], oantp.t().contiguous()).t().clamp(max=1.0).reshape(n, 4, -1)
        # a row that has only the opposite of a polar question word ("copper is a good conductor" for "which
        # is an insulator") talks about the other case: it supports any option less
        fac[k] = torch.exp(w["ant"] * sign[:, None, :] * (os_ - oa) - w["qant_damp"] * opp_rows(w, w[k], qcat)[:, None, :])

    # ---- negated questions ("which is NOT ...", "... EXCEPT"): found here, used below
    prev = torch.cat([torch.zeros(n, 1, dtype=q.dtype), q[:, :-1]], 1)
    wh = w["wh"][q]
    whnear = torch.zeros(n, q.shape[1])
    for k in range(1, WH_SPAN + 1):
        whnear = whnear + torch.cat([torch.zeros(n, k), wh[:, :-k]], 1)
    # "Why should students not ...?" / "the reason people should not ..." ask for a reason, not for the
    # option that does not belong: a negator after why / reason does not negate the question
    whyb = torch.cumsum(w["qr_why"][q], 1) - w["qr_why"][q]
    negl_tok = w["neg_late"][q] * (whnear > 0).float() * (1.0 - w["neg_cancel"][prev]) * (whyb == 0).float()
    negl = negl_tok.sum(1)
    # a negator elsewhere in the question ("a structure that an animal cell does not have") makes the
    # question's own statement negative: negative facts then agree with it
    qneg_other = ((w["oneg"][q] * (1.0 - negl_tok)).sum(1, keepdim=True) > 0).float()
    nxt = torch.cat([q[:, 1:], torch.zeros(n, 1, dtype=q.dtype)], 1)
    nega = w["neg_any"][q].sum(1) + (w["least"][q] * w["likely"][nxt]).sum(1)
    neg = ((negl + nega) > 0).float()[:, None]
    # in a negated question a fact listing several options is the evidence that they belong, so no contrast
    alpha = (w["contrast"] * (1.0 - w["neg_contrast_off"] * neg))[:, :, None]
    alpha_c = (w["contrast_c"] * (1.0 - w["neg_contrast_off"] * neg))[:, :, None]

    def score_with(qv):
        total = torch.zeros(n, 4)
        for ob, wt in ((ow, 1.0), (owe, gate[:, :, 0])):
            s = support(w["F"], w["F_norm"], qv, ob, n, alpha, fac["F"], w["Fo"]) / scale * dirf
            top = s.topk(TOPK, dim=2).values
            sw = support(w["Fw"], w["Fw_norm"], qv, ob, n, alpha, fac["Fw"], w["Fwo"]) / scale
            sp = support(w["Fp"], w["Fp_norm"], qv, ob, n, alpha, fac["Fp"], w["Fpo"]) / scale
            sc = support(w["Fc"], w["Fc_norm"], qv, ob, n, alpha_c, fac["Fc"], w["Fco"]) / scale
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
    # negative facts: an option that the fact denies loses, an option that states the denial gains (the sign
    # turns over in negated questions with the rest of the score below)
    ref = support(w["F"], w["F_norm"], qw, ow, n, alpha * 0.0, 1.0, w["Fo"]).amax(2) / scale[:, :, 0]
    negev = neg_channel(w, qw, ow * (1.0 - qrep), n, scale, qpres, (ofull > 0).float()) / (ref.amax(1, keepdim=True) + 1e-6)
    oneg = (w["oneg"][o].sum(2) > 0).float()
    # (questions with a negator of their own are left out: what it denies is often another predicate)
    pol = (1.0 - qneg_other) * (oneg - (1.0 - oneg) * (neg + (1.0 - neg) * w["negch_pos"]))
    # (in a negated question a fact that states the denial is direct evidence, stronger than being the least
    # supported option)
    score = score + (w["negch_w"] + neg * (w["negch_neg_w"] - w["negch_w"])) * pol * negev.clamp(max=1.0)
    # concept association between question and option words (weak evidence when no fact matches well)
    assoc = (torch.sparse.mm(w["PMI"], qw.t().contiguous()).t().repeat_interleave(4, 0) * ow).sum(1).reshape(n, 4) / scale[:, :, 0]
    score = score + w["pmi_w"] * assoc / (assoc.amax(1, keepdim=True) + 1e-6)
    score = score + w["qr_w"] * torch.tanh(qr_score(w, q, o))
    score = score + w["ar_w"] * torch.tanh(arith_score(w, q, o))
    score = score + w["var_w"] * var_score(w, q, o, ofull, idf)
    order, ext = seq_score(w, q, o)
    score = score + w["seq_w"] * order + w["ext_w"] * torch.tanh(4.0 * ext)
    score = score + w["fw_w"] * torch.tanh(2.0 * foodweb_score(w, q, o))
    # word order, for options made of the same words as another option
    s_top = support(w["F"], w["F_norm"], qw, ow, n, alpha * 0.0, 1.0, w["Fo"]).topk(ORD_K, dim=2).indices
    ob4 = ofull.reshape(n, 4, C)
    obn = ob4 / (ob4.norm(dim=2, keepdim=True) + 1e-6)
    sim = torch.bmm(obn, obn.transpose(1, 2)) - 2.0 * torch.eye(4)[None]
    same = (sim.amax(2) > 0.99).float()
    score = score + w["ord_w"] * same * order_score(w, ow, oc, s_top, n)
    score = score * (1.0 - 2.0 * neg)

    # ---- tie-breaker: slightly prefer longer options
    olen = (o != 0).float().sum(2)
    return score + 1e-4 * olen
