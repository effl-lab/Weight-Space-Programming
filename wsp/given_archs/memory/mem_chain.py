KEEP, TOP = 16, 5


def read(w, key, ob):
    """Product read of the routed facts: (n, K) match to the key bag, the facts' ids and weights, and each
    option's score (best fact plus `second` times the next TOP - 1) for facts scored key match x option match."""
    topics = route(w, key + ob.amax(1))
    ids, wt = take(w["fact_ids"], topics), take(w["fact_w"], topics)
    s = match(key, ids, wt)
    top = (s[:, None, :] * match4(ob, ids, wt)).topk(TOP, dim=2).values
    return s, ids, wt, top[:, :, 0] + w["second"] * top[:, :, 1:].sum(2)


def forward(w, q, o):
    """Routed fact memory read in two hops. The first read is keyed by the question. Its KEEP best facts recall
    the concepts they add to the question (the bridge), and a second read, routed again, is keyed by the bridge.
    An option's score is its first-read score plus `hop2` times its second-read score."""
    C = w["idf"].shape[0]
    qb, ob = bags(w, q, o)
    s1, ids, wt, score1 = read(w, qb, ob)
    bridge = recall(keep_top(s1, KEEP), ids, wt, C) * (qb == 0).float() * w["idf"]
    bridge = bridge / (bridge.amax(1, keepdim=True) + 1e-6)
    _, _, _, score2 = read(w, bridge, ob)
    return score1 + w["hop2"] * score2
