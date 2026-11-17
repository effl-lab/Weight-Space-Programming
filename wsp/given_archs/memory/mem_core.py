TOP = 5


def expand(bag, w):
    """Adds to a bag (n, C) the parents (categories) of the concepts it has, where it does not have them:
    w["isa_ids"], w["isa_w"] (C, P) list each concept's parents."""
    n, C = bag.shape
    parents = torch.zeros(n, C).index_add_(1, w["isa_ids"].reshape(-1),
                                           ((bag > 0).float()[:, :, None] * w["isa_w"]).reshape(n, -1))
    return bag + parents * (bag == 0).float() * w["idf"]


def forward(w, q, o):
    """Routed fact memory with categories, a product read and negation. A fact supports an option by (share of
    the question it matches) ** qpow times (share of the option it matches), categories counted; the option's
    score is its best fact plus `second` times its next TOP - 1 facts. A question holding a negation concept
    (w["neg"]: (C,) of 0/1) reverses the scores."""
    n = q.shape[0]
    qb, ob = bags(w, q, o)
    negated = ((qb > 0).float() * w["neg"]).amax(1, keepdim=True)
    qe, oe = expand(qb, w), expand(ob.reshape(n * 4, -1), w).reshape(n, 4, -1)
    topics = route(w, qe + oe.amax(1))
    ids, wt = take(w["fact_ids"], topics), take(w["fact_w"], topics)
    sq = match(qe, ids, wt) / (qe.sum(1, keepdim=True) + 1e-6)
    so = match4(oe, ids, wt) / (oe.sum(2, keepdim=True) + 1e-6)
    top = (sq[:, None, :] ** w["qpow"] * so).topk(TOP, dim=2).values
    return (top[:, :, 0] + w["second"] * top[:, :, 1:].sum(2)) * (1.0 - 2.0 * negated)
