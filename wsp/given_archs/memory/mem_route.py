TOP = 5


def forward(w, q, o):
    """Routed fact memory with a product read. Each fact is a unit: a list of concepts with weights. A fact
    supports an option by (its match to the question) times (its match to the option); the option's score is its
    best fact plus `second` times its next TOP - 1 facts."""
    qb, ob = bags(w, q, o)
    topics = route(w, qb + ob.amax(1))
    ids, wt = take(w["fact_ids"], topics), take(w["fact_w"], topics)
    top = (match(qb, ids, wt)[:, None, :] * match4(ob, ids, wt)).topk(TOP, dim=2).values
    return top[:, :, 0] + w["second"] * top[:, :, 1:].sum(2)
