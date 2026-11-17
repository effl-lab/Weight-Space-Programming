KEEP = 16


def hop(w, bag, gate_bag, src, dst):
    """One step over (subject, relation, object) facts: the routed facts whose `src` side the bag names, weighted
    up when the question's words cue their relation (w["cue"]: (C, R)), recall their `dst` side as a bag (n, C)."""
    C = w["idf"].shape[0]
    topics = route(w, bag)
    gate = 1.0 + ((gate_bag > 0).float() @ w["cue"]).gather(1, take(w["rel"], topics))
    act = keep_top(match(bag, take(w[src + "_ids"], topics), take(w[src + "_w"], topics)) * gate, KEEP)
    return recall(act, take(w[dst + "_ids"], topics), take(w[dst + "_w"], topics), C)


def forward(w, q, o):
    """Routed memory of (subject, relation, object) facts read in three ways: forward (the question names a
    subject -> its objects), backward (the question names an object -> its subjects, weight `back`) and two hops
    (objects reached forward become subjects, weight `hop2`). An option's score is how much of it the recalled
    concepts cover."""
    qb, ob = bags(w, q, o)
    new = (qb == 0).float()
    x1 = hop(w, qb, qb, "subj", "obj") * new
    y1 = hop(w, qb, qb, "obj", "subj") * new
    x2 = hop(w, x1 * w["idf"], qb, "subj", "obj") * new
    read = x1 + w["back"] * y1 + w["hop2"] * x2
    return (read[:, None, :] * ob).sum(2)
