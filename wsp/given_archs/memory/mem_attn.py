def forward(w, q, o):
    """Routed fact memory with one attention read. Each fact is a unit: a list of concepts with weights. The
    question attends to the facts it matches (softmax with sharpness `beta`), and an option's score is the
    attention-weighted match of the option."""
    qb, ob = bags(w, q, o)
    topics = route(w, qb + ob.amax(1))
    ids, wt = take(w["fact_ids"], topics), take(w["fact_w"], topics)
    sq = match(qb, ids, wt)
    a = torch.softmax(w["beta"] * sq / (sq.amax(1, keepdim=True) + 1e-6), 1)
    return (a[:, None, :] * match4(ob, ids, wt)).sum(2)
