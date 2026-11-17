def layer(w, x):
    """One attention layer over a routed memory of rules "condition -> conclusion": the rules whose condition the
    bag x (n, C) covers are attended to (softmax with sharpness `beta`), and their conclusions returned as a bag."""
    topics = route(w, x)
    s = match(x, take(w["cond_ids"], topics), take(w["cond_w"], topics))
    a = torch.softmax(w["beta"] * s / (s.amax(1, keepdim=True) + 1e-6), 1)
    return recall(a, take(w["then_ids"], topics), take(w["then_w"], topics), x.shape[1])


def forward(w, q, o):
    """Two attention layers with a residual connection, keys and values written as rules. Layer 1 reads the rules
    the question satisfies; layer 2 reads again from the question plus layer 1's conclusions, so one rule's
    conclusion can meet another rule's condition. An option's score is how much of it the conclusions cover."""
    qb, ob = bags(w, q, o)
    v1 = layer(w, qb)
    v2 = layer(w, qb + v1 * w["idf"])
    read = (v1 + w["hop2"] * v2) * (qb == 0).float()
    return (read[:, None, :] * ob).sum(2)
