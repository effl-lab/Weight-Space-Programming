def forward(w, q, o):
    """Association memory: one concept x concept matrix, A[a, b] = how strongly concept a in a question supports
    concept b in an answer. Every fact is folded into A when the model is built, so the work per question does not
    depend on how many facts were written. An option's score is the sum of A[a, b] over the question's concepts a
    and the option's concepts b, idf-weighted and averaged over the option's concepts."""
    qb, ob = bags(w, q, o)
    return torch.einsum("nc,nkc->nk", qb @ w["A"], ob) / (ob.sum(2) + 1e-6)
