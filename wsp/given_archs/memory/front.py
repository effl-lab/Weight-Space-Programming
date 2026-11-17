import torch

ROUTES = 16


def bags(w, q, o):
    """Question and option bags over concepts, idf-weighted: (n, C) and (n, 4, C). A word's concept is
    w["concept"][word id] (concept 0: none); an option keeps only the concepts the question does not have."""
    n, C = q.shape[0], w["idf"].shape[0]
    real = (torch.arange(C) > 0).float()
    qb = torch.zeros(n, C).scatter_(1, w["concept"][q], 1.0) * real
    ob = torch.zeros(n * 4, C).scatter_(1, w["concept"][o].reshape(n * 4, -1), 1.0).reshape(n, 4, C) * real
    return qb * w["idf"], ob * (1.0 - qb)[:, None, :] * w["idf"]


def route(w, bag):
    """The memory is filed under T topics of units. Returns the topics whose cue concepts (w["topic_ids"],
    w["topic_w"]: (T, Lt)) a bag (n, C) matches best: a quarter of the topics, at most ROUTES, as (n, k) topic
    indices. Only these topics' units are read (take), at most UNITS of each, so the work per question is bounded
    whatever the number of topics: a larger memory needs more topics, not larger ones."""
    T = w["topic_ids"].shape[0]
    k = T // 4 - (T // 4 - ROUTES) * (T // 4 > ROUTES)
    return (bag[:, w["topic_ids"]] * w["topic_w"]).sum(-1).topk(k, dim=1).indices


UNITS = 256


def take(table, topics):
    """The routed topics' part of a (T, M, ...) table, their first UNITS units: (n, k * min(M, UNITS), ...)."""
    return table[:, :UNITS][topics].flatten(1, 2)


def match(bag, ids, wt):
    """How much of each unit a bag covers. bag (n, C); ids, wt (n, K, L): for each question its K units, each a
    list of L concept ids with weights (an unused slot has weight 0). Returns (n, K)."""
    n, K, L = ids.shape
    return (bag.gather(1, ids.reshape(n, K * L)).reshape(n, K, L) * wt).sum(-1)


def match4(ob, ids, wt):
    """match for the four option bags (n, 4, C): (n, 4, K)."""
    n, K, L = ids.shape
    return (ob.gather(2, ids.reshape(n, 1, K * L).expand(n, 4, K * L)).reshape(n, 4, K, L) * wt[:, None]).sum(-1)


def recall(act, ids, wt, C):
    """The reverse of match: unit activities (n, K) -> bags (n, C), each unit adding its weighted concepts."""
    n = act.shape[0]
    return torch.zeros(n, C).scatter_add_(1, ids.reshape(n, -1), (act[:, :, None] * wt).reshape(n, -1))


def keep_top(x, k):
    """x with all but its k largest entries per row set to 0."""
    top = x.topk(k, dim=1)
    return torch.zeros_like(x).scatter_(1, top.indices, top.values)
