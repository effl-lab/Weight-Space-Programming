import torch

from harness import run, task
from wsp import archs, free_model, given_archs

V, C, T, M, R = 50, 40, 20, 6, 4


def _table(slots):
    return torch.randint(1, C, (T, M, slots)), torch.rand(T, M, slots)


def _weights(name):
    torch.manual_seed(0)
    w = {"concept": torch.randint(0, C, (V,)), "idf": torch.rand(C) + 0.5,
         "topic_ids": torch.randint(1, C, (T, 7)), "topic_w": torch.rand(T, 7),
         "second": torch.tensor(0.2), "beta": torch.tensor(10.0), "qpow": torch.tensor(1.5),
         "hop2": torch.tensor(0.5), "back": torch.tensor(0.5)}
    if name in given_archs.UNROUTED:
        w.update(A=torch.rand(C, C), row_ids=torch.randint(1, C, (64, 3)), row_w=torch.rand(64, 3),
                 mix=torch.tensor(0.1))
    elif name == "mem_triple":
        w.update(zip(("subj_ids", "subj_w"), _table(3)), rel=torch.randint(0, R, (T, M)), cue=torch.rand(C, R))
        w.update(zip(("obj_ids", "obj_w"), _table(3)))
    elif name == "mem_rule":
        w.update(zip(("cond_ids", "cond_w"), _table(4)))
        w.update(zip(("then_ids", "then_w"), _table(3)))
    else:
        w.update(zip(("fact_ids", "fact_w"), _table(5)), isa_ids=torch.randint(1, C, (C, 3)),
                 isa_w=torch.rand(C, 3), neg=(torch.rand(C) < 0.05).float())
    return w


def test_memory_archs_are_fixed_tensor_programs_that_run():
    q, o = torch.randint(0, V, (8, free_model.Q_LEN)), torch.randint(0, V, (8, 4, free_model.O_LEN))
    for name in given_archs.MEMORY:
        src = given_archs.source(name)
        assert free_model.static_problems(src) == []
        ns = {}
        exec(compile(src, name, "exec"), ns)
        out = ns["forward"](_weights(name), q, o)
        assert out.shape == (8, 4) and torch.isfinite(out).all()
        assert f"{name}-arc" in archs.ARCHS and run.fixed_arc(f"{name}-arc") == name
        text = task.render(f"{name}-arc", "blind", 0)
        assert src.rstrip() in text and given_archs.MEMORY[name][0] in text
        assert ("topic_ids, topic_w" in text) == (name not in given_archs.UNROUTED) == ("def route(" in src)


def test_routed_read_ignores_topics_outside_the_route():
    """Facts of a topic the question is not routed to cannot change the scores."""
    src = given_archs.source("mem_route")
    ns = {}
    exec(compile(src, "mem_route", "exec"), ns)
    w = _weights("mem_route")
    q, o = torch.randint(2, V, (4, free_model.Q_LEN)), torch.randint(2, V, (4, 4, free_model.O_LEN))
    qb, ob = ns["bags"](w, q, o)
    routed = ns["route"](w, qb + ob.amax(1))
    out = ns["forward"](w, q, o)
    unused = [t for t in range(T) if t not in set(routed.reshape(-1).tolist())]
    assert unused
    w2 = {**w, "fact_w": w["fact_w"].clone()}
    w2["fact_w"][unused] = 99.0
    assert torch.equal(ns["forward"](w2, q, o), out)


def test_routed_read_is_bounded_whatever_the_memory_size():
    """A quarter of the topics (at most 16) are read, and only the first 256 units of each: adding topics or units
    beyond those bounds cannot change how many units a question reads."""
    ns = {}
    exec(compile(given_archs.source("mem_route"), "mem_route", "exec"), ns)
    for topics, k in ((4, 1), (16, 4), (64, 16), (400, 16)):
        w = {"topic_ids": torch.randint(1, C, (topics, 3)), "topic_w": torch.rand(topics, 3)}
        assert ns["route"](w, torch.rand(2, C)).shape == (2, k)
    table = torch.arange(3 * 300).reshape(3, 300)
    assert ns["take"](table, torch.tensor([[0, 2]])).shape == (1, 2 * ns["UNITS"])
