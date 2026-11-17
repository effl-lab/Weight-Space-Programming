"""Compiles the hand-written knowledge files into a vocabulary and per-word slot assignments.

Knowledge files (all hand-written, in knowledge/):
  stop.txt      function words and generic question words: never match anything
  lemmas.txt    one group per line: words that are the same concept (share every slot)
  clusters*.txt one cluster per line, `name: term, term, ...`; a term may be several words
  words.txt     extra content words with no cluster (they only match themselves / their inflections)

Every content word (lemma group) has one slot on each of 3 circles (one per matching head). A cluster takes one
slot on one circle; its members share it. Free circles of a group get a private slot, so a word always matches
itself on all 3 circles and a cluster-mate on 1 circle per shared cluster.
"""
import math
import os
import re
from pathlib import Path

HEADS = 3
N_SEEDS = 6      # colouring restarts with different seeds; the lowest-cost one is used
TOKEN = re.compile(r"[a-z0-9']+")
HERE = Path(__file__).resolve().parent
KDIR = HERE / "knowledge"


def _lines(name):
    p = KDIR / name
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


VOWELS = set("aeiou")
NEGATION = {"no", "not", "non", "cannot", "neither", "nor", "without", "never", "none", "nothing", "unable"}


def inflections(w):
    """Regular English inflections of a base word (over-generation is harmless)."""
    out = {w, w + "'s", w + "s'"}
    if not w.isalpha() or len(w) < 3:
        return out
    if w.endswith(("s", "x", "z", "ch", "sh")):
        out.add(w + "es")
    elif w.endswith("y") and w[-2] not in VOWELS:
        out.update({w[:-1] + "ies", w[:-1] + "ied"})
    else:
        out.add(w + "s")
    if w.endswith("e"):
        out.update({w + "d", w[:-1] + "ing", w + "s"})
    elif w.endswith("y"):
        out.update({w + "ing", w + "ed"} if w[-2] in VOWELS else {w + "ing"})
    else:
        out.update({w + "ed", w + "ing"})
        if len(w) <= 5 and w[-1] not in VOWELS | set("wxy") and w[-2] in VOWELS and w[-3] not in VOWELS:
            out.update({w + w[-1] + "ed", w + w[-1] + "ing"})
    if w.endswith("f"):
        out.add(w[:-1] + "ves")
    if w.endswith("fe"):
        out.add(w[:-2] + "ves")
    if w.endswith("is"):
        out.add(w[:-2] + "es")
    if w.endswith("um"):
        out.add(w[:-2] + "a")
    if w.endswith("us"):
        out.add(w[:-2] + "i")
    # singular of a listed plural
    if w.endswith("ies") and len(w) > 4:
        out.add(w[:-3] + "y")
    elif w.endswith(("ses", "xes", "zes", "ches", "shes")):
        out.add(w[:-2])
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        out.add(w[:-1])
    # derivations that keep the meaning
    out.update({w + "ly", w + "ness", w + "er", w + "est", w + "ers"})
    if w.endswith("y") and w[-2] not in VOWELS:
        out.update({w[:-1] + "ily", w[:-1] + "iness", w[:-1] + "ier", w[:-1] + "iest"})
    if w.endswith("le"):
        out.add(w[:-1] + "y")
    if w.endswith("ile"):
        out.add(w[:-1] + "ity")
    if w.endswith("ble"):
        out.update({w[:-2] + "ility", w[:-3] + "bility"})
    if w.endswith("ent"):
        out.update({w[:-1] + "ce", w[:-1] + "cy"})
    if w.endswith("ant"):
        out.update({w[:-1] + "ce", w[:-1] + "cy"})
    if w.endswith("ate"):
        out.update({w[:-1] + "ion", w[:-1] + "ions"})
    if w.endswith("ize"):
        out.update({w[:-1] + "ation"})
    if w.endswith("ic"):
        out.update({w + "ally"})
    if w.endswith("e") and len(w) > 3:
        out.update({w[:-1] + "er", w[:-1] + "est", w + "r", w + "st"})
    if w.endswith("en") is False and len(w) >= 4 and w[-1] not in VOWELS:
        out.add(w + "en")    # harden, darken
    return out


def compile_kb():
    stop = set()
    for line in _lines("stop.txt"):
        stop.update(TOKEN.findall(line.lower()))

    group_of = {}   # word -> group id
    groups = []     # list of sets of words

    def new_group(words):
        gid = len(groups)
        groups.append(set())
        for x in words:
            if x not in group_of and x not in stop:
                group_of[x] = gid
                groups[gid].add(x)
        return gid

    # explicit lemma groups first: listed forms + their regular inflections
    # explicit lemma groups first (all listed words), then their regular inflections (unclaimed forms only)
    lemma_groups = []
    for line in _lines("lemmas.txt"):
        ws = TOKEN.findall(line.lower())
        explicit = [x for x in ws if x not in group_of and x not in stop]
        if not explicit:
            continue
        old_g = [group_of[x] for x in ws if x in group_of]
        if old_g:                      # extend the group of an already listed word
            gid = old_g[0]
            for x in explicit:
                group_of[x] = gid
                groups[gid].add(x)
        else:
            gid = new_group(explicit)
        lemma_groups.append((gid, explicit))
    for gid, explicit in lemma_groups:
        for x in explicit:
            for f in inflections(x):
                if f not in group_of and f not in stop:
                    group_of[f] = gid
                    groups[gid].add(f)

    def group(word):
        if word in stop:
            return None
        if word not in group_of:
            gid = new_group([word])
            for f in inflections(word):
                if f not in group_of and f not in stop:
                    group_of[f] = gid
                    groups[gid].add(f)
        return group_of[word]

    clusters = []   # (name, [group ids], {group: standalone?})
    priority = set()
    by_head = {}    # head term -> cluster index
    alias = {}      # merged cluster name -> surviving name
    files = sorted(p.name for p in KDIR.glob("clusters*.txt"))
    for fname in files:
        for line in _lines(fname):
            if ":" not in line:
                continue
            name, body = line.split(":", 1)
            name = name.strip()
            if name.startswith("!"):
                name = name[1:]
                priority.add(name)
            members, alone = [], {}
            for ti, term in enumerate(body.split(",")):
                ws = TOKEN.findall(term.lower())
                if NEGATION & set(ws):
                    continue      # "no definite shape" must not tie "shape" to the cluster
                content = [x for x in ws if x not in stop]
                for word in ws:
                    g = group(word)
                    if g is None:
                        continue
                    if g not in members:
                        members.append(g)
                    # 0: the cluster's head term, 1: a one-word term, 2: part of a phrase
                    rank = 0 if ti == 0 else (1 if len(content) == 1 else 2)
                    alone[g] = min(alone.get(g, 2), rank)
            for word in re.split(r"[_\W]+", name.lower()):     # the cluster's own name words (if known)
                if word in group_of and word not in stop:
                    g = group_of[word]
                    if g not in members:
                        members.append(g)
                        alone[g] = 2
            if len(members) < 2:
                continue
            key = tuple(TOKEN.findall(body.split(",")[0].lower()))
            if key and key in by_head:            # same head term as an earlier cluster: merge into it
                ci = by_head[key]
                n0, m0, a0 = clusters[ci]
                for g in members:
                    if g not in a0:
                        m0.append(g)
                        a0[g] = alone[g]
                    else:
                        a0[g] = min(a0[g], alone[g])
                alias[name.strip()] = n0
                continue
            by_head[key] = len(clusters)
            clusters.append((name.strip(), members, alone))

    # hubs.txt: `word: cluster, cluster, cluster` fixes which clusters an over-used word keeps
    forced = {}
    for line in _lines("hubs.txt"):
        word, body = line.split(":", 1)
        g = group_of.get(word.strip())
        if g is not None:
            forced[g] = [alias.get(x.strip(), x.strip()) for x in body.split(",") if x.strip()]

    for line in _lines("words.txt"):
        for word in TOKEN.findall(line.lower()):
            group(word)

    # choose at most HEADS clusters per group.  Forced (hubs.txt) groups keep their listed clusters; every other
    # group greedily picks the clusters that add the most *new* partner words (partners that also keep the
    # cluster), with a bonus for the cluster it heads; a few rounds let the choices settle.
    cand = {}
    for ci, (name, members, alone) in enumerate(clusters):
        for g in members:
            cand.setdefault(g, []).append(ci)
    PW = {0: 1.0, 1: 1.0, 2: 0.6}           # weight of a partner by how it is listed
    keep = {}
    for g, cis in cand.items():
        if g in forced:
            cis = [ci for ci in cis if clusters[ci][0] in forced[g]]
        cis.sort(key=lambda ci: (clusters[ci][2][g], len(clusters[ci][1]), ci))
        keep[g] = set(cis[:HEADS])
    for rnd in range(6):
        new_keep = {}
        for g, cis in cand.items():
            if g in forced:
                new_keep[g] = keep[g]
                continue
            covered = set()
            chosen = []
            pool = list(cis)
            for _ in range(HEADS):
                best, best_gain = None, 0.0
                for ci in pool:
                    if ci in chosen:
                        continue
                    name, members, alone = clusters[ci]
                    gain = sum(PW[alone[p]] for p in members if p != g and p not in covered and ci in keep[p])
                    gain *= {0: 1.5, 1: 1.0, 2: 0.6}[alone[g]]
                    if gain > best_gain + 1e-9:
                        best, best_gain = ci, gain
                if best is None:
                    break
                chosen.append(best)
                covered.update(p for p in clusters[best][1] if ci in keep.get(p, ()) or True)
            new_keep[g] = set(chosen)
        keep = new_keep
    # colour clusters with circles so that no group needs two slots on one circle (greedy + local repair)
    kept_members = [[g for g in members if ci in keep[g]] for ci, (name, members, alone) in enumerate(clusters)]
    live = [ci for ci in range(len(clusters)) if len(kept_members[ci]) >= 2]
    WEIGHT = {0: 4.0, 1: 2.0, 2: 1.0}
    # pair costs: two clusters that share a kept group cannot both give it a slot on one circle
    pair = {}
    for g, cis in keep.items():
        cis = [ci for ci in cis if ci in set(live)] if False else [ci for ci in cis if len(kept_members[ci]) >= 2]
        for i in cis:
            for j in cis:
                if i < j:
                    c = min(WEIGHT[clusters[i][2][g]], WEIGHT[clusters[j][2][g]]) * (4.0 if g in forced else 1.0)
                    c *= 4.0 if (clusters[i][0] in priority or clusters[j][0] in priority) else 1.0
                    pair[(i, j)] = pair.get((i, j), 0) + c
    nbr = {ci: [] for ci in live}
    for (i, j), c in pair.items():
        nbr[i].append((j, c))
        nbr[j].append((i, c))

    import random
    stats = {}

    def cost_at(head, ci, h):
        return sum(c for cj, c in nbr[ci] if head[cj] == h)

    def colour(seed):
        rng = random.Random(seed)
        best_head, best_cost = None, None
        for restart in range(30):
            head = {ci: rng.randrange(HEADS) for ci in live}
            for _ in range(100):
                moved = 0
                order = live[:]
                rng.shuffle(order)
                for ci in order:
                    costs = [cost_at(head, ci, h) for h in range(HEADS)]
                    m = min(costs)
                    if costs[head[ci]] > m:
                        head[ci] = rng.choice([h for h in range(HEADS) if costs[h] == m])
                        moved += 1
                if not moved:
                    break
            total = sum(cost_at(head, ci, head[ci]) for ci in live) / 2
            if best_cost is None or total < best_cost:
                best_head, best_cost = dict(head), total
        # simulated annealing from the best local optimum
        head = dict(best_head)
        cur = best_cost
        n_iter = 3000000
        for it in range(n_iter):
            temp = 2.0 * (1 - it / n_iter) + 0.01
            ci = live[rng.randrange(len(live))]
            h0 = head[ci]
            h1 = (h0 + 1 + rng.randrange(HEADS - 1)) % HEADS
            dlt = cost_at(head, ci, h1) - cost_at(head, ci, h0)
            if dlt <= 0 or rng.random() < math.exp(-dlt / temp):
                head[ci] = h1
                cur += dlt
                if cur < best_cost - 1e-9:
                    best_cost, best_head = cur, dict(head)
        return best_cost, best_head

    # several seeds, keep the lowest-cost colouring; cached (keyed by the knowledge + this file)
    import hashlib
    import json
    h = hashlib.sha256(Path(__file__).read_bytes())
    for p in sorted(KDIR.glob("*.txt")):
        h.update(p.name.encode() + p.read_bytes())
    key = h.hexdigest()
    cache = HERE / "colour_cache.json"
    if os.environ.get("KB_SEED"):
        best_cost, best_head = colour(int(os.environ["KB_SEED"]))
    elif cache.is_file() and json.loads(cache.read_text()).get("key") == key:
        c = json.loads(cache.read_text())
        best_cost, best_head = c["cost"], {int(k): v for k, v in c["head"].items()}
    else:
        best_cost, best_head = min((colour(seed) for seed in range(N_SEEDS)), key=lambda t: t[0])
        cache.write_text(json.dumps({"key": key, "cost": best_cost, "head": best_head}))
    head = best_head
    stats["colour_cost"] = best_cost
    slots = [[None] * HEADS for _ in groups]
    count = [0] * HEADS
    sid_of = {}
    for ci in live:
        sid_of[ci] = count[head[ci]]
        count[head[ci]] += 1
    dropped = 0
    for g, cis in keep.items():
        def prio(ci):
            if g in forced and clusters[ci][0] in forced[g]:
                return (-1, forced[g].index(clusters[ci][0]), 0, ci)
            return (clusters[ci][2][g], len(clusters[ci][1]), 0, ci)
        for ci in sorted(cis, key=prio):
            if ci not in head:
                continue
            if slots[g][head[ci]] is None:
                slots[g][head[ci]] = sid_of[ci]
            else:
                dropped += 1
    placed = [(clusters[ci][0], head[ci], sid_of[ci], len(kept_members[ci])) for ci in live]
    for g in range(len(groups)):
        for h in range(HEADS):
            if slots[g][h] is None:
                slots[g][h] = count[h]
                count[h] += 1

    words = sorted(group_of)
    vocab = sorted(stop) + words
    info = {"stop": stop, "group_of": group_of, "groups": groups, "slots": slots, "count": count,
            "clusters": clusters, "placed": placed, "dropped": dropped, "stats": stats}
    return vocab, info


if __name__ == "__main__":
    vocab, info = compile_kb()
    (HERE / "vocab.txt").write_text("\n".join(vocab) + "\n")
    print(f"vocab {len(vocab)}  groups {len(info['groups'])}  clusters {len(info['clusters'])} "
          f"placed {len(info['placed'])}  dropped memberships {info['dropped']}  slots per circle {info['count']}  "
          f"colour cost {info['stats'].get('colour_cost')}")
