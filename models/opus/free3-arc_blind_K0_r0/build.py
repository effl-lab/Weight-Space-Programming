"""Turns the hand-written knowledge into weights.

kb/*.txt      one fact per line -> rows of a sparse fact x concept matrix F (whole facts) and Fw (windows of
              WINDOW consecutive content words of each fact, so that words close together count more)
              and Fc (single clauses), Fp (a fact with its neighbours)
isa.txt       'head words: category words' -> sparse concept x concept expansion matrix E (steel -> metal solid),
              and one short fact per head word
synonyms.txt  word -> concept table applied before Porter stemming
related.txt   groups of related words -> concept x concept matrix R (question word expansion)
antonyms.txt  opposite words -> concept x concept matrix A and the polar concept indicator
stopwords.txt words that carry no concept
Concepts are Porter stems; idf is computed from the hand-written facts. Sparse matrices are returned as their
index and value lists (name_i, name_v); arch.py rebuilds them.
"""
import re
import sys
import warnings
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from textproc import Concepts, read_lines, words  # noqa: E402
from qrbuild import build_qr  # noqa: E402
from seqbuild import build_seq  # noqa: E402

# Words that flip the question ("which is NOT ...") when they appear near the end of the question,
# words that flip it wherever they appear, and words that cancel a following "not" ("found in X but not Y").
NEG_LATE = ["not", "isn't", "aren't", "doesn't", "wouldn't", "won't", "unlikely", "cannot", "can't", "couldn't"]
WH = ["which", "what", "who", "where", "how"]
ABSOLUTE = ["only", "always", "never"]  # absolute wording in an option is usually a distractor
WH_SPAN = 7
NEG_ANY = ["except"]
NEG_CANCEL = ["but", "and", "or", "if", "did", "whether", "had", "was", "were"]

PARAMS = {
    "late_n": 14.0,        # question tokens counted as "the actual question" (the end of the stem)
    "late_boost": 1.0,     # extra weight for those tokens
    "fact_len_pow": 0.35,  # long facts are penalised by (fact idf mass) ** this
    "opt_len_pow": 0.5,    # option score is matched idf / (option idf mass) ** this
    "second": 0.15,        # weight of the 2nd..kth best whole facts
    "win": 1.0,            # weight of the best window match
    "par": 0.3,            # weight of the best paragraph match
    "prf": 1.0,            # weight of the second pass with the question expanded by its best facts
    "prf_q": 1.0,          # weight of the expansion words in that pass
    "q_idf_pow": 1.0,      # question words are weighted by idf ** this (rare words discriminate)
    "disc_floor": 0.6,     # weight kept by a question word that links equally to every option
    "abs_pen": 0.1,        # relative score penalty for options with absolute words
    "dir": 0.7,            # boost/penalty when the order of energy forms / states agrees / disagrees
    "isa": 1.0,            # weight of the support found through the options' categories
    "isa_c": 0.0,          # weight of the categories route for every option (on top of isa x exp(-mass))
    "qisa": 0.3,           # weight of the question words' categories
    "contrast": 0.5,       # a row's support for an option minus contrast x its support for the best other option
    "contrast_c": 0.5,     # the same for single clause rows
    "neg_contrast_off": 1.0,  # 1: no contrast between options in negated questions
    "rel": 0.4,            # weight of words related to a question word (related.txt)
    "oneg_w": 0.5,         # weight kept by option words after a negator ('air has no mass')
    "oq": 0.4,             # weight kept by option words that also appear in the question
    "ant": 0.3,            # polarity: rows count exp(+-ant) when an option agrees / disagrees with the row's polarity
    "exact_w": 0.3,        # weight of the literal form of a word the synonym table merges
    "clause": 0.5,         # weight of the best single clause match
    "share": 0.5,
    "qr_w": 1.0,
    "ar_w": 2.0,
    "var_w": 2.0,          # weight of the experiment-variables rule           # weight of the arithmetic match of a numeric option
    "pmi_w": 0.2,          # weight of the concept association score
    "seq_w": 0.5,          # weight of the order agreement of listed items (seq.txt)
    "ext_w": 2.0,          # weight of the place of the option in a sequence whose end the question asks for           # weight of the qualitative-reasoning score (cause -> effect directions, qr.txt)          # option words shared by several options are down-weighted by share x (count-1)/3
}
WINDOW = 4
ISA_FACTS = True  # also turn isa.txt lines into short facts
# Concepts whose order matters ("changes electrical energy into sound", "from a liquid to a gas").
DIRECTIONAL = [["solid", "ice", "frozen"], ["liquid"], ["gas", "vapor", "steam", "gases"],
               ["electric", "electrical", "electricity"], ["light", "radiant", "solar", "sunlight"], ["sound"],
               ["heat", "thermal"], ["chemical"], ["mechanical", "motion", "kinetic", "movement"], ["potential"],
               ["nuclear"]]
CLAUSE_WINDOWS = False
PMI_MIN = 2  # concept pairs seen together in fewer facts get no association
PAR = 1  # paragraph rows: each fact with PAR neighbours on each side (same section)

FACT_NEG = {"not", "cannot", "can't", "don't", "doesn't", "isn't", "aren't", "never", "won't", "didn't", "wasn't",
            "weren't", "neither", "nor"}
CLAUSE = re.compile(r"[,;:.()]")


def fact_clauses(f):
    """Words of each clause of a fact, leaving out the rest of a clause after a negator
    ('X is not Y, ...' keeps only X)."""
    out = []
    for clause in CLAUSE.split(f):
        cur = []
        for t in words(clause):
            if t in FACT_NEG:
                break
            cur.append(t)
        out.append(cur)
    return out


def fact_words(f):
    return [t for c in fact_clauses(f) for t in c]


def kb_file_sections():
    """Facts grouped by '#' section headers of the kb files."""
    sections = []
    for path in sorted((HERE / "kb").glob("*.txt")):
        cur = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#"):
                if cur:
                    sections.append(cur)
                cur = []
            elif line:
                cur.append(line.split("#", 1)[0].strip())
        if cur:
            sections.append(cur)
    return sections


def isa_facts():
    """Each head word of isa.txt with its categories becomes a short fact ('whale mammal vertebrate animal')."""
    out = []
    for line in read_lines(HERE / "isa.txt"):
        head, _, rest = line.partition(":")
        seen = set()
        for h in words(head):
            if h not in seen and not h.endswith("s") or h in ("bass", "octopus", "walrus", "hippopotamus", "rhinoceros"):
                seen.add(h)
                out.append(h + " " + rest.strip())
    return out


def kb_sections():
    return kb_file_sections() + ([isa_facts()] if ISA_FACTS else [])


def kb_lines():
    return [f for sec in kb_sections() for f in sec]


def isa_pairs(cm):
    pairs = []
    for line in read_lines(HERE / "isa.txt"):
        head, _, rest = line.partition(":")
        targets = [k for k in (cm.key(t) for t in words(rest)) if k]
        for h in words(head):
            hk = cm.key(h)
            if hk:
                pairs += [(hk, t) for t in targets if t != hk]
    return pairs


def sparse_rows(rows_of_ids, ncols):
    r, c = [], []
    for i, ids in enumerate(rows_of_ids):
        for j in sorted(set(ids)):
            r.append(i)
            c.append(j)
    return torch.sparse_coo_tensor(torch.tensor([r, c]), torch.ones(len(r)), (len(rows_of_ids), ncols)).coalesce()


def build():
    vocab = [line.strip() for line in (HERE / "vocab.txt").read_text().splitlines()]
    cm = Concepts()
    facts = kb_lines()
    seqs, cseqs = [], []
    cid = {"": 0}
    for f in facts:
        seq, cl = [], []
        for clause in fact_clauses(f):
            cs = []
            for t in clause:
                k = cm.key(t)
                if k and (not cs or cs[-1] != k):
                    cs.append(k)
                e = cm.exact(t)
                if e:
                    cs.append(e)
            cl.append(cs)
            seq += cs
        seqs.append(seq)
        cseqs.append(cl)
        for k in seq:
            cid.setdefault(k, len(cid))
    pairs = isa_pairs(cm)
    for a, b in pairs:
        cid.setdefault(a, len(cid))
        cid.setdefault(b, len(cid))
    C, N = len(cid), len(facts)

    fact_ids = [[cid[k] for k in s] for s in seqs]
    clause_ids = [[cid[k] for k in c] for cl in cseqs for c in cl if len(set(c)) >= 2]

    paragraphs, i = [], 0
    for sec in kb_sections():
        ids = fact_ids[i:i + len(sec)]
        for j in range(len(sec)):
            paragraphs.append([c for f in ids[max(0, j - PAR):j + PAR + 1] for c in f])
        i += len(sec)
    windows = []
    for ids in (fact_ids if not CLAUSE_WINDOWS else clause_ids):
        if len(ids) <= WINDOW:
            windows.append(ids)
        else:
            windows += [ids[i:i + WINDOW] for i in range(len(ids) - WINDOW + 1)]
    df = torch.zeros(C)
    for ids in fact_ids:
        for j in set(ids):
            df[j] += 1
    idf = torch.where(df > 0, torch.log(1.0 + N / df.clamp(min=1)), torch.zeros(C))
    idf[0] = 0.0
    F = sparse_rows(fact_ids, C)
    # concept association: positive PMI of two concepts occurring in the same fact
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        co = torch.sparse.mm(F.t(), F).coalesce()
    ci, cv = co.indices(), co.values()
    na = df.clamp(min=1.0)
    pmi = torch.log(N * cv / (na[ci[0]] * na[ci[1]]))
    keep = (pmi > 0) & (ci[0] != ci[1]) & (cv >= PMI_MIN)
    PMI = torch.sparse_coo_tensor(ci[:, keep], pmi[keep], (C, C)).coalesce()
    Fw = sparse_rows(windows, C)
    Fp = sparse_rows(paragraphs, C)
    Fc = sparse_rows(clause_ids, C)

    def norm(M):
        mass = torch.sparse.mm(M, idf[:, None])[:, 0]
        return mass.clamp(min=1.0) ** -PARAMS["fact_len_pow"]

    ant = []
    for line in read_lines(HERE / "antonyms.txt"):
        left, _, right = line.partition("|")
        for x in words(left):
            for y in words(right):
                kx, ky = cm.key(x), cm.key(y)
                if kx in cid and ky in cid and kx != ky:
                    ant += [(cid[kx], cid[ky]), (cid[ky], cid[kx])]
    ant = sorted(set(ant))
    A = torch.sparse_coo_tensor(torch.tensor(ant).t(), torch.ones(len(ant)), (C, C)).coalesce()
    polar = torch.zeros(C)
    polar[sorted({x for x, _ in ant})] = 1.0
    rel = set()
    for line in read_lines(HERE / "related.txt"):
        ks = sorted({cid[k] for k in (cm.key(t) for t in words(line)) if k in cid})
        rel |= {(x, y) for x in ks for y in ks if x != y}
    rel = sorted(rel)
    R = torch.sparse_coo_tensor(torch.tensor(rel).t(), torch.ones(len(rel)), (C, C)).coalesce()
    E = torch.sparse_coo_tensor(torch.tensor([[cid[a] for a, b in pairs], [cid[b] for a, b in pairs]]),
                                torch.ones(len(pairs)), (C, C)).coalesce()
    E = torch.sparse_coo_tensor(E.indices(), torch.ones(E.indices().shape[1]), (C, C)).coalesce()

    ND = len(DIRECTIONAL)
    dir_of = torch.full((C,), ND, dtype=torch.long)
    for d, group in enumerate(DIRECTIONAL):
        for wd in group:
            k = cm.key(wd)
            if k in cid:
                dir_of[cid[k]] = d
    FP = torch.zeros(N, ND * ND)
    for i, ids in enumerate(fact_ids):
        ds = [int(dir_of[c]) for c in ids if int(dir_of[c]) < ND]
        for a in range(len(ds)):
            for b in range(a + 1, len(ds)):
                if ds[a] != ds[b]:
                    FP[i, ds[a] * ND + ds[b]] = 1.0

    V = len(vocab) + 2
    concept = torch.zeros(V, dtype=torch.long)
    concept2 = torch.zeros(V, dtype=torch.long)
    for i, wd in enumerate(vocab):
        k = cm.key(wd)
        concept[i + 2] = cid.get(k, 0) if k else 0
        e = cm.exact(wd)
        concept2[i + 2] = cid.get(e, 0) if e else 0
    index = {wd: i + 2 for i, wd in enumerate(vocab)}

    def indicator(ws):
        v = torch.zeros(V)
        for wd in ws:
            if wd in index:
                v[index[wd]] = 1.0
        return v

    w = {"dir_of": dir_of, "FP": FP.to_sparse(), "dir_eye": torch.eye(ND),
         "A": A, "polar": polar, "R": R, "oneg": indicator(["no", "not", "never", "cannot", "can't", "don't", "doesn't", "isn't", "aren't",
                                           "won't", "without", "nothing", "none", "neither", "nor", "didn't", "wasn't"]),
         "concept": concept, "concept2": concept2, "Fc": Fc, "Fc_norm": norm(Fc), "idf": idf, "F": F, "F_norm": norm(F), "Fw": Fw, "Fw_norm": norm(Fw), "Fp": Fp, "Fp_norm": norm(Fp),
         "Et": E.t().coalesce(), "neg_late": indicator(NEG_LATE), "neg_any": indicator(NEG_ANY),
         "neg_cancel": indicator(NEG_CANCEL), "wh": indicator(WH), "absolute": indicator(ABSOLUTE),
         "least": indicator(["least"]),
         "seq_big": indicator(["largest", "biggest", "greatest", "most", "highest", "longest", "farthest", "oldest",
                               "hottest", "heaviest", "fastest", "deepest", "outermost"]),
         "seq_small": indicator(["smallest", "least", "lowest", "shortest", "closest", "nearest", "youngest",
                                 "coldest", "coolest", "lightest", "slowest", "shallowest", "innermost"]),
         "var_verb": indicator(["affect", "affects", "affected", "influence", "influences", "changes", "change"]),
         "var_noun": indicator(["effect", "effects", "impact"]),
         "var_mark": indicator(["whether", "how", "if", "does", "do", "that", "what", "which", "to"]),
         "var_on": indicator(["on", "upon"]),
         "var_stop": indicator(["which", "what", "why", "where", "who"]),
         "var_q": indicator(["variable", "variables", "change", "changed", "measured", "measure", "same", "constant",
                             "controlled", "control", "kept", "vary", "varied", "manipulated", "record", "observe"]),
         "var_ind": indicator(["independent", "manipulated", "manipulate", "change", "changed", "vary", "varied",
                               "test", "tested", "testing"]),
         "var_dep": indicator(["dependent", "responding", "measured", "measure", "observe", "observed", "record",
                               "recorded", "outcome"]),
         "var_ctl": indicator(["controlled", "control", "constant", "same", "unchanged", "kept", "keep"]),
         "ar_div": indicator(["speed", "density", "rate", "velocity", "pressure", "each", "average"]),
         "ar_mul": indicator(["work", "area", "times", "far", "product", "distance", "energy", "travel"]),
         "ar_add": indicator(["total", "altogether", "sum", "combined", "together", "mixes", "mixed", "added", "adds"]),
         "ar_sub": indicator(["difference", "change", "changed", "increased", "decreased", "rose", "dropped",
                              "warmer", "colder", "longer", "shorter", "more", "less", "left", "remain", "remaining"]),
         "ar_mean": indicator(["average", "mean"]),
         "qr_pass": indicator(["rate", "amount", "amounts", "number", "level", "levels", "speed", "size", "chance"]),
         "qr_wh": indicator(["which", "what", "how", "why", "where", "when", "who"]),
         "qr_goal": indicator(["want", "wants", "wanted", "need", "needs", "needed", "trying", "goal"]),
         "qr_why": indicator(["why", "because", "reason", "reasons", "explain", "explains", "explanation", "purpose"]),
         "qcue": indicator(["into", "change", "changes", "changed", "changing", "transform", "transforms",
                            "transformed", "transformation", "transformations", "convert", "converts", "converted",
                            "conversion", "becomes", "become", "turns", "turned", "from"]),
         "ocue": indicator(["to", "into", "then", "from", "becomes"]), "likely": indicator(["likely", "probably"])}
    w["num_val"] = torch.tensor([0.0, 0.0] + [float(x) if x.isdigit() else 0.0 for x in vocab])
    w["num_is"] = torch.tensor([0.0, 0.0] + [1.0 if x.isdigit() else 0.0 for x in vocab])
    w["num_3d"] = torch.tensor([0.0, 0.0] + [1.0 if x.isdigit() and len(x) == 3 else 0.0 for x in vocab])
    w.update(build_qr(vocab))
    w.update(build_seq(vocab))
    for k, v in PARAMS.items():
        w[k] = torch.tensor(v)
    # sparse matrices are stored as their index and value lists; arch.py rebuilds them
    w["PMI"] = PMI
    for k in ("F", "Fw", "Fp", "Fc", "A", "R", "Et", "FP", "PMI"):
        M = w.pop(k).coalesce()
        w[k + "_i"] = M.indices()
        w[k + "_v"] = M.values()
    return w
