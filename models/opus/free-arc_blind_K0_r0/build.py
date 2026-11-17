"""Turns the hand-written knowledge into weights.

kb/*.txt      one fact per line -> rows of a sparse fact x concept matrix F (whole facts) and Fw (windows of
              WINDOW consecutive content words of each fact, so that words close together count more)
isa.txt       'head words: category words' -> sparse concept x concept expansion matrix E (steel -> metal solid)
synonyms.txt  word -> concept table applied before Porter stemming
stopwords.txt words that carry no concept
Concepts are Porter stems; idf is computed from the hand-written facts.
"""
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from textproc import Concepts, read_lines, words  # noqa: E402

# Words that flip the question ("which is NOT ...") when they appear near the end of the question,
# words that flip it wherever they appear, and words that cancel a following "not" ("found in X but not Y").
NEG_LATE = ["not", "isn't", "aren't", "doesn't", "wouldn't", "won't", "unlikely"]
WH = ["which", "what", "who", "where", "how"]
ABSOLUTE = ["only", "always", "never"]  # absolute wording in an option is usually a distractor
WH_SPAN = 7
NEG_ANY = ["except"]
NEG_CANCEL = ["but", "and", "or", "if", "did", "whether", "had", "was", "were"]

PARAMS = {
    "late_n": 14.0,        # question tokens counted as "the actual question" (the end of the stem)
    "late_boost": 1.0,     # extra weight for those tokens
    "fact_len_pow": 0.35,  # long facts are penalised by (fact idf mass) ** this
    "opt_len_pow": 0.6,    # option score is matched idf / (option idf mass) ** this
    "second": 0.15,        # weight of the 2nd..kth best whole facts
    "win": 0.5,            # weight of the best window match
    "par": 0.3,            # weight of the best paragraph match
    "prf": 1.0,            # weight of the second pass with the question expanded by its best facts
    "prf_q": 1.0,          # weight of the expansion words in that pass
    "q_idf_pow": 1.0,      # question words are weighted by idf ** this (rare words discriminate)
    "cov": 0.0,            # weight of evidence collected across facts per question word
    "disc_floor": 0.6,     # weight kept by a question word that links equally to every option
    "abs_pen": 0.1,        # relative score penalty for options with absolute words
    "dir": 0.7,            # boost/penalty when the order of energy forms / states agrees / disagrees
    "isa": 1.0,            # weight of the support found through the options' categories
    "qisa": 0.3,           # weight of the question words' categories
}
WINDOW = 4
ISA_FACTS = False  # also turn isa.txt lines into short facts
# Concepts whose order matters ("changes electrical energy into sound", "from a liquid to a gas").
DIRECTIONAL = [["solid", "ice", "frozen"], ["liquid"], ["gas", "vapor", "steam", "gases"],
               ["electric", "electrical", "electricity"], ["light", "radiant", "solar", "sunlight"], ["sound"],
               ["heat", "thermal"], ["chemical"], ["mechanical", "motion", "kinetic", "movement"], ["potential"],
               ["nuclear"]]
PAR = 1  # paragraph rows: each fact with PAR neighbours on each side (same section)

FACT_NEG = {"not", "cannot", "can't", "don't", "doesn't", "isn't", "aren't", "never", "won't", "didn't", "wasn't",
            "weren't", "neither", "nor"}
CLAUSE = re.compile(r"[,;:.()]")


def fact_words(f):
    """Words of a fact, leaving out the rest of a clause after a negator ('X is not Y, ...' keeps only X)."""
    out = []
    for clause in CLAUSE.split(f):
        for t in words(clause):
            if t in FACT_NEG:
                break
            out.append(t)
    return out


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
    seqs = []
    cid = {"": 0}
    for f in facts:
        seq = []
        for k in (cm.key(t) for t in fact_words(f)):
            if k and (not seq or seq[-1] != k):
                seq.append(k)
        seqs.append(seq)
        for k in seq:
            cid.setdefault(k, len(cid))
    pairs = isa_pairs(cm)
    for a, b in pairs:
        cid.setdefault(a, len(cid))
        cid.setdefault(b, len(cid))
    C, N = len(cid), len(facts)

    fact_ids = [[cid[k] for k in s] for s in seqs]
    paragraphs, i = [], 0
    for sec in kb_sections():
        ids = fact_ids[i:i + len(sec)]
        for j in range(len(sec)):
            paragraphs.append([c for f in ids[max(0, j - PAR):j + PAR + 1] for c in f])
        i += len(sec)
    windows = []
    for ids in fact_ids:
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
    Fw = sparse_rows(windows, C)
    Fp = sparse_rows(paragraphs, C)

    def norm(M):
        mass = torch.sparse.mm(M, idf[:, None])[:, 0]
        return mass.clamp(min=1.0) ** -PARAMS["fact_len_pow"]

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
    for i, wd in enumerate(vocab):
        k = cm.key(wd)
        concept[i + 2] = cid.get(k, 0) if k else 0
    index = {wd: i + 2 for i, wd in enumerate(vocab)}

    def indicator(ws):
        v = torch.zeros(V)
        for wd in ws:
            if wd in index:
                v[index[wd]] = 1.0
        return v

    w = {"dir_of": dir_of, "FP": FP.to_sparse(), "dir_eye": torch.eye(ND),
         "concept": concept, "idf": idf, "F": F, "F_norm": norm(F), "Fw": Fw, "Fw_norm": norm(Fw), "Fp": Fp, "Fp_norm": norm(Fp),
         "Et": E.t().coalesce(), "neg_late": indicator(NEG_LATE), "neg_any": indicator(NEG_ANY),
         "neg_cancel": indicator(NEG_CANCEL), "wh": indicator(WH), "absolute": indicator(ABSOLUTE),
         "least": indicator(["least"]),
         "qcue": indicator(["into", "change", "changes", "changed", "changing", "transform", "transforms",
                            "transformed", "transformation", "transformations", "convert", "converts", "converted",
                            "conversion", "becomes", "become", "turns", "turned", "from"]),
         "ocue": indicator(["to", "into", "then", "from", "becomes"]), "likely": indicator(["likely", "probably"])}
    for k, v in PARAMS.items():
        w[k] = torch.tensor(v)
    return w
