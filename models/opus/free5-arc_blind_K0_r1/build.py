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
NEG_LATE = ["avoid", "avoided", "not", "isn't", "aren't", "doesn't", "wouldn't", "won't", "unlikely", "cannot", "can't", "couldn't"]
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
    "oq": 0.25,             # weight kept by option words that also appear in the question
    "qant_damp": 0.7,      # rows with only the opposite of a polar question word count exp(-this)
    "ant": 0.3,            # polarity: rows count exp(+-ant) when an option agrees / disagrees with the row's polarity
    "exact_w": 0.3,        # weight of the literal form of a word the synonym table merges
    "clause": 0.5,         # weight of the best single clause match
    "share": 0.5,
    "qr_w": 1.0,
    "ar_w": 2.0,
    "var_w": 2.0,          # weight of the experiment-variables rule           # weight of the arithmetic match of a numeric option
    "pmi_w": 0.2,          # weight of the concept association score
    "seq_w": 0.5,          # weight of the order agreement of listed items (seq.txt)
    "fw_w": 1.5,           # weight of the food-web rule (relations stated in the question)
    "ord_w": 0.3,
    "negch_w": 0.5,        # weight of the negative-fact channel (arch.neg_channel)
    "negch_neg_w": 1.0,    # its weight in negated questions
    "negch_pos": 0.5,
    "than_w": 0.3,         # option-side weight of a concept a fact names only after "than"
    "weak_w": 0.5,         # weight of question frame words (WEAK)      # its share for a plain option in a plain question (the noisiest case)
            # weight of the word order check for options with the same words
    "ext_w": 2.0,          # weight of the place of the option in a sequence whose end the question asks for           # weight of the qualitative-reasoning score (cause -> effect directions, qr.txt)          # option words shared by several options are down-weighted by share x (count-1)/3
}
WINDOW = 4
ISA_FACTS = True  # also turn isa.txt lines into short facts
# Concepts whose order matters ("changes electrical energy into sound", "from a liquid to a gas").
DIRECTIONAL = [["solid", "ice", "frozen"], ["liquid"], ["gas", "vapor", "steam", "gases"],
               ["electric", "electrical", "electricity"], ["light", "radiant", "solar", "sunlight"], ["sound"],
               ["heat", "thermal"], ["chemical"], ["mechanical", "motion", "kinetic", "movement"], ["potential"],
               ["nuclear"]]
PMI_MIN = 2  # concept pairs seen together in fewer facts get no association
PAR = 1  # paragraph rows: each fact with PAR neighbours on each side (same section)

POLAR_CAT = ["conductor", "insulator", "conduct", "insulate", "renewable", "nonrenewable", "living", "nonliving",
             "biotic", "abiotic", "acid", "base", "acidic", "basic", "physical", "chemical", "solid", "liquid", "gas",
             "predator", "prey", "producer", "consumer", "herbivore", "carnivore", "dominant", "recessive",
             "inherited", "learned", "acquired", "transparent", "opaque", "vertebrate", "invertebrate", "mitosis",
             "meiosis", "artery", "vein", "sexual", "asexual", "series", "parallel", "convex", "concave", "summer",
             "winter", "attract", "repel", "melt", "freeze", "evaporate", "condense", "positive", "negative",
             "proton", "electron", "erosion", "deposition", "constructive", "destructive", "divergent",
             "convergent", "inner", "outer", "harmful", "helpful", "weathering", "mechanical", "igneous",
             "sedimentary", "metamorphic", "rotation", "revolution", "rotate", "revolve", "endothermic",
             "exothermic", "conduction", "convection", "radiation", "weather", "climate", "element", "compound",
             "mixture", "virus", "bacteria", "independent", "dependent", "omnivore", "mammal", "reptile",
             "amphibian", "bird", "fish", "insect", "spider", "arachnid", "decomposer", "evaporation",
             "condensation", "precipitation", "mass", "weight", "plant", "animal"]
# question frame words: rare in the facts (so their idf is high) but they say how the question asks, not what
# it is about ("What is the main purpose of ...", "Which helps ...")
WEAK = ["main", "mainly", "important", "importance", "purpose", "purposes", "role", "roles", "primary", "primarily",
        "major", "mostly", "responsible", "directly", "help", "helps", "helped", "helping", "occur", "occurs",
        "occurred", "occurring", "happen", "happens", "happened", "result", "results", "resulted", "affect",
        "affects", "affected", "effect", "effects", "determine", "determines", "determined", "factor", "factors",
        "feature", "features", "benefit", "benefits", "typically", "common", "commonly", "generally", "usually",
        "process", "processes", "important", "necessary", "needed", "allow", "allows"]
# organisms for the food-web rule, besides the animal and plant head words of isa.txt
ORGANISMS = ["insect", "insects", "plant", "plants", "seed", "seeds", "shrub", "shrubs", "bush", "bushes", "clover",
             "kelp", "urchin", "urchins", "plankton", "zooplankton", "krill", "berries", "berry", "acorn", "acorns",
             "nuts", "leaves", "weeds", "fish", "bird", "birds", "bacteria", "fungi", "hay", "nectar", "fruit",
             "crops", "grain", "perch", "bluegill", "larvae", "tadpole", "tadpoles", "egret", "egrets", "lynx",
             "hare", "hares", "caribou", "lemming", "lemmings", "vole", "voles", "falcon", "falcons", "gazelle",
             "gazelles", "antelope", "wildebeest", "mussels", "algae", "phytoplankton", "worms", "snails", "slugs"]
ORG_CATS = {"animal", "plant", "fungus", "producer", "consumer", "decomposer", "herbivore", "carnivore", "omnivore"}
NEGW = {"no", "not", "never", "cannot", "can't", "don't", "doesn't", "isn't", "aren't", "won't", "without", "nothing",
        "none", "neither", "nor", "didn't", "wasn't", "lack", "lacks", "lacking"}
FACT_NEG = {"not", "cannot", "can't", "don't", "doesn't", "isn't", "aren't", "never", "won't", "didn't", "wasn't",
            "weren't", "neither", "nor"}
NEG_END = {"because", "so", "since", "while", "but", "although", "though", "which", "when", "if", "unless",
           "until", "as", "whereas", "instead"}
NEG_COND = {"if", "when", "unless", "until", "whenever", "without"}
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


def sparse_rows(rows_of_ids, ncols, low=None, low_w=1.0):
    """0/1 rows; with low (one set of ids per row), those ids get low_w instead of 1."""
    r, c, v = [], [], []
    for i, ids in enumerate(rows_of_ids):
        for j in sorted(set(ids)):
            r.append(i)
            c.append(j)
            v.append(low_w if low is not None and j in low[i] else 1.0)
    return torch.sparse_coo_tensor(torch.tensor([r, c]), torch.tensor(v), (len(rows_of_ids), ncols)).coalesce()


def build():
    vocab = [line.strip() for line in (HERE / "vocab.txt").read_text().splitlines()]
    cm = Concepts()
    facts = kb_lines()
    seqs, cseqs = [], []
    # concepts that a fact names only after "than" ("spreads faster in warm water than in cold water"): the
    # reference of a comparison, which is weak support for an option
    than_f, than_c = [], []
    cid = {"": 0}
    for f in facts:
        seq, cl = [], []
        fin, fout = set(), set()
        for clause in fact_clauses(f):
            cs = []
            cin, cout = set(), set()
            after = False
            for t in clause:
                if t == "than":
                    after = True
                k = cm.key(t)
                if k and (not cs or cs[-1] != k):
                    cs.append(k)
                e = cm.exact(t)
                if e:
                    cs.append(e)
                for x in (k, e):
                    if x:
                        (cin if after else cout).add(x)
            cl.append(cs)
            than_c.append(cin - cout)
            fin |= cin
            fout |= cout
            seq += cs
        than_f.append(fin - fout)
        # a fact that states an absence or a negation ("the Moon has no atmosphere") carries the concept NEG,
        # which negated options carry too
        if any(t in NEGW for t in words(f)):
            seq = seq + ["=NEG"]
            cl = cl[:-1] + [cl[-1] + ["=NEG"]] if cl else [["=NEG"]]
        seqs.append(seq)
        cseqs.append(cl)
        for k in seq:
            cid.setdefault(k, len(cid))
    # negative clauses ("magnets do not attract wood, plastic, ..."): the words after the negator (N part) and
    # the rest of the fact (P part) become rows of a separate channel, used with a sign in arch.neg_channel
    negP, negN = [], []
    for fi, f in enumerate(facts):
        parts = CLAUSE.split(f)
        seps = CLAUSE.findall(f)
        for ci, clause in enumerate(parts):
            ws = words(clause)
            idx = [i for i, t in enumerate(ws) if t in FACT_NEG]
            # a negation inside a condition ("when a plant does not get enough water, it wilts") denies nothing
            if not idx or any(t in NEG_COND for t in ws[:idx[0]]):
                continue
            post = []
            for t in ws[idx[0] + 1:]:
                # the denied part ends where a reason or another clause begins ("do not move it because ...")
                if t in NEG_END:
                    break
                k = cm.key(t)
                if k and k not in post:
                    post.append(k)
            if post:
                # what the denial is said of: the sentence of the negated clause (up to
                # ";" or "."), so "...; fossils are found in sedimentary rocks" is not the subject of "igneous
                # rocks do not contain fossils"
                lo = ci
                while lo > 0 and seps[lo - 1] == ",":
                    lo -= 1
                hi = ci
                while hi < len(seps) and seps[hi] == ",":
                    hi += 1
                sent = [cm.key(t) for c in parts[lo:hi + 1] for t in words(c) if t not in FACT_NEG]
                negP.append([k for k in dict.fromkeys(sent) if k and k not in post])
                negN.append(post)
    for r in negN:
        for k in r:
            cid.setdefault(k, len(cid))
    pairs = isa_pairs(cm)
    for a, b in pairs:
        cid.setdefault(a, len(cid))
        cid.setdefault(b, len(cid))
    C, N = len(cid), len(facts)

    fact_ids = [[cid[k] for k in s] for s in seqs]
    clause_ids = [[cid[k] for k in c] for cl in cseqs for c in cl if len(set(c)) >= 2]
    flat_cl = [c for cl in cseqs for c in cl]
    clause_low = [{cid[k] for k in than_c[i]} for i, c in enumerate(flat_cl) if len(set(c)) >= 2]
    fact_low = [{cid[k] for k in t} for t in than_f]

    paragraphs, par_low, i = [], [], 0
    for sec in kb_sections():
        ids = fact_ids[i:i + len(sec)]
        for j in range(len(sec)):
            paragraphs.append([c for f in ids[max(0, j - PAR):j + PAR + 1] for c in f])
            others = {c for jj in range(max(0, j - PAR), min(len(sec), j + PAR + 1)) if jj != j for c in ids[jj]}
            par_low.append(fact_low[i + j] - others)
        i += len(sec)
    windows, win_low = [], []
    for fi, ids in enumerate(fact_ids):
        if len(ids) <= WINDOW:
            windows.append(ids)
            win_low.append(fact_low[fi])
        else:
            windows += [ids[i:i + WINDOW] for i in range(len(ids) - WINDOW + 1)]
            win_low += [fact_low[fi]] * (len(ids) - WINDOW + 1)
    df = torch.zeros(C)
    for ids in fact_ids:
        for j in set(ids):
            df[j] += 1
    for P_, N_ in zip(negP, negN):
        for j in {cid[k] for k in P_ + N_} - {cid[k] for k in P_}:
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
    tw = PARAMS["than_w"]
    # the same rows for the option side of support(), with the "than" references weakened (same indices)
    low_v = {"F": sparse_rows(fact_ids, C, fact_low, tw).values(), "Fw": sparse_rows(windows, C, win_low, tw).values(),
             "Fp": sparse_rows(paragraphs, C, par_low, tw).values(), "Fc": sparse_rows(clause_ids, C, clause_low, tw).values()}
    FnP = sparse_rows([[cid[k] for k in r] for r in negP], C)
    FnN = sparse_rows([[cid[k] for k in r] for r in negN], C)
    FnA = sparse_rows([[cid[k] for k in a + b] for a, b in zip(negP, negN)], C)

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
    # category opposites (not amounts or directions, whose opposite rows are evidence by sign)
    polar_cat = torch.zeros(C)
    for wd in POLAR_CAT:
        k = cm.key(wd)
        if k in cid:
            polar_cat[cid[k]] = 1.0
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

    # concepts of each fact in order (for the word order check of arch.order_score)
    LF = 40
    Ftok = torch.zeros(N, LF, dtype=torch.long)
    for i, sq in enumerate(seqs):
        ids = [cid[k] for k in sq if not k.startswith("~")][:LF]
        Ftok[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    V = len(vocab) + 2
    concept = torch.zeros(V, dtype=torch.long)
    concept2 = torch.zeros(V, dtype=torch.long)
    for i, wd in enumerate(vocab):
        k = cm.key(wd)
        concept[i + 2] = cid.get(k, 0) if k else 0
        e = cm.exact(wd)
        concept2[i + 2] = cid.get(e, 0) if e else 0
        if wd in NEGW:
            concept2[i + 2] = cid["=NEG"]
    index = {wd: i + 2 for i, wd in enumerate(vocab)}

    def indicator(ws):
        v = torch.zeros(V)
        for wd in ws:
            if wd in index:
                v[index[wd]] = 1.0
        return v

    qweak = torch.ones(C)
    for wd in WEAK:
        k = cm.key(wd)
        if k in cid:
            qweak[cid[k]] = PARAMS["weak_w"]
    notneg = torch.ones(C)
    notneg[cid["=NEG"]] = 0.0
    w = {"polar_cat": polar_cat, "notneg": notneg, "qweak": qweak, "Ftok": Ftok, "dir_of": dir_of, "FP": FP.to_sparse(), "dir_eye": torch.eye(ND),
         "A": A, "polar": polar, "R": R, "oneg": indicator(["no", "not", "never", "cannot", "can't", "don't", "doesn't", "isn't", "aren't",
                                           "won't", "without", "nothing", "none", "neither", "nor", "didn't", "wasn't"]),
         "concept": concept, "Fn_norm": norm(FnA), "FnN_mass": torch.sparse.mm(FnN, idf[:, None])[:, 0], "concept2": concept2, "Fc": Fc, "Fc_norm": norm(Fc), "idf": idf, "F": F, "F_norm": norm(F), "Fw": Fw, "Fw_norm": norm(Fw), "Fp": Fp, "Fp_norm": norm(Fp),
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
         "var_ctx": indicator(["experiment", "experiments", "experimental", "investigation", "investigations",
                               "investigate", "investigates", "investigating", "variable", "variables", "test",
                               "tests", "tested", "testing", "hypothesis", "measure", "measures", "measured",
                               "measuring", "measurement", "measurements", "data", "trial", "trials", "procedure",
                               "study", "studies", "studying", "studied", "record", "records", "recorded", "recording",
                               "observe", "observes", "observed", "control", "controlled", "constant", "fair",
                               "students", "student", "scientist", "scientists", "class", "researcher", "researchers",
                               "groups", "group", "setup", "same", "different", "pots", "plants", "conclusion"]),
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
         "qr_than": indicator(["than"]),
         "qr_from": indicator(["from"]),
         "qr_agent": indicator(["kill", "kills", "killed", "killing", "destroy", "destroys", "destroyed", "damage",
                                "damages", "damaged", "harm", "harms", "harmed", "infect", "infects", "infected",
                                "poison", "poisons", "poisoned", "wipe", "wipes", "wiped"]),
         "qr_motion": indicator(["fall", "falls", "fell", "falling", "drop", "drops", "dropped", "dropping"]),
         "qr_to": indicator(["to", "into"]),
         "qr_loss": indicator(["loss", "losses"]),
         "qr_affect": indicator(["affect", "affects", "affected", "effect", "effects", "happen", "happens",
                                 "happened", "result", "results"]),
         "qr_cond": indicator(["if", "when", "as", "after", "while", "because", "since", "during"]),
         "qr_pron": indicator(["it", "they", "its", "this", "these"]),
         "qr_link": indicator(["become", "becomes", "became", "becoming", "get", "gets", "got", "getting", "grow",
                               "grows", "grew", "is", "are", "was", "were", "be", "been", "stays", "remains"]),
         "var_of": indicator(["of"]),
         "var_generic": indicator(["amount", "amounts", "number", "numbers", "type", "types", "kind", "kinds",
                                   "brand", "brands", "brands", "variety"]),
         "var_diff": indicator(["different", "various", "several", "varying", "increasing", "decreasing"]),
         "var_meas": indicator(["measures", "measured", "measuring", "records", "recorded", "recording", "observes",
                                "counts", "counted", "times", "timed", "compares", "observed"]),
         "eat_v": indicator(["eat", "eats", "ate", "eating", "feed", "feeds", "fed", "feeding", "hunt", "hunts",
                             "hunting", "consume", "consumes", "consumed"]),
         "eat_pass": indicator(["eaten", "hunted", "consumed", "preyed"]),
         "eat_by": indicator(["by"]),
         "qr_pass": indicator(["rate", "amount", "amounts", "number", "level", "levels", "speed", "size", "chance"]),
         "qr_wh": indicator(["which", "what", "how", "why", "where", "when", "who"]),
         "qr_goal": indicator(["want", "wants", "wanted", "need", "needs", "needed", "trying", "goal"]),
         "qr_why": indicator(["why", "because", "reason", "reasons", "explain", "explains", "explanation", "purpose"]),
         "qcue": indicator(["into", "change", "changes", "changed", "changing", "transform", "transforms",
                            "transformed", "transformation", "transformations", "convert", "converts", "converted",
                            "conversion", "becomes", "become", "turns", "turned", "from"]),
         "ocue": indicator(["to", "into", "then", "from", "becomes"]), "likely": indicator(["likely", "probably"])}
    orgs = set(ORGANISMS)
    for line in read_lines(HERE / "isa.txt"):
        head, _, rest = line.partition(":")
        if set(words(rest)) & ORG_CATS:
            orgs |= set(words(head))
    w["org"] = indicator(sorted(orgs))
    w["num_val"] = torch.tensor([0.0, 0.0] + [float(x) if x.isdigit() else 0.0 for x in vocab])
    w["num_is"] = torch.tensor([0.0, 0.0] + [1.0 if x.isdigit() else 0.0 for x in vocab])
    w["num_3d"] = torch.tensor([0.0, 0.0] + [1.0 if x.isdigit() and len(x) == 3 else 0.0 for x in vocab])
    w.update(build_qr(vocab))
    w.update(build_seq(vocab))
    for k, v in PARAMS.items():
        w[k] = torch.tensor(v)
    # sparse matrices are stored as their index and value lists; arch.py rebuilds them
    w["PMI"] = PMI
    w["FnP"], w["FnN"], w["FnA"] = FnP, FnN, FnA
    for k in ("F", "Fw", "Fp", "Fc", "A", "R", "Et", "FP", "PMI", "FnP", "FnN", "FnA"):
        M = w.pop(k).coalesce()
        w[k + "_i"] = M.indices()
        w[k + "_v"] = M.values()
    for k, v in low_v.items():
        w[k + "_ov"] = v
    return w
