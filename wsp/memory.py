"""MMLU answered by a written key-value memory: one architecture `mem-<subject>` per MMLU subject.

Nothing is pretrained. The writer delivers facts.txt:
  cue words => answer words     one memory unit: the cue is matched against the question, the answer against
                                each option
  = word word word              words that count as the same word
build() turns the file into the weights (two sparse unit-by-word matrices and the word list), bags() turns
question and option text into the network input, forward() scores the four options.
"""
import re

import torch

SUBJECTS = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine", "college_physics",
    "computer_security", "conceptual_physics", "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine",
    "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology",
    "world_religions")
DELIVERABLE = "facts.txt"
TOKEN = re.compile(r"[a-z0-9']+")
MAX_LINES = 20000
MAX_LINE_CHARS = 300
THETA = 0.3  # a unit is silent unless its cue matches the question at least this well (cosine)


def words(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def parse(text: str) -> tuple[list[tuple[list[str], list[str]]], list[list[str]], list[str]]:
    """(units as (cue words, answer words), groups of same words, reasons the file cannot be used)."""
    units, groups, problems = [], [], []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > MAX_LINES:
        problems.append(f"{len(lines)} lines exceed the limit of {MAX_LINES}")
    for i, line in enumerate(lines, 1):
        if len(line) > MAX_LINE_CHARS:
            problems.append(f"line {i} exceeds {MAX_LINE_CHARS} characters")
        elif line.startswith("="):
            groups.append(words(line))
        elif line.count("=>") == 1 and all(words(part) for part in line.split("=>")):
            cue, answer = line.split("=>")
            units.append((words(cue), words(answer)))
        else:
            problems.append(f"line {i} is neither `cue words => answer words` nor `= word word`: {line[:60]}")
    if not units:
        problems.append("no memory units")
    return units, groups, problems[:20]


def build(text: str) -> dict:
    """facts.txt -> weights. A word's weight is log(1 + units / units containing it); rows have unit length."""
    units, groups, _ = parse(text)
    same = {w: g[0] for g in groups for w in g}
    units = [([same.get(w, w) for w in cue], [same.get(w, w) for w in answer]) for cue, answer in units]
    vocab = sorted({w for cue, answer in units for w in cue + answer})
    index = {w: i for i, w in enumerate(vocab)}
    df = torch.zeros(len(vocab))
    for cue, answer in units:
        df[[index[w] for w in set(cue + answer)]] += 1
    idf = torch.log(1 + len(units) / df)

    def rows(side: int) -> torch.Tensor:
        r, c = zip(*[(j, index[w]) for j, unit in enumerate(units) for w in set(unit[side])])
        m = torch.sparse_coo_tensor(torch.tensor([r, c]), idf[list(c)], (len(units), len(vocab))).coalesce()
        norm = torch.sparse.sum(m * m, dim=1).to_dense().sqrt()
        return torch.sparse_coo_tensor(m.indices(), m.values() / norm[m.indices()[0]], m.shape).coalesce()

    return {"keys": rows(0), "values": rows(1), "idf": idf, "vocab": vocab, "same": same}


def bags(w: dict, texts: list[str]) -> torch.Tensor:
    """(len(texts), words) unit-length rows: each known word of a text once, weighted like the memory rows."""
    index = {v: i for i, v in enumerate(w["vocab"])}
    out = torch.zeros(len(texts), len(w["vocab"]))
    for r, text in enumerate(texts):
        ids = sorted({index[v] for v in (w["same"].get(t, t) for t in words(text)) if v in index})
        out[r, ids] = w["idf"][ids]
    return out / out.norm(dim=1, keepdim=True).clamp(min=1e-9)


def forward(w, q, o):
    """q: (n, words) question bags, o: (n, 4, words) option bags -> (n, 4) logits."""
    active = torch.relu(torch.sparse.mm(w["keys"], q.T).T - THETA)                    # (n, units)
    match = torch.sparse.mm(w["values"], o.reshape(-1, o.shape[2]).T).T.reshape(o.shape[0], 4, -1)
    return torch.einsum("nj,nkj->nk", active, match)


def evaluate(w: dict, questions: list[str], options: list[list[str]], answers) -> dict:
    """answered: share of questions where the options get different logits. accuracy counts an unanswered
    question as a uniform guess (1/4)."""
    q = bags(w, questions)
    o = bags(w, [x for opts in options for x in opts]).reshape(len(questions), 4, -1)
    logits = forward(w, q, o)
    answered = (logits.max(1).values > logits.min(1).values).numpy()
    right = (logits.argmax(1).numpy() == answers) & answered
    return {"accuracy": float(right.mean() + 0.25 * (1 - answered.mean())), "answered": float(answered.mean()),
            "accuracy_answered": float(right.sum() / max(answered.sum(), 1)), "n": int(len(answers))}


def n_weights(w: dict) -> int:
    return int(w["keys"].values().numel() + w["values"].values().numel() + len(w["vocab"]))


def _mem_shapes():
    return {"keys": ("units", "words"), "values": ("units", "words"), "idf": ("words",)}
