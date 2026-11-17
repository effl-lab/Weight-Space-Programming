"""Generations of a writer-designed model. free-<task> is generation 1 and free<g>-<task> generation g, where <task>
is an MMLU subject, "arc", "mb" (MathBench, generation 1 only), "mm" (MMLU, all subjects, generation 1 only), "sq" (SQuAD 2.0, generation 1 only) or "gen" (SciQ
generation, generation 1 only). An MMLU generation 1 starts from the memory mem-<subject>; the other tasks'
generation 1 has no parent. Importable without torch.
"""
import re

PATTERN = re.compile(r"free(\d*)-([a-z_]+)")
MAX_GENERATION = 10
# Variants of ARC generation 1, <variant>-arc: par asks for a structure that runs well on parallel hardware, noq
# forbids running the model on made-up questions, conv rewrites the writer's best last-generation model for parallel
# hardware (its parent), nofact forbids a store of facts or associations that is matched against the question, fb
# ("feedback") lets the writer measure its model on labelled practice questions (wsp.arc.practice), py lets arch.py be
# any Python over the raw texts (wsp.free_model.run_py), fit lets the writer fit the model on the ARC training
# questions (wsp.arc.training).
ARC_VARIANTS = ("par", "noq", "conv", "conv1", "nofact", "fb", "py", "fit")
CONVERT_VARIANTS = ("conv", "conv1")  # rewrite a parent for parallel hardware: the last generation, or generation 1
# par and noq also run as lineages: <variant><g>-arc is generation g >= 2 and edits the best valid run of generation
# g - 1 (<variant>-arc for g = 2), with the variant's task text.
LINEAGE_VARIANTS = ("par", "noq")
VARIANT_PATTERN = re.compile(r"(par|noq)(\d*)-arc")
# A given architecture can run as a lineage too: <name>_g<g>-arc is generation g >= 2, keeps the given arch.py and
# starts from the best valid run of generation g - 1 (<name>-arc for g = 2).
GIVEN_LINEAGES = ("mem_assoc2", "mem_route")
GIVEN_PATTERN = re.compile(rf"({'|'.join(GIVEN_LINEAGES)})(?:_g(\d+))?-arc")


def parse(arch: str) -> tuple[int, str] | None:
    """(generation, task) of a free-design architecture, None for any other."""
    m = PATTERN.fullmatch(arch)
    if m is None or m[1] in ("0", "1"):
        return None
    return int(m[1] or 1), m[2]


def arc_variant(arch_name: str) -> str | None:
    if (v := variant_lineage(arch_name)) is not None:
        return v[0]
    name = arch_name.removesuffix("-arc")
    return name if arch_name.endswith("-arc") and name in ARC_VARIANTS else None


def variant_lineage(arch_name: str) -> tuple[str, int] | None:
    """(variant, generation) of a par or noq architecture (<variant>-arc is generation 1), None for any other."""
    m = VARIANT_PATTERN.fullmatch(arch_name)
    if m is None or m[2] in ("0", "1"):
        return None
    return m[1], int(m[2] or 1)


def variant_arch(variant: str, generation: int) -> str:
    return f"{variant}-arc" if generation == 1 else f"{variant}{generation}-arc"


def given_lineage(arch_name: str) -> tuple[str, int] | None:
    """(given architecture, generation) of a given-architecture lineage (<name>-arc is generation 1), None for any
    other."""
    m = GIVEN_PATTERN.fullmatch(arch_name)
    if m is None or m[2] in ("0", "1"):
        return None
    return m[1], int(m[2] or 1)


def given_arch(name: str, generation: int) -> str:
    return f"{name}-arc" if generation == 1 else f"{name}_g{generation}-arc"


def arch(generation: int, task: str) -> str:
    return f"free-{task}" if generation == 1 else f"free{generation}-{task}"


def parent_arch(arch_name: str) -> str | None:
    generation, task = parse(arch_name)
    if generation > 1:
        return arch(generation - 1, task)
    return None if task in ("arc", "mb", "mm", "sq", "gen") else f"mem-{task}"
