"""Analytic weights for a two-layer, four-head science association circuit.

No optimization, examples, random initialization, downloaded resources, or data
inspection is involved.  knowledge.tsv is an explicitly authored dictionary.
"""
from pathlib import Path
import math
import re
import torch

ROOT = Path(__file__).resolve().parent
D = 16
ANCHOR = 10_000_000_000.0
SIGMA = ANCHOR * math.sqrt(15.0) / 16.0
ROLE = 100.0
TEMP = 16384.0
MATCH = 8.0
VALUE = 256.0
GAIN = 8.0
# Powers of two make equal-coordinate role contrasts exactly zero in float64.
VALUE_DIVISOR = 128.0
# Question query-role stays 2; option queries get 2 + 100/32 = 5.125.
# A unit-code option self-match is then bounded by
# TEMP - (1 + ROLE/64)*(TEMP-MATCH), well below null competitors.
ROLE_QUERY_BOOST = ROLE / 32.0


def read_word_aliases():
    """Explicit lexical equivalents; no stemming, corpus counts, or fitting."""
    aliases = {}
    for lineno, line in enumerate((ROOT / "word_aliases.tsv").read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        source, spellings = line.split("|")
        source = source.strip()
        if re.fullmatch(r"[a-z0-9']+", source) is None:
            raise ValueError(f"Invalid lexical source on line {lineno}")
        for word in spellings.split():
            if re.fullmatch(r"[a-z0-9']+", word) is None:
                raise ValueError(f"Unreachable lexical spelling: {word!r}")
            if word in aliases:
                raise ValueError(f"Duplicate lexical alias: {word!r}")
            aliases[word] = source
    return aliases


def read_knowledge():
    aliases = {}
    for line in (ROOT / "aliases.tsv").read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            alias, canonical = line.split()
            if alias in aliases:
                raise ValueError(f"Duplicate concept alias: {alias}")
            aliases[alias] = canonical
    def canonicalize(tags):
        result = []
        for tag in tags:
            visited = set()
            while tag in aliases:
                if tag in visited:
                    raise ValueError(f"Cycle in concept aliases: {tag}")
                visited.add(tag)
                tag = aliases[tag]
            if tag not in result:
                result.append(tag)
        return result
    for tag in aliases:
        canonicalize([tag])
    entries = {}
    for lineno, line in enumerate((ROOT / "knowledge.tsv").read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        queries, keys, spellings = (s.strip().split() for s in line.split("|"))
        queries = canonicalize([] if queries == ["-"] else queries)
        keys = canonicalize([] if keys == ["-"] else keys)
        if len(queries) > 2 or len(keys) > 2:
            raise ValueError(f"Too many semantic slots on line {lineno}")
        for word in spellings:
            if re.fullmatch(r"[a-z0-9']+", word) is None:
                raise ValueError(f"Unreachable tokenizer spelling: {word!r}")
            if word in entries:
                raise ValueError(f"Duplicate word {word!r} on line {lineno}")
            entries[word] = (queries, keys)
    # Lexical aliases copy existing concepts rather than adding new facts.
    # Sources must be in the authored base dictionary, ruling out alias chains.
    base_words = set(entries)
    for word, source in read_word_aliases().items():
        if source not in base_words:
            raise ValueError(f"Unknown lexical alias source: {source!r}")
        if word in entries and entries[word] != entries[source]:
            raise ValueError(f"Lexical alias changes an existing meaning: {word!r}")
        entries[word] = entries[source]
    # Unlike double quotes, ASCII single quotes survive the fixed tokenizer.
    # A quoted known word should therefore receive the same features. This is
    # a mechanical spelling rule, not an additional scientific association.
    for word, features in list(entries.items()):
        quoted = "'" + word + "'"
        if quoted in entries and entries[quoted] != features:
            raise ValueError(f"Conflicting quoted spelling: {quoted!r}")
        entries[quoted] = features
    return entries


def build():
    torch.set_num_threads(4)
    dtype = torch.float64
    def zeros(*shape):
        return torch.zeros(shape, dtype=dtype)
    entries = read_knowledge()
    vocab = (ROOT / "vocab.txt").read_text().splitlines()
    if vocab != sorted(entries):
        raise ValueError("vocab.txt must be the sorted dictionary word list")
    # Static reachability, not empirical fitting: a tag occurring on only one
    # side cannot express any association. Removing it prevents incidental
    # spherical-code proximity from giving evidence for an impossible edge.
    query_labels = {tag for qs, _ in entries.values() for tag in qs}
    key_labels = {tag for _, ks in entries.values() for tag in ks}
    active_labels = query_labels & key_labels
    entries = {word: ([tag for tag in qs if tag in active_labels],
                      [tag for tag in ks if tag in active_labels])
               for word, (qs, ks) in entries.items()}
    labels = sorted(active_labels)
    # A deterministic spherical code, not a learned embedding.  Distinct science
    # concepts have distinct directions; synonyms deliberately share directions.
    codes = {}
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i, label in enumerate(labels):
        z = 1.0 - 2.0 * (i + 0.5) / len(labels)
        r = math.sqrt(1.0 - z*z)
        codes[label] = torch.tensor([r * math.cos(i * golden),
                                    r * math.sin(i * golden), z], dtype=dtype)
    w = {"embed": zeros(len(vocab) + 2, D), "pos": zeros(176, D)}
    w["embed"][:, 12] = ANCHOR
    negatives = {"not", "except", "least", "incorrect", "incorrectly", "false", "avoid", "never"}
    # Contractions and any other lexical aliases inherit every feature of their
    # source, including exclusion flags, not just its science-association codes.
    negative_aliases = {word for word, source in read_word_aliases().items()
                        if source in negatives}
    negatives = negatives | negative_aliases
    for i, word in enumerate(vocab, 2):
        qs, ks = entries[word]
        for j, tag in enumerate(qs):
            w["embed"][i, 3*j:3*j+3] = codes[tag]
        for j, tag in enumerate(ks):
            w["embed"][i, 6+3*j:9+3*j] = codes[tag]
        unquoted = word[1:-1] if word.startswith("'") and word.endswith("'") else word
        if unquoted in negatives:
            w["embed"][i, 15] = ROLE
    w["pos"][128:, 13] = ROLE
    for layer in range(2):
        p = f"block{layer}."
        for ln in ("ln1", "ln2"):
            w[p + ln + ".weight"] = torch.full((D,), SIGMA, dtype=dtype)
            w[p + ln + ".bias"] = zeros(D)
        for name in ("q", "k", "v", "o"):
            w[p + name] = zeros(D, D)
            w[p + name + ".bias"] = zeros(D)
        w[p + "fc1"] = zeros(D, 32)
        w[p + "fc1.bias"] = zeros(32)
        w[p + "fc2"] = zeros(32, D)
        w[p + "fc2.bias"] = zeros(D)

    def contrast(j):
        v = zeros(D)
        v[j], v[14] = 1.0, -1.0
        return v

    # First block: broadcast the question's exclusion flag.  Answer tokens are
    # excluded from this attention head using their positional role.
    p = "block0."
    w[p + "q.bias"][0] = 2.0
    w[p + "k"][:, 0] = 32.0 / ROLE * contrast(15) - 100.0 / ROLE * contrast(13)
    # No non-binary value scaling: exact zero markers remain exactly zero.
    w[p + "v"][:, 0] = contrast(15)
    w[p + "o"][0, 15] = 1.0
    # A second, uniform head measures f = real option tokens / all real tokens.
    # Store f below the exclusion bit's 100-unit spacing in the same register.
    # This head does not inspect science words or any question labels.
    w[p + "v"][:, 4] = contrast(13) / VALUE_DIVISOR
    w[p + "o"][4, 15] = VALUE_DIVISOR / ROLE

    # Zero the answer-key slots at question positions. GELU pairs implement
    # signed linear gates. The large fixed anchor makes normalization almost
    # exactly an identity on coordinate differences, without depending on text.
    hidden = 0
    for j in range(6, 12):
        for sign in (1.0, -1.0):
            w[p + "fc1"][:, hidden] = 20.0 * (sign * contrast(j) - 0.04 * contrast(13))
            w[p + "fc1.bias"][hidden] = 40.0
            w[p + "fc2"][hidden, j] = -sign / 40.0
            hidden += 1
    # Fold local-plus-broadcast 200+f back to 100+f. A step, rather than
    # the previous triangular fold, preserves the fractional length register.
    fold_scale = 32.0
    for threshold, coefficient in ((150.0, -ROLE), (151.0, ROLE)):
        w[p + "fc1"][:, hidden] = fold_scale * contrast(15)
        w[p + "fc1.bias"][hidden] = -fold_scale * threshold
        w[p + "fc2"][hidden, 15] = coefficient / fold_scale
        hidden += 1

    # Second block: all four combinations of two query tags and two answer
    # tags. Non-answer keys are null competitors. Only a close semantic match
    # overcomes the fixed answer-key penalty; values identify answer positions.
    p = "block1."
    for head, (query_slot, key_slot) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        offset = 4 * head
        for j in range(3):
            w[p + "q"][:, offset+j] = contrast(3*query_slot+j)
            w[p + "k"][:, offset+j] = 2.0 * TEMP * contrast(6+3*key_slot+j)
        w[p + "q.bias"][offset+3] = 2.0
        # With a real question present, option queries must retrieve only null
        # question values, never score themselves. For question queries this
        # added contrast is exactly zero, preserving their retrieval logits.
        w[p + "q"][:, offset+3] = ROLE_QUERY_BOOST / ROLE * contrast(13)
        w[p + "k"][:, offset+3] = -(TEMP - MATCH) / ROLE * contrast(13)
        w[p + "v"][:, offset] = contrast(13) / VALUE_DIVISOR
        w[p + "o"][offset, 14] = VALUE * VALUE_DIVISOR / ROLE

    # Conditional arithmetic: keep evidence at question positions, reverse it
    # for exclusion questions, and suppress option self-matches. Paired ReLUs
    # cancel the gate's offset. GELU equals ReLU to machine precision here.
    reference = zeros(D)
    reference[:12] = 1.0 / 12.0
    def relative(j):
        v = -reference.clone()
        v[j] += 1.0
        return v
    # Question role is exactly zero. Using it as the evidence reference avoids
    # the old semantic-code-dependent baseline S - mean(semantic coordinates).
    # The semantic mean is used only to separate the widely spaced role gates,
    # where its sub-unit variation cannot change the gate's state.
    s = zeros(D)
    s[14], s[13] = 1.0, -1.0
    n = zeros(D)
    n[15], n[13] = 1.0, -1.0
    r = relative(13)
    # n is relative to role as well, so twice the role coefficient is needed
    # to suppress option positions even when their exclusion marker is zero.
    base_plus = -64.0*n - 128.0*r
    base_minus = 64.0*n - 128.0*r
    neurons = [
        (s + base_plus, 2048.0, GAIN),
        (base_plus, 2048.0, -GAIN),
        (s + base_minus, 2048.0 - 6400.0, -GAIN),
        (base_minus, 2048.0 - 6400.0, GAIN),
        (s, 0.0, -1.0),
        (-s, 0.0, 1.0),
    ]
    for j, (direction, bias, coefficient) in enumerate(neurons):
        w[p + "fc1"][:, j] = direction
        w[p + "fc1.bias"][j] = bias
        w[p + "fc2"][j, 14] = coefficient
    # Decode the fractional register from n = -100+f, f, or 100+f.
    # The three clipped ramps and two unit-width reset steps are a single
    # piecewise-linear function, implemented in ten otherwise-unused neurons.
    # Cap only degenerate empty-question cases: valid inputs have f <= 48/49,
    # strictly below 255/256. This keeps the final normalization anchor nonzero.
    cap = 255.0 / 256.0
    ramp_scale = 4096.0
    length_neurons = []
    for origin in (-ROLE, 0.0, ROLE):
        length_neurons.extend(((origin, 1.0), (origin + cap, -1.0)))
    for reset in (-ROLE / 2.0, ROLE / 2.0):
        length_neurons.extend(((reset, -cap), (reset + 1.0, cap)))
    for j, (threshold, coefficient) in enumerate(length_neurons, len(neurons)):
        w[p + "fc1"][:, j] = ramp_scale * n
        w[p + "fc1.bias"][j] = -ramp_scale * threshold
        w[p + "fc2"][j, 12] = -ANCHOR * coefficient / ramp_scale
    # The last MLP changes only the anchor to A*(1-f). The final layer norm
    # therefore multiplies question evidence by 1/(1-f) = (nq+no)/nq,
    # cancelling the prescribed mean pool's option-length dilution.
    w["ln_f.weight"] = torch.full((D,), SIGMA, dtype=dtype)
    w["ln_f.bias"] = zeros(D)
    w["score"] = s / (GAIN * VALUE)
    return w
