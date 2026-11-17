"""An analytic, untrained transformer: lexical associations compiled to weights.

Only the four human-authored TSV files and vocab.txt are read by build().
No datasets, optimization, pretrained weights, randomness, or network access.
"""
from pathlib import Path
import math
import re

import torch

ROOT = Path(__file__).resolve().parent
D = 12
ANCHOR = 8192.0
LN_SCALE = ANCHOR / math.sqrt(6.0)
MAX_QUESTION_KEYS = 128
EXACT_MATCH_LEAKAGE = 1e-6

# Function words deliberately carry no scientific evidence. Unknown words use
# the same neutral entry. These words still participate in ordinary attention.
STOP = """a an the this that these those which what who whom whose where when why
how is are was were be been being am do does did doing have has had having can
could may might will would shall should to of in on at by for from with as and
or but if then than so such it its it's they their them they're theirs he his
him she her hers we our us you your yours i me my mine itself themselves himself
herself ourselves yourself yourselves there here about into onto upon out up
during before after while through between among against under over around near
across along within toward towards until since because whether also very much
many several some any another other others certain following below above shown
shows show showing shown according example examples statement statements best
better answer answers choice choices choose student students scientist scientists
researcher researchers researcher teacher teachers classroom class school
determine determines determining determined describe describes description
most likely typically usually generally often sometimes once twice next first
second third fourth last one two three four five six seven eight nine ten""".split()

# Small explicitly designed language priors, not frequencies or fitted scores.
PRIORS = {
    "all": 0.010, "both": 0.008, "variety": 0.008,
    "only": -0.020, "solely": -0.020, "exclusively": -0.020,
    "always": -0.030, "never": -0.030, "nothing": -0.020,
    "instantly": -0.030, "perfectly": -0.030, "completely": -0.010,
}
NEGATION = {"not", "except", "least", "incorrect", "false", "cannot",
            "can't", "unlikely", "avoid", "avoids", "avoided", "avoiding"}

# Only these authored noun lists receive grammatical possessive aliases.
# No blanket suffix guessing is applied to verbs, adjectives, or unknown words.
SINGULAR_POSSESSIVES = """animal organism plant leaf root stem flower seed fruit
tree cell chromosome gene atom molecule electron proton neutron nucleus particle
rock mineral crystal fossil glacier river ocean lake pond star planet moon sun
earth cloud wave magnet circuit battery wire bulb lens mirror thermometer
barometer microscope telescope scientist student experiment hypothesis sample
population community ecosystem habitat species predator prey consumer producer
decomposer mammal bird fish insect reptile amphibian rabbit mouse frog butterfly
heart lung brain kidney liver skin bone muscle tissue organ membrane chloroplast
mitochondrion ribosome vacuole skeleton water air wind light sound heat energy
force gravity mass volume speed velocity momentum pressure temperature density
frequency amplitude wavelength oxygen nitrogen carbon hydrogen helium sulfur
chlorophyll glucose protein dna food soil sand clay metal wood glass plastic
color behavior environment atmosphere climate weather""".split()
PLURAL_POSSESSIVES = """animals organisms plants leaves roots stems flowers seeds
fruits trees cells chromosomes genes atoms molecules electrons protons neutrons
nuclei particles rocks minerals crystals fossils glaciers rivers oceans lakes
ponds stars planets moons clouds waves magnets circuits batteries wires bulbs
lenses mirrors thermometers barometers microscopes telescopes scientists students
experiments hypotheses samples populations communities ecosystems habitats
species predators consumers producers decomposers mammals birds insects reptiles
amphibians rabbits mice frogs butterflies lungs kidneys bones muscles tissues
organs membranes chloroplasts mitochondria ribosomes vacuoles proteins""".split()


def _rows(name, columns):
    for line in (ROOT / name).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) != columns:
            raise ValueError(f"Malformed {name} row: {line!r}")
        yield cells


def lexicon():
    """Compile authored facts; tuple = entity, cue target, topic, salience."""
    entities = ["__neutral__"]
    topics = ["__neutral__"]
    entity_topic = {"__neutral__": "__neutral__"}
    entries = {}
    for entity, topic, text in _rows("knowledge.tsv", 3):
        if entity not in entity_topic:
            entities.append(entity)
            entity_topic[entity] = topic
        if topic not in topics:
            topics.append(topic)
        for word in text.split():
            # First sense wins for ambiguous words; explicit exceptions below.
            if word not in entries:
                salience = float(topic != "general")
                entries[word] = [entity, entity, topic, salience]

    # Definitions for the few association targets without a separate TSV row.
    for entity, topic in [("anemometer", "weather"), ("mutualism", "ecology"),
                          ("classification", "experiment")]:
        if entity not in entity_topic:
            entities.append(entity)
            entity_topic[entity] = topic
        entries[entity] = [entity, entity, topic, 1.0]

    # Deliberate disambiguations of words shared by several definitions.
    senses = {
        "weather": "weather", "mineral": "mineral", "minerals": "mineral",
        "plasma": "plasma", "nuclear": "nuclearenergy",
        "solar": "sun", "mercury": "mercuryplanet", "current": "electricity",
        "currents": "electricity", "volume": "volume", "reaction": "chemical",
        "reactions": "chemical", "radiation": "radiation",
        "fission": "nuclearenergy", "fusion": "nuclearenergy",
        "erosion": "erosion", "conservation": "conservation",
        "reflect": "reflection", "reflected": "reflection",
        "reflections": "reflection", "permeability": "membrane",
    }
    for word, entity in senses.items():
        entries[word] = [entity, entity, entity_topic[entity], 1.0]
    # The Sun is also the input to photosynthesis and renewable solar power.
    for word in ("sun", "sun's", "sunshine", "solar"):
        if word in entries:
            entries[word][2] = "photosynthesis"

    # Numbers are exact lexical entities, not an arithmetic engine.
    number_words = {
        0: "zero", 1: "one single", 2: "two double pair", 3: "three triple",
        4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
        10: "ten", 11: "eleven", 12: "twelve dozen", 13: "thirteen",
        14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
        18: "eighteen", 19: "nineteen", 20: "twenty", 30: "thirty",
        40: "forty", 50: "fifty", 60: "sixty", 70: "seventy",
        80: "eighty", 90: "ninety", 100: "hundred", 1000: "thousand",
        1000000: "million",
    }
    for number in list(range(101)) + [180, 212, 273, 360, 365, 366, 1000, 1000000]:
        entity = f"number_{number}"
        entities.append(entity)
        entity_topic[entity] = "measurement"
        for word in [str(number)] + number_words.get(number, "").split():
            entries[word] = [entity, entity, "measurement", 1.0]

    for target, text in _rows("associations.tsv", 2):
        if target not in entity_topic:
            raise ValueError(f"Unknown association target: {target}")
        for word in text.split():
            if word not in entries:
                # A cue-only word gets its own identity rather than pretending
                # it is a synonym of the answer it suggests.
                entity = "cue_" + word
                entities.append(entity)
                entity_topic[entity] = entity_topic[target]
                entries[word] = [entity, target, entity_topic[target], 1.0]
            else:
                entries[word][1] = target

    # Explicit lexical distinctions override the earlier coarse synonym groups.
    # Register all new entities before resolving targets, allowing forward refs.
    refinements = list(_rows("refinements.tsv", 4))
    for entity, target, topic, text in refinements:
        if entity not in entity_topic:
            entities.append(entity)
            entity_topic[entity] = topic
        if topic not in topics:
            topics.append(topic)
    for entity, target, topic, text in refinements:
        if target not in entity_topic:
            raise ValueError(f"Unknown refinement target: {target}")
        for word in text.split():
            entries[word] = [entity, target, topic, 1.0]

    for word in STOP:
        # Keep useful numerical and physical meanings of common words.
        if word not in entries:
            entries[word] = ["__neutral__", "__neutral__", "__neutral__", 0.0]
    # Generic prompt boilerplate should not be evidence for an option.
    for word in ("which", "what", "these", "those", "best", "most", "likely",
                 "following", "example", "examples", "statement", "statements",
                 "cause", "causes", "caused", "causing", "effect", "effects",
                 "affect", "affects", "affected", "affecting", "due", "since",
                 "object", "objects", "thing", "things", "system", "systems"):
        if word in entries:
            entries[word][3] = 0.0
    for word in sorted(NEGATION | set(PRIORS)):
        if word not in entries:
            entity = "grammar_" + word
            entities.append(entity)
            entity_topic[entity] = "general"
            entries[word] = [entity, entity, "general", 1.0]
        # Such terms need a scoring gate even if their topic is general.
        entries[word][3] = 1.0
    # Aliases copy all lexical features; the embedding compiler also copies
    # their canonical negation flag and prior. Existing explicit spellings can
    # be unified, but aliases never introduce new ontology/codebook entries.
    aliases = {}
    alias_rows = [(canonical, spelling) for canonical, text in _rows("aliases.tsv", 2)
                  for spelling in text.split()]
    alias_rows += [(noun, noun + "'s") for noun in SINGULAR_POSSESSIVES]
    alias_rows += [(noun, noun + ("'" if noun.endswith("s") else "'s"))
                   for noun in PLURAL_POSSESSIVES]
    for canonical, spelling in alias_rows:
        if canonical not in entries:
            raise ValueError(f"Unknown alias canonical form: {canonical}")
        if spelling in aliases and aliases[spelling] != canonical:
            raise ValueError(f"Conflicting alias: {spelling}")
        aliases[spelling] = canonical
    if any(canonical in aliases for canonical in aliases.values()):
        raise ValueError("Alias chains are not allowed; specify the root word")
    for spelling, canonical in sorted(aliases.items()):
        entries[spelling] = entries[canonical].copy()

    invalid = [word for word in entries if not re.fullmatch(r"[a-z0-9']+", word)]
    if invalid:
        raise ValueError(f"Vocabulary entries unreachable by the tokenizer: {invalid}")
    if len(entities) != len(set(entities)) or len(topics) != len(set(topics)):
        raise ValueError("Codebook names must be unique")
    return entries, entities, topics, aliases


def vocabulary():
    return sorted(lexicon()[0])


def _feature(j):
    """Read coordinate j while undoing LayerNorm's subtraction of the mean.

    The two fixed anchors sum to zero, so subtracting their average removes
    the common centering offset. Large anchors make variance almost constant.
    """
    vector = torch.zeros(D, dtype=torch.float64)
    vector[j] = 1.0
    vector[10] -= 0.5
    vector[11] -= 0.5
    return vector


def _circle(names):
    # Ordered codebook: neighboring entries are usually neighboring concepts
    # in the authored ontology. No stochastic or data-dependent embedding.
    # A neutral token has no semantic direction: putting it on the circle
    # would make frequent unknown/function words spuriously match neighbors.
    semantic_names = [name for name in names if name != "__neutral__"]
    codes = {name: (math.cos(2 * math.pi * i / len(semantic_names)),
                    math.sin(2 * math.pi * i / len(semantic_names)))
             for i, name in enumerate(semantic_names)}
    if "__neutral__" in names:
        codes["__neutral__"] = (0.0, 0.0)
    return codes


def _sharp_temperature(names):
    """Closed-form scale for exact-code heads, not a fitted hyperparameter.

    Distinct unit-circle codes have minimum self-versus-other dot gap
    d = 1-cos(2*pi/n). The neutral zero key has gap 1. If T is the inverse
    temperature, at most Q wrong question keys contribute Q*exp(-T*d)
    relative to an exact option self-key. Ignore the additional option-key
    bonus for a conservative bound, and allow half the gap as a generous
    margin for the approximately affine normalization and segment gates.
    """
    n = sum(name != "__neutral__" for name in names)
    if n < 1:
        raise ValueError("An exact-match codebook needs at least one concept")
    gap = min(1.0, 2.0 * math.sin(math.pi / n) ** 2) if n > 1 else 1.0
    return 2.0 * math.log(MAX_QUESTION_KEYS / EXACT_MATCH_LEAKAGE) / gap


def build():
    entries, entities, topics, aliases = lexicon()
    vocab = (ROOT / "vocab.txt").read_text().splitlines()
    if vocab != sorted(entries):
        raise ValueError("vocab.txt does not match the authored lexicon")
    codes = _circle(entities)
    topic_codes = _circle(topics)
    w = {"embed": torch.zeros(len(vocab) + 2, D),
         "pos": torch.zeros(176, D),
         "ln_f.weight": torch.full((D,), LN_SCALE, dtype=torch.float64),
         "ln_f.bias": torch.zeros(D),
         "score": 100.0 * (torch.eye(D)[9] - torch.eye(D)[7])}
    for i in range(2):
        p = f"block{i}."
        for layer in ("ln1", "ln2"):
            w[p + layer + ".weight"] = torch.full((D,), LN_SCALE, dtype=torch.float64)
            w[p + layer + ".bias"] = torch.zeros(D)
        for name in ("q", "k", "v", "o"):
            w[p + name] = torch.zeros(D, D)
            w[p + name + ".bias"] = torch.zeros(D)
        w[p + "fc1"] = torch.zeros(D, 24)
        w[p + "fc1.bias"] = torch.zeros(24)
        w[p + "fc2"] = torch.zeros(24, D)
        w[p + "fc2.bias"] = torch.zeros(D)

    # Double precision is essential: recovery subtracts large normalization
    # anchors, and sharp code matching amplifies float32 cancellation error.
    w = {name: value.to(torch.float64) for name, value in w.items()}
    neutral = ["__neutral__", "__neutral__", "__neutral__", 0.0]
    for i, word in enumerate([None, None] + vocab):
        entity, target, topic, salience = entries.get(word, neutral)
        w["embed"][i, 0:2] = torch.tensor(codes[target], dtype=torch.float64)
        w["embed"][i, 2:4] = torch.tensor(codes[entity], dtype=torch.float64)
        w["embed"][i, 4:6] = torch.tensor(topic_codes[topic], dtype=torch.float64)
        w["embed"][i, 7] = salience
        canonical = aliases.get(word, word)
        w["embed"][i, 8] = float(canonical in NEGATION)
        w["embed"][i, 9] = PRIORS.get(canonical, 0.0)
        w["embed"][i, 10] = ANCHOR
        w["embed"][i, 11] = -ANCHOR
    w["pos"][:128, 6] = -1.0
    w["pos"][128:, 6] = 1.0

    # Block 0, head 3: broadcast explicit question negation. Query is constant;
    # the segment offset excludes option words from the negation detector.
    # Use score as temporary storage, so the MLP can erase local negation
    # markers instead of unintentionally retaining them through the residual.
    p = "block0."
    w[p + "q.bias"][9] = 1.0
    w[p + "k"][:, 9] = math.sqrt(3) * (20 * _feature(8) - 30 * _feature(6))
    w[p + "v"][:, 9] = _feature(8)
    w[p + "o"][9, 9] = 1.0

    # Block 0 MLP: for option tokens replace cue-key with their own answer
    # entity. This creates an exact self-match baseline in block 1's softmax.
    # Question keys retain their directed cue targets. GELU is linear well
    # above zero, and zero well below, implementing a bounded segment switch.
    gate = 8.0 * _feature(6)
    scale = 16.0
    for j in (0, 1):
        a, b = 2 * j, 2 * j + 1
        delta = _feature(j + 2) - _feature(j)
        w[p + "fc1"][:, a] = scale * (gate + delta)
        w[p + "fc1"][:, b] = scale * gate
        w[p + "fc1.bias"][[a, b]] = -4 * scale
        w[p + "fc2"][a, j] = 1 / scale
        w[p + "fc2"][b, j] = -1 / scale

    # Remove each token's original marker via GELU(x)-GELU(-x)=x.
    w[p + "fc1"][:, 4] = _feature(8)
    w[p + "fc1"][:, 5] = -_feature(8)
    w[p + "fc2"][4, 8] = -1.0
    w[p + "fc2"][5, 8] = 1.0
    # Negation + prior lies either near 0 or near 1. A saturating ramp with
    # knees at 1/4 and 3/4 recovers the question flag independently of priors.
    # Move that flag to coordinate 8, leaving the lexical prior in coordinate 9.
    ramp_scale = 64.0
    for slot, knee, coefficient in [(6, 0.25, 2.0), (7, 0.75, -2.0)]:
        w[p + "fc1"][:, slot] = ramp_scale * _feature(9)
        w[p + "fc1.bias"][slot] = -ramp_scale * knee
        w[p + "fc2"][slot, 8] = coefficient / ramp_scale
        w[p + "fc2"][slot, 9] = -coefficient / ramp_scale

    # Block 1: question-key versus option-self competition. For a scientifically
    # meaningful option token, same-code question words receive appreciable
    # mass; unrelated keys lose to its exact option-side self key.
    p = "block1."
    entity_sharpness = _sharp_temperature(entities)
    topic_sharpness = _sharp_temperature(topics)
    heads = [
        # query pair, key pair, inverse temperature, contribution
        (2, 0, entity_sharpness, 0.50),  # directed definition / association
        (2, 2, entity_sharpness, 0.28),  # entity and synonym overlap
        (4, 4, topic_sharpness, 0.17),   # exact shared science topic
        (2, 0, 700.0, 0.05),             # intentionally soft ontology neighbors
    ]
    for h, (query, key, temperature, contribution) in enumerate(heads):
        base = h * 3
        magnitude = math.sqrt(temperature * math.sqrt(3))
        for j in (0, 1):
            w[p + "q"][:, base + j] = magnitude * _feature(query + j)
            w[p + "k"][:, base + j] = magnitude * _feature(key + j)
        # An option-key bonus of 0.8 log units gives a conservative baseline.
        w[p + "q.bias"][base + 2] = 1.0
        w[p + "k"][:, base + 2] = 0.4 * math.sqrt(3) * _feature(6)
        w[p + "v"][:, base] = -0.5 * _feature(6)
        w[p + "v.bias"][base] = 0.5
        w[p + "o"][base, 9] = contribution

    # Block 1 MLP: retain scores only on salient option tokens; reverse them
    # when the question explicitly asks for an exception or false alternative.
    # The identity GELU(x)-GELU(-x)=x cancels the unfiltered residual score.
    y = _feature(9)
    scale = 32.0
    w[p + "fc1"][:, 0] = y
    w[p + "fc1"][:, 1] = -y
    w[p + "fc2"][0, 9] = -1.0
    w[p + "fc2"][1, 9] = 1.0
    for slot, negated, coefficient in [(2, False, 1.0), (4, True, -2.0)]:
        gate = 8 * (_feature(6) + _feature(7))
        bias = -12.0
        if negated:
            gate = gate + 8 * _feature(8)
            bias = -20.0
        w[p + "fc1"][:, slot] = scale * (gate + y)
        w[p + "fc1"][:, slot + 1] = scale * gate
        w[p + "fc1.bias"][[slot, slot + 1]] = scale * bias
        w[p + "fc2"][slot, 9] = coefficient / scale
        w[p + "fc2"][slot + 1, 9] = -coefficient / scale
    # Salience is no longer needed. Clear it, then read score-minus-salience
    # at the final layer. This cancels LayerNorm's mean without subtracting
    # the large anchors in the final dot product (avoiding roundoff loss).
    w[p + "fc1"][:, 6] = _feature(7)
    w[p + "fc1"][:, 7] = -_feature(7)
    w[p + "fc2"][6, 7] = -1.0
    w[p + "fc2"][7, 7] = 1.0
    return {name: tensor.contiguous() for name, tensor in w.items()}


if __name__ == "__main__":
    # Deterministic vocabulary generation, not a training/build dependency.
    (ROOT / "vocab.txt").write_text("\n".join(vocabulary()) + "\n")
    print(f"Wrote {len(vocabulary())} vocabulary entries.")
