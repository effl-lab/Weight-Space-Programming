"""Text handling shared by build.py and the vocabulary script: tokenizer, light stemmer, stopwords, knowledge reader.

Everything here is hand-written; nothing is learned.
"""
import re
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9']+")

# Function words and question boilerplate. In a question they act as the attention sink (a fixed logit that
# association matches must beat); in an option they score nothing.
STOPWORDS = set("""
a an the of to in on at by for from with without into onto upon over under about above below between among
through during before after since until than then that this these those there here where when what which who whom
whose why how is are was were be been being am do does did done doing have has had having will would shall
should can could may might must it its it's itself they them their theirs themselves he him his she her hers we
us our ours you your yours i me my mine and or but nor so if because as while also too very just only not no
yes any some each every all both either neither other another such same own more most less least many much few
several one ones s t what's which's that's there's
following best better describes describe described explains explain explained statement statements example
examples likely probably true false correct incorrect answer question questions shown show shows below above
student students teacher teachers class classroom scientist scientists researcher researchers person people
someone something thing things way ways kind kinds type types part parts called known used use uses using
mr mrs ms named name names group groups list listed choose chosen select item items option options
function functions purpose purposes job jobs main mainly
would will always never often usually sometimes possible possibly certain certainly
""".split())

# Words that flip what the question asks for.
NEGATIONS = {"not", "except", "least", "unlikely", "never", "cannot", "can't", "isn't", "doesn't", "wouldn't"}

IRREGULAR = {
    "leaves": "leaf", "leaf": "leaf", "teeth": "tooth", "feet": "foot", "mice": "mouse", "geese": "goose",
    "children": "child", "men": "man", "women": "woman", "people": "person", "lives": "life", "wolves": "wolf",
    "knives": "knife", "halves": "half", "shelves": "shelf", "calves": "calf", "loaves": "loaf",
    "fungi": "fungus", "bacteria": "bacterium", "nuclei": "nucleus", "cacti": "cactus", "larvae": "larva",
    "pupae": "pupa", "algae": "alga", "antennae": "antenna", "data": "datum", "media": "medium",
    "criteria": "criterion", "phenomena": "phenomenon", "vertebrae": "vertebra", "stimuli": "stimulus",
    "species": "species", "series": "series", "gas": "gas", "gases": "gas", "glasses": "glass", "glass": "glass",
    "grass": "grass", "grasses": "grass", "mass": "mass", "masses": "mass", "class": "class", "moss": "moss",
    "mosses": "moss", "is": "is", "was": "was", "has": "has", "does": "does", "this": "this", "his": "his",
    "its": "its", "us": "us", "lens": "lens", "lenses": "lens", "axis": "axis", "axes": "axis", "news": "news",
    "physics": "physics", "mathematics": "mathematics", "genetics": "genetics", "electronics": "electronics",
    "analysis": "analysis", "hypothesis": "hypothesis", "hypotheses": "hypothesis", "photosynthesis": "photosynthesis",
    "mitosis": "mitosis", "meiosis": "meiosis", "osmosis": "osmosis", "metamorphosis": "metamorphosis",
    "virus": "virus", "viruses": "virus", "nucleus": "nucleus", "fungus": "fungus", "cactus": "cactus",
    "uranus": "uranus", "venus": "venus", "status": "status", "surplus": "surplus", "humus": "humus",
    "octopus": "octopus", "platypus": "platypus", "hippopotamus": "hippopotamus", "fetus": "fetus",
    "stimulus": "stimulus", "radius": "radius", "celsius": "celsius", "fahrenheit": "fahrenheit",
    "focus": "focus", "bus": "bus", "plus": "plus", "thus": "thus", "minus": "minus", "chorus": "chorus",
    "cirrus": "cirrus", "cumulus": "cumulus", "stratus": "stratus", "nimbus": "nimbus", "cumulonimbus": "cumulonimbus",
    "asbestos": "asbestos", "pancreas": "pancreas", "atlas": "atlas", "canvas": "canvas", "bias": "bias",
    "iris": "iris", "pelvis": "pelvis", "epidermis": "epidermis", "dermis": "dermis", "trachea": "trachea",
    "chrysalis": "chrysalis", "chlorosis": "chlorosis", "tuberculosis": "tuberculosis", "thesis": "thesis",
    "volcanoes": "volcano", "potatoes": "potato", "tomatoes": "tomato", "echoes": "echo", "heroes": "hero",
    "mosquitoes": "mosquito", "tornadoes": "tornado", "dried": "dry", "drying": "dry", "dries": "dry",
    "dying": "die", "died": "die", "dies": "die", "lying": "lie", "tying": "tie",
    "went": "go", "gone": "go", "goes": "go", "ran": "run", "running": "run", "grew": "grow", "grown": "grow",
    "grows": "grow", "growing": "grow", "growth": "grow", "froze": "freeze", "frozen": "freeze", "freezes": "freeze",
    "freezing": "freeze", "melted": "melt", "melting": "melt", "melts": "melt", "rose": "rise", "risen": "rise",
    "rises": "rise", "rising": "rise", "fell": "fall", "fallen": "fall", "falls": "fall", "falling": "fall",
    "ate": "eat", "eaten": "eat", "eats": "eat", "eating": "eat", "flew": "fly", "flown": "fly", "flies": "fly",
    "flying": "fly", "blew": "blow", "blown": "blow", "threw": "throw", "thrown": "throw", "drew": "draw",
    "drawn": "draw", "bit": "bite", "bitten": "bite", "hid": "hide", "hidden": "hide", "hiding": "hide",
    "laid": "lay", "lain": "lie", "made": "make", "making": "make", "makes": "make", "took": "take",
    "taken": "take", "taking": "take", "gave": "give", "given": "give", "broke": "break", "broken": "break",
    "spun": "spin", "swam": "swim", "swum": "swim", "swimming": "swim", "dug": "dig", "digging": "dig",
    "fed": "feed", "feeding": "feed", "bred": "breed", "breeding": "breed", "bled": "bleed", "held": "hold",
    "kept": "keep", "slept": "sleep", "felt": "feel", "left": "left", "lost": "lose", "built": "build",
    "sent": "send", "spent": "spend", "bent": "bend", "caught": "catch", "taught": "teach", "thought": "think",
    "brought": "bring", "bought": "buy", "fought": "fight", "sought": "seek", "shook": "shake", "shaken": "shake",
    "wore": "wear", "worn": "wear", "tore": "tear", "torn": "tear", "struck": "strike", "stuck": "stick",
    "sank": "sink", "sunk": "sink", "shrank": "shrink", "shrunk": "shrink", "drank": "drink", "drunk": "drink",
    "sang": "sing", "sung": "sing", "rang": "ring", "rung": "ring", "began": "begin", "begun": "begin",
    "saw": "see", "seen": "see", "sees": "see", "knew": "know", "known": "know", "wrote": "write",
    "written": "write", "drove": "drive", "driven": "drive", "rode": "ride", "ridden": "ride", "chose": "choose",
    "spoke": "speak", "woke": "wake", "led": "lead", "lit": "light", "shone": "shine", "shot": "shoot",
    "slid": "slide", "dove": "dive", "stung": "sting", "swung": "swing", "clung": "cling", "hung": "hang",
    "heated": "heat", "heating": "heat", "heats": "heat", "boiled": "boil", "boiling": "boil", "boils": "boil",
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    "larger": "large", "largest": "large", "smaller": "small", "smallest": "small", "bigger": "big",
    "biggest": "big", "hotter": "hot", "hottest": "hot", "colder": "cold", "coldest": "cold", "warmer": "warm",
    "warmest": "warm", "cooler": "cool", "coolest": "cool", "faster": "fast", "fastest": "fast",
    "slower": "slow", "slowest": "slow", "higher": "high", "highest": "high", "lower": "low", "lowest": "low",
    "longer": "long", "longest": "long", "shorter": "short", "shortest": "short", "heavier": "heavy",
    "heaviest": "heavy", "lighter": "light", "lightest": "light", "harder": "hard", "hardest": "hard",
    "softer": "soft", "softest": "soft", "denser": "dense", "densest": "dense", "greater": "great",
    "greatest": "great", "stronger": "strong", "strongest": "strong", "weaker": "weak", "weakest": "weak",
    "closer": "close", "closest": "close", "farther": "far", "farthest": "far", "further": "far",
    "deeper": "deep", "deepest": "deep", "thicker": "thick", "thinner": "thin", "wetter": "wet",
    "drier": "dry", "driest": "dry", "brighter": "bright", "brightest": "bright", "darker": "dark",
    "louder": "loud", "loudest": "loud", "quieter": "quiet", "older": "old", "oldest": "old",
    "younger": "young", "youngest": "young", "newer": "new", "newest": "new", "earlier": "early",
    "later": "late", "latest": "late", "wider": "wide", "widest": "wide", "steeper": "steep",
    "rougher": "rough", "smoother": "smooth", "smoothest": "smooth", "roughest": "rough",
    "thickest": "thick", "thinnest": "thin", "taller": "tall", "tallest": "tall", "simpler": "simple",
    "easier": "easy", "easiest": "easy", "safer": "safe", "safest": "safe",
    "nearer": "near", "nearest": "near", "darkest": "dark", "warmth": "warm", "heights": "height",
    "dense": "dense", "density": "dense", "densities": "dense",
}

_NO_STEM_END = ("ss", "us", "is", "ous", "as")
_VOWEL = re.compile(r"[aeiouy]")


def stem(w: str) -> str:
    """A small suffix stripper. It only has to map the forms of one word together; it never has to be pretty."""
    if w in IRREGULAR:
        return IRREGULAR[w]
    if w.endswith("'s"):
        w = w[:-2]
    w = w.replace("'", "")
    if len(w) <= 3 or w.isdigit():
        return w
    if w in IRREGULAR:
        return IRREGULAR[w]
    for suf, rep in (("ies", "y"), ("ied", "y")):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[: -len(suf)] + rep
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith(("ches", "shes", "xes")) and len(w) > 4:
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(_NO_STEM_END):
        w = w[:-1]
    if w in IRREGULAR:
        return IRREGULAR[w]
    for suf in ("ing", "ed"):
        if w.endswith(suf) and not w.endswith("eed"):
            base = w[: -len(suf)]
            if len(base) >= 3 and _VOWEL.search(base):
                if base[-1] == base[-2] and base[-1] not in "lsz":
                    base = base[:-1]                       # stopped -> stop, but not spilled -> spil
                w = base
                break
    if len(w) > 4 and w.endswith("e") and not w.endswith(("ee", "ie")):
        w = w[:-1]                                         # change / changed / changing -> chang
    return w


def words(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def knowledge_lines(folder: Path, with_source: bool = False) -> list:
    """Every cluster line of knowledge/*.txt as (weight, raw words). '#' starts a comment.

    A line may start with '*N ' to give the cluster weight N (default 1)."""
    out = []
    for f in sorted((folder / "knowledge").glob("*.txt")):
        n = 0
        for line in f.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            weight = 1.0
            if line.startswith("*"):
                head, _, line = line.partition(" ")
                weight = float(head[1:])
            ws = words(line)
            if ws:
                out.append((weight, ws, f.name, n) if with_source else (weight, ws))
                n += 1
    return out


DERIV_SUFFIXES = ("'s", "s'", "ally", "ly", "ness", "ment", "ments", "ful", "ity", "ities", "ation", "ations",
                  "ion", "ions", "ive", "al", "er", "ers", "est", "ist", "ists", "ic", "ical", "ism", "able")


def stem_candidates(w: str) -> list[str]:
    """Stems to try for a word, best first: its own stem, then the stems left after removing a derivational
    suffix (rapidly -> rapid, completely -> complete, earth's -> earth). Prefixes are never removed, since
    they can reverse the meaning (nonrenewable, unbalanced)."""
    out = [stem(w)]
    for suf in DERIV_SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            for b in (base, base + "e", base[:-1] if len(base) > 3 and base[-1] == base[-2] else base,
                      base[:-1] + "y" if base.endswith("i") else base):
                s = stem(b)
                if s not in out:
                    out.append(s)
    return out
