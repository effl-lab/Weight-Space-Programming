"""Text handling shared by build.py and the vocabulary script: stop words, a small stemmer, negation words."""
import re

TOKEN = re.compile(r"[a-z0-9']+")

# words that carry no science content of their own (they act as attention sinks)
STOP = set("""
a an the and or but if then than so as of to in on at by for from with without within into onto upon over under
about above below across after against along among around before behind beneath beside besides between beyond
down during except inside near off out outside past since through throughout till toward towards until up via
is are was were be been being am do does did done doing have has had having will would shall should can could may
might must ought need needs needed i me my mine we us our ours you your yours he him his she her hers it its they
them their theirs this that these those who whom whose which what when where why how whether while also too very
just only even still yet already again ever once both each either neither every all any some such no nor not
own same other another more most less least much many few several lot lots one two three four five six seven
eight nine ten first second third last next new here there now well really quite rather almost often usually
sometimes always never s t don't doesn't didn't isn't aren't wasn't weren't won't can't cannot couldn't
shouldn't wouldn't it's that's there's they're we're you're i'm let let's
following best likely describe describes described explain explains explained statement statements example
examples answer question questions student students scientist scientists teacher class person people group
show shows showed shown way ways thing things something someone type types kind kinds called call
probably possible possibly true false correct incorrect likely unlikely mostly mainly
use uses used using make makes made making happen happens happened occur occurs occurred result results
help helps helped get gets got take takes took give gives gave go goes went come comes came put puts see sees saw
seen look looks looked find finds found know knows knew think thinks thought want wants say says said tell tells
like likes able because therefore however although though thus hence ever per
process processes appear appears appeared known cause causes caused causing affect affects affected effect effects part parts
important main certain different various mostly usually generally specific particular following describe
identify identifies determine determines determined explanation reason reasons idea ideas information
situation activity activities factor factors characteristic characteristics feature features likely
would could should might must probably done does doing need needs makes become becomes became
""".split())

NEGATION = {"not": 1.0, "except": 1.0, "least": 1.0, "cannot": 1.0, "isn't": 1.0, "aren't": 1.0,
            "doesn't": 1.0, "don't": 1.0, "can't": 1.0, "won't": 1.0, "wouldn't": 1.0}

IRREGULAR = {
    "grew": "grow", "grown": "grow", "ate": "eat", "eaten": "eat", "fell": "fall", "fallen": "fall",
    "froze": "freeze", "frozen": "freeze", "children": "child", "mice": "mouse", "teeth": "tooth", "feet": "foot",
    "geese": "goose", "men": "man", "women": "woman", "leaves": "leaf", "lives": "live", "wolves": "wolf",
    "knives": "knife", "halves": "half", "shelves": "shelf", "calves": "calf", "hooves": "hoof", "fungi": "fungus",
    "nuclei": "nucleus", "larvae": "larva", "pupae": "pupa", "antennae": "antenna", "phenomena": "phenomenon",
    "criteria": "criterion", "flew": "fly", "flown": "fly", "swam": "swim", "swum": "swim", "ran": "run",
    "spun": "spin", "dug": "dig", "built": "build", "kept": "keep", "lost": "lose", "held": "hold",
    "brought": "bring", "bought": "buy", "caught": "catch", "taught": "teach", "sank": "sink", "sunk": "sink",
    "shone": "shine", "began": "begin", "begun": "begin", "broke": "break", "broken": "break", "chose": "choose",
    "chosen": "choose", "drove": "drive", "driven": "drive", "rode": "ride", "ridden": "ride", "rose": "rise",
    "risen": "rise", "struck": "strike", "blew": "blow", "blown": "blow", "drank": "drink", "drunk": "drink",
    "sang": "sing", "sung": "sing", "rang": "ring", "hung": "hang", "fed": "feed", "gave": "give", "given": "give",
    "took": "take", "taken": "take", "shook": "shake", "shaken": "shake", "wrote": "write", "written": "write",
    "hid": "hide", "hidden": "hide", "fought": "fight", "slid": "slide", "felt": "feel", "meant": "mean",
    "sent": "send", "spent": "spend", "went": "go", "gone": "go", "came": "come", "saw": "see", "seen": "see",
    "knew": "know", "known": "know", "told": "tell", "sold": "sell", "stood": "stand", "understood": "understand",
    "woke": "wake", "worn": "wear", "wore": "wear", "torn": "tear", "tore": "tear", "stolen": "steal",
    "spoke": "speak", "spoken": "speak", "born": "bear", "laid": "lay", "lain": "lie", "made": "make",
    "found": "find", "thrown": "throw", "threw": "throw", "drew": "draw", "drawn": "draw", "bit": "bite",
    "bitten": "bite", "shot": "shoot", "lit": "light", "led": "lead", "bled": "bleed", "bred": "breed",
    "dealt": "deal", "dove": "dive", "forgot": "forget", "froze": "freeze", "ground": "ground", "heard": "hear",
    "paid": "pay", "rid": "rid", "sat": "sit", "slept": "sleep", "swept": "sweep", "swung": "swing",
    "stung": "sting", "stuck": "stick", "wept": "weep", "won": "win", "wound": "wind", "wove": "weave",
    "woven": "weave", "mated": "mate", "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    "data": "data", "bacteria": "bacteria", "algae": "algae", "species": "species", "series": "series",
    "news": "news", "glasses": "glass", "gas": "gas", "gases": "gas", "lens": "lens", "lenses": "lens",
    "axis": "axis", "axes": "axis", "this": "this", "mass": "mass", "has": "has", "was": "was", "does": "does",
    "atlas": "atlas", "canvas": "canvas", "cactus": "cactus", "cacti": "cactus", "octopus": "octopus",
    "means": "mean", "focus": "focus", "status": "status", "always": "always", "physics": "physics",
    "genetics": "genetics", "mathematics": "mathematics", "electronics": "electronics", "politics": "politics",
    "people": "person", "dice": "die", "oxen": "ox", "vertebrae": "vertebra", "fish": "fish", "sheep": "sheep",
    "deer": "deer", "moose": "moose", "salmon": "salmon", "trout": "trout", "shrimp": "shrimp",
    "dying": "die", "died": "die", "dies": "die", "lying": "lie", "tying": "tie", "eyes": "eye", "dyes": "dye",
    "goes": "go", "does": "do", "echoes": "echo", "potatoes": "potato", "tomatoes": "tomato",
    "volcanoes": "volcano", "tornadoes": "tornado", "heroes": "hero", "mosquitoes": "mosquito",
    "less": "less", "loss": "loss", "moss": "moss", "grass": "grass", "dress": "dress", "stress": "stress",
}

# base adjectives whose -er / -est / -ly forms are reduced to the base
ADJ = set("""
warm cold hot cool large small fast slow high low great hard soft heavy light dense close long short strong weak
thick thin deep shallow wide narrow bright dark dry wet loud quiet big tall old young rough smooth near far safe
easy early late simple fine coarse rich poor clear sharp dull fresh pure full empty quick cheap tiny huge steep
flat bold calm wild tough firm loose tight brief rapid slight loud mild fierce harsh rare common
""".split())

VOWELS = set("aeiouy")


def words(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _has_vowel(s: str) -> bool:
    return any(ch in VOWELS for ch in s)


def _undouble(s: str) -> str:
    if len(s) > 2 and s[-1] == s[-2] and s[-1] not in "aeioulsz":
        return s[:-1]
    return s


def _adj_base(w: str, suffix: str):
    base = w[: -len(suffix)]
    for cand in (base, base + "e", _undouble(base), base[:-1] + "y" if base.endswith("i") else None):
        if cand and cand in ADJ:
            return cand
    return None


def stem(w: str) -> str:
    """A crude, consistent stemmer: inflected forms of a word share one stem."""
    w = w.strip("'")
    if w.endswith("'s"):
        w = w[:-2]
    if w in IRREGULAR:
        w = IRREGULAR[w]
    elif len(w) > 3 and not w.isdigit():
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("sses"):
            w = w[:-2]
        elif w.endswith(("xes", "ches", "shes", "zes")):
            w = w[:-2]
        elif w.endswith("s") and not w.endswith(("ss", "us", "is", "ous")):
            w = w[:-1]
        if w.endswith("ied") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("eed"):
            pass
        elif w.endswith("ed") and len(w) > 3 and _has_vowel(w[:-2]):
            w = _undouble(w[:-2])
        elif w.endswith("ing") and len(w) > 4 and _has_vowel(w[:-3]):
            w = _undouble(w[:-3])
        for suf in ("est", "er", "ly"):
            if w.endswith(suf) and len(w) > len(suf) + 2:
                b = _adj_base(w, suf)
                if b:
                    w = b
                    break
        if w.endswith("tion") and len(w) > 6:
            w = w[:-3]
    if len(w) > 2 and w.endswith("e") and not w.endswith("ee"):
        w = w[:-1]
    return w


def content_stems(text: str) -> list[str]:
    return [stem(t) for t in words(text) if t not in STOP and not t.isdigit() and len(t) > 1]


# derivational families: every word in a group shares one association code (overlap detection still uses stem())
FAMILY_GROUPS = """
pollen pollinate pollination pollinator pollinated pollinating
magnet magnetic magnetism magnetize magnetized
electric electricity electrical electronic
sun solar sunlight sunshine
moon lunar
star stellar
gravity gravitational
atom atomic
nucleus nuclear
volcano volcanic
glacier glacial
ocean oceanic
cell cellular
climate climatic
reproduce reproduction reproductive
digest digestion digestive
circulate circulation circulatory
respire respiration respiratory breathe breathing breath
excrete excretion excretory
nerve nervous neuron
muscle muscular
skeleton skeletal bone bones
immune immunity
infect infection infectious
evaporate evaporation vapor vaporize vaporization
condense condensation
precipitate precipitation
erode erosion erosional
deposit deposition
absorb absorption absorbs
reflect reflection reflective
refract refraction
conduct conduction conductor conductive conductivity
insulate insulation insulator
convect convection
radiate radiation radiant
expand expansion
contract contraction
dissolve dissolving solubility soluble
germinate germination
migrate migration migratory
hibernate hibernation
evolve evolution evolutionary
mutate mutation mutant
extinct extinction
inherit inheritance inherited heredity hereditary
adapt adaptation adaptive
compete competition competitive
consume consumer consumption
produce producer production
decompose decomposer decomposition decay
predator predation predatory
photosynthesis photosynthetic photosynthesize
transpire transpiration
vibrate vibration vibrating
rotate rotation rotational
revolve revolution
orbit orbital
pollute pollution pollutant
conserve conservation
renew renewable
fertilize fertilization
oxidize oxidation
combust combustion
hypothesis hypothesize hypothesis
predict prediction
observe observation
measure measurement
experiment experimental
vary variable variation
survive survival
depend dependent dependence
friction frictional
thermal heat
temperature thermometer
freeze freezing frozen
melt melting molten
""".strip().split("\n")

FAMILY = {}
for _g in FAMILY_GROUPS:
    _ws = _g.split()
    for _w in _ws:
        FAMILY.setdefault(stem(_w), stem(_ws[0]))


def concept(w: str) -> str:
    """The association key of a word: its stem, merged with its derivational family."""
    s = stem(w)
    return FAMILY.get(s, s)
