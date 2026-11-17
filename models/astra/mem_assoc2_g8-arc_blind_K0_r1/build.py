"""Compile explicitly authored science rules into the fixed association memory.
No fitting, external data, random weights, or inference-time retrieval.
"""
from pathlib import Path
import itertools
import re
from collections import defaultdict
import numpy as np
import torch
from audit import audit_rules, ANCHOR, COMPARATIVE_SELECTORS

ROOT = Path(__file__).resolve().parent
TOKEN = re.compile(r"[a-z0-9']+")

# Exact, authored names for compounds that may also occur as one formula token.
# These are lexical alternatives, not inferred chemical reactions or fact reversals.
PHRASE_ALTERNATIVES = (
    ('carbon dioxide', 'carbondioxide'),
    ('sulfur dioxide', 'sulfurdioxide'),
    ('nitrous oxide', 'nitrousoxide'),
    ('hydrogen peroxide', 'hydrogenperoxide'),
    ('sodium chloride', 'sodiumchloride'),
    ('calcium carbonate', 'calciumcarbonate'),
    ('silicon dioxide', 'silica'),
    ('hydrogen chloride', 'hydrogenchloride'),
    ('carbon monoxide', 'carbonmonoxide'),
    ('sodium hydroxide', 'sodiumhydroxide'),
    ('sodium bicarbonate', 'sodiumbicarbonate'),
    ('sodium hydrogen carbonate', 'sodiumbicarbonate'),
    ('calcium hydroxide', 'calciumhydroxide'),
    ('sulfuric acid', 'sulfuricacid'),
    ('nitric acid', 'nitricacid'),
    ('acetic acid', 'aceticacid'),
)

# Authored spacing variants. Unlike formula substitutions, these also expand
# fused source cues into the ordinary two-word spelling used by the tokenizer.
# This is a closed lexical list, not generic fact reversal or inferred knowledge.
SURFACE_COMPOUNDS = (
    ('food chain', 'foodchain'),
    ('food web', 'foodweb'),
    ('water cycle', 'watercycle'),
    ('fossil fuel', 'fossilfuel'),
    ('fossil fuels', 'fossilfuels'),
    ('sea floor', 'seafloor'),
    ('sea bed', 'seabed'),
    ('light year', 'lightyear'),
    ('light years', 'lightyears'),
    ('salt water', 'saltwater'),
    ('fresh water', 'freshwater'),
    ('sea water', 'seawater'),
    ('ground water', 'groundwater'),
    ('rain water', 'rainwater'),
    ('rain forest', 'rainforest'),
    ('blood stream', 'bloodstream'),
    ('solar cell', 'solarcell'),
    ('light bulb', 'lightbulb'),
    ('air column', 'aircolumn'),
    ('net force', 'netforce'),
    ('root nodules', 'rootnodules'),
    ('pressure cooker', 'pressurecooker'),
    ('earth shadow', 'earthshadow'),
    ('moon shadow', 'moonshadow'),
    ('dry mass', 'drymass'),
    ('cell wall', 'cellwall'),
    ('host cell', 'hostcell'),
    ('sea breeze', 'seabreeze'),
    ('land breeze', 'landbreeze'),
    ('paper plane', 'paperplane'),
    ('power line', 'powerline'),
    ('tuning fork', 'tuningfork'),
    ('warm blooded', 'warmblooded'),
    ('cold blooded', 'coldblooded'),
    ('bar graph', 'bargraph'),
    ('line graph', 'linegraph'),
    ('pie chart', 'piechart'),
    ('scatter plot', 'scatterplot'),
    ('low melting', 'lowmelting'),
    ('long term', 'longterm'),
)

# Grammatical words do not constitute science evidence.
STOP = set('''a an the and or of to in on at by for from with into onto upon through
that this these those it its they their them he she his her we our you your i
is are was were be been being am do does did doing has have had having
will would shall should can could may might must which what who whom whose
where when why how than then there here as if so because therefore thus
also only very most more much many some any each every both either neither
such other another same following below above shown shows show showing
example examples statement statements describes describe description
best better correctly correct likely probably possible possibly usually
happens happen happening occurs occur occurring according based using used use
one two three four five six seven eight nine ten first second third fourth
student students teacher class classroom scientist scientists researchers researcher
investigation investigations experiment experiments activity activities study studies
observed observes observe observation observations information data conclusion conclusions
question questions answer answers choice choices option options diagram diagrams
picture pictures table tables graph graphs figure figures illustration illustrations
certain particular several different difference differences similar similarity
result results resulting cause causes caused causing make makes made making
thing things way ways time times part parts type types kind kinds form forms
new old large small long short gets get got given determine determines
need needs needed able ability help helps helped affect affects affected
about during before after while until since between among within without
next last often sometimes always never not no except least less instead
'''.split())
# Negation and several relational terms are intentional concepts, not discarded.
STOP -= set('not no except least less different similar before after cause result need'.split())


def sources():
    alias = {}
    groups = []
    for raw in (ROOT / 'aliases.txt').read_text().splitlines():
        raw = raw.split('#', 1)[0].strip()
        if not raw:
            continue
        terms = raw.split()
        canonical = alias.get(terms[0], terms[0])
        groups.append(terms)
        for term in terms:
            if term in alias and alias[term] != canonical:
                raise ValueError('Repeated alias: ' + term)
            alias[term] = canonical
    # Explicitly authored additions can unify previously disconnected inflections.
    # Resolve entire old groups, rather than silently changing just one spelling.
    def root(term):
        seen = set()
        while term in alias and alias[term] != term:
            if term in seen:
                raise ValueError('Cyclic supplementary alias: ' + term)
            seen.add(term)
            term = alias[term]
        return term

    supplemental = ('aliases_extra.txt', 'aliases_g5.txt', 'aliases_g6.txt', 'aliases_g7.txt', 'aliases_g8.txt')
    for raw in itertools.chain.from_iterable(
            (ROOT / filename).read_text().splitlines() for filename in supplemental):
        terms = raw.split('#', 1)[0].split()
        if not terms:
            continue
        canonical = root(terms[0])
        groups.append(terms)
        for term in terms:
            old = root(term)
            if old != canonical:
                alias[old] = canonical
            alias[term] = canonical
    alias = {word: root(word) for word in alias}
    rules = []
    for file in sorted(ROOT.glob('facts*.txt')):
        for line_no, raw in enumerate(file.read_text().splitlines(), 1):
            raw = raw.split('#', 1)[0].strip()
            if not raw:
                continue
            fields = [s.strip().lower() for s in raw.split('|')]
            if len(fields) not in (2, 3, 4):
                raise ValueError(f'{file.name}:{line_no}: malformed rule')
            cue, positive = fields[:2]
            negative = fields[2] if len(fields) >= 3 else ''
            scale = float(fields[3]) if len(fields) == 4 else 1.0
            rules.append((cue, positive, negative, scale))
    return alias, groups, rules


def extend_surface_forms(words, aliases):
    """Recognize possessives and single-quoted tokens without new concepts.

    The fixed tokenizer retains ASCII apostrophes: Earth's is one token, not
    earth + s. Unknown possessives would therefore lose the entire noun. The
    same issue affects single-quoted scientific terms and quoted phrases.
    Explicit aliases take priority, and expansion is finite and idempotent.
    """
    for word in sorted(words):
        root = aliases.get(word, word)
        # Regular plural possessives use a final apostrophe, supplied below.
        # Singular roots ending in s, e.g. mass, may also take apostrophe-s.
        if word.isalpha() and len(word) > 1 and (not word.endswith('s') or word == root):
            form = word + "'s"
            aliases.setdefault(form, root)
            words.add(form)
    # One opening quote, closing quote, or a matched pair. Also works for
    # authored contractions and the possessives just generated. Avoid recursive
    # quoting if this helper is called again on its own output.
    for word in sorted(words):
        if word.startswith("'") or word.endswith("'"):
            continue
        root = aliases.get(word, word)
        for form in ("'" + word, word + "'", "'" + word + "'"):
            aliases.setdefault(form, root)
            words.add(form)
    return words, aliases


def lexicon():
    alias, groups, rules = sources()
    # Keep grammar filtering local: build() is pure with respect to module state.
    # Authored concepts such as "second" and "new" override the grammar list.
    stop = STOP.difference(alias)
    words = set()
    for cue, pos, neg, _ in rules:
        words.update(TOKEN.findall(cue + ' ' + pos + ' ' + neg))
    for group in groups:
        words.update(group)
    for phrase, fused in SURFACE_COMPOUNDS:
        words.update(TOKEN.findall(phrase + ' ' + fused))
    words.difference_update(stop)
    # Only conservative inflections: explicit aliases take priority.
    # Generate from source words, then resolve roots before adding generated
    # forms. Otherwise e.g. 'organ' -> 'organs' plus 'organs' -> 'organses'
    # creates a disconnected second concept even though both source words are
    # already the same concept. This is a lexical invariant, not corpus fitting.
    base_alias = dict(alias)
    generated = {}
    for word in sorted(words):
        root = base_alias.get(word, word)
        if len(word) > 3 and word.isalpha() and word == root:
            if word.endswith('y') and word[-2] not in 'aeiou':
                form = word[:-1] + 'ies'
            elif word.endswith(('s', 'x', 'ch', 'sh')):
                form = word + 'es'
            else:
                form = word + 's'
            if form not in alias:
                alias[form] = root
                generated[form] = root

    def canonical(word):
        seen = set()
        while word in alias and alias[word] != word:
            if word in seen:
                raise ValueError('Cyclic morphological alias: ' + word)
            seen.add(word)
            word = alias[word]
        return word

    redundant = [form for form, root in generated.items()
                 if form not in words and canonical(root) != root]
    for form in redundant:
        del alias[form]
    alias = {word: canonical(root) for word, root in alias.items()}
    words.update(w for w in alias if w not in stop)
    words, alias = extend_surface_forms(words, alias)
    return sorted(words), alias, rules


def phrase_variants(text):
    """Finite exact lexical alternatives; apostrophes count as token characters."""
    variants = {text}
    for phrase, fused in SURFACE_COMPOUNDS:
        pattern = r"(?<![a-z0-9'])" + re.escape(fused) + r"(?![a-z0-9'])"
        variants.update(re.sub(pattern, phrase, value) for value in tuple(variants))
    for phrase, replacement in PHRASE_ALTERNATIVES + SURFACE_COMPOUNDS:
        pattern = r"(?<![a-z0-9'])" + r"\s+".join(map(re.escape, phrase.split())) + r"(?![a-z0-9'])"
        variants.update(re.sub(pattern, replacement, value) for value in tuple(variants))
    return tuple(sorted(variants))


def compile_rules(vocab, aliases, rules):
    """Preserve original rules and their explicitly authored formula/name forms.

    Both cue and target alternatives are needed: otherwise CO2 in an option would
    lose facts that recalled the two words 'carbon dioxide'. Signs, conditions,
    and scales are preserved. No generic reversal or multi-hop inference occurs.
    """
    known = set(vocab)
    def concepts(text):
        return {aliases.get(t, t) for t in TOKEN.findall(text) if t in known}
    compiled = set()
    for cue, positive, negative, scale in rules:
        anchors = concepts(' '.join(ANCHOR.findall(cue)))
        for q, p, n in itertools.product(phrase_variants(cue), phrase_variants(positive),
                                          phrase_variants(negative)):
            qs, ps, ns = concepts(q), concepts(p), concepts(n)
            # A spelling alternative must not change the meaning of a mandatory
            # cue (e.g. @foodchain cannot silently become a mandatory food token).
            if concepts(' '.join(ANCHOR.findall(q))) != anchors:
                continue
            # Compression cannot turn an explicitly pair-only condition into an
            # unconditional single-concept rule. Keep the original pair instead.
            if (scale < 0 or anchors) and len(qs) < 2:
                continue
            # A pure spelling bridge can become x -> x when compressed. The
            # architecture masks that target, so this variant is a tautology.
            if (ps | ns) - qs:
                compiled.add((q, p, n, scale))
    return sorted(compiled)


def condition_guard(cues, selectors):
    """Retain an unambiguous comparison, without guessing roles in a contrast.

    This is a source-level logical restriction, not a learned word statistic.
    Multiple different selectors may describe a relation (e.g. high versus low),
    so these retain the original encoding rather than guessing which is a guard.
    """
    present = set(cues).intersection(selectors)
    return present if len(cues) > 1 and len(present) == 1 else set()


def recall_pairs(cues, negators, guards=(), anchors=()):
    """A retained pair must not silently drop an explicit condition.

    Negation takes priority over inferred comparative guards. Authored @anchors
    are mandatory in addition to either condition. This only restricts the pairs
    emitted at build time; it adds no inference-time parser or third-order logic.
    """
    required = set(cues).intersection(negators) or set(guards)
    mandatory = set(anchors)
    return tuple((a, b) for a, b in itertools.combinations(cues, 2)
                 if (not required or a in required or b in required)
                 and mandatory.issubset((a, b)))


# Closed authored alternatives. They are defeasible defaults, not universally
# exclusive scientific categories (e.g. the water cycle has both evaporation
# and condensation). Keep this shared with source-level regression tests.
OPPOSITION_TEXT = '''increase decrease
more less
hot cold
fast slow
positive negative
attract repel
north south
east west
identical different
absorb release
conduct insulate
renewable nonrenewable
endothermic exothermic
series parallel
open closed
summer winter
transparent opaque
melting freezing
evaporation condensation
larger smaller
high low
strong weak
independent dependent
sexual asexual
mitosis meiosis
'''


def inferred_opposites(positive, negative, groups):
    """Infer candidate negatives, excluding either explicitly authored sign."""
    inferred = set()
    for group in groups:
        hit = group.intersection(positive)
        if len(hit) == 1:
            inferred.update(group - hit)
    return inferred.difference(positive).difference(negative)


def merge_opposition_defaults(authored, defaults):
    """Apply inferred negatives only where no explicit positive supports the slot.

    Keys are semantic singletons or pairs, BEFORE hashing. Explicit negative
    evidence is retained unchanged if a positive blocks a default. Without an
    explicit positive, the inherited strongest-negative rule still applies.
    This is symbolic provenance precedence, independent of strength thresholds,
    source frequency, source order, questions, or accuracy.
    """
    for key, targets in defaults.items():
        destination = authored.setdefault(key, {})
        for target, (positive, negative) in targets.items():
            if positive != 0.0 or negative > 0.0:
                raise ValueError('Opposition defaults must contain only negative evidence')
            old_positive, old_negative = destination.get(target, (0.0, 0.0))
            if old_positive == 0.0:
                destination[target] = (old_positive, min(old_negative, negative))


def build():
    vocab = (ROOT / 'vocab.txt').read_text().splitlines()
    expected, aliases, rules = lexicon()
    if vocab != expected:
        raise ValueError('vocab.txt does not match authored lexicon; run build.py to regenerate')
    problems = audit_rules(vocab, aliases, rules)
    if problems:
        raise ValueError('Authored-rule audit failed:\n' + '\n'.join(problems))
    rules = compile_rules(vocab, aliases, rules)
    problems = audit_rules(vocab, aliases, rules)
    if problems:
        raise ValueError('Lexical-variant audit failed:\n' + '\n'.join(problems))
    names = sorted({aliases.get(w, w) for w in vocab})
    ids = {name: i + 1 for i, name in enumerate(names)}
    C = len(names) + 1
    concept = np.zeros(len(vocab) + 2, dtype=np.int64)
    for i, word in enumerate(vocab, 2):
        concept[i] = ids[aliases.get(word, word)]

    known = set(vocab)

    def concepts(text):
        return sorted({ids[aliases.get(w, w)] for w in TOKEN.findall(text.lower()) if w in known})

    # Fixed semantic salience, chosen by meaning rather than corpus statistics.
    idf = np.full(C, 1.0, dtype=np.float32)
    idf[0] = 0
    generic = '''increase decrease change energy matter object material process system environment
    living organism animal plant water air earth human person body grow movement motion
    hot cold heat temperature light force mass size amount number speed direction
    produce require contain absorb release store transfer carry break combine separate
    not except least less different similar before after cause result need all
    survive die food color shape location distance measure explain test predict
    protect support structure function natural artificial advantage disadvantage'''
    for c in concepts(generic):
        idf[c] = 0.72
    relational = '''increase decrease not except least less before after produce require absorb release
    change compare faster slower positive negative more fewer high low greatest smallest'''
    for c in concepts(relational):
        idf[c] = 0.84
    commonplace = '''experiment observe conclusion variable control measure test predict scientific
    amount number object material size large small long short same different similar
    body human animal plant all one two three four five six seven eight nine ten
    first last second third fourth time day year location before after cause result
    produce require contain change grow development structure function role purpose
    inside outside center surface high low largest least strong weak bright dim
    heavy lightweight far near energy matter process system environment explain class family
    identify collect record obtain investigate condition factor relationship interaction
    connection sample trial setup treatment experimental represent source property
    example amount purpose method procedure materials steps perform determine
    observed measured given general particular certain demonstrate show description
    close complete incomplete'''
    for c in concepts(commonplace):
        idf[c] = min(idf[c], 0.76)
    # Comparatives retained by condition_guard must survive the top-16 bottleneck.
    selectors = set(concepts(' '.join(sorted(COMPARATIVE_SELECTORS))))
    for c in selectors:
        idf[c] = 1.03
    # Polarity and reactant/product selectors must survive the top-16 bottleneck.
    for c in concepts('increase decrease produce require absorb release above below'):
        idf[c] = 1.06
    for c in concepts('not except'):
        idf[c] = 1.12
    # A minute deterministic tie-break keeps selection stable; it encodes no answers.
    for i in range(1, C):
        idf[i] += (i % 97) * 0.00001

    A = np.zeros((C, C), dtype=np.float32)
    first = defaultdict(dict)
    pairs = defaultdict(dict)
    default_first = defaultdict(dict)
    default_pairs = defaultdict(dict)
    # A fixed 50% enlargement of the generation-6 address space reduces false
    # recall from unrelated pairs while retaining every target and a <2 GiB peak.
    # Chosen from the tensor memory budget, never from question accuracy.
    H = 1572869

    def add_max(table, key, target, value):
        positive, negative = table[key].get(target, (0.0, 0.0))
        # Independent maxima prevent repetition from becoming training frequency.
        # Opposing evidence cancels instead of favoring whichever file came first.
        table[key][target] = (max(positive, value), min(negative, value))

    def resolve(table):
        result = {}
        for key, targets in table.items():
            resolved = {target: pos + neg for target, (pos, neg) in targets.items()
                        if abs(pos + neg) > 1e-12}
            if resolved:
                result[key] = resolved
        return result

    # Candidate semantic opposites remain useful when a supported answer concept
    # is masked by the question. Their negative evidence is a default, however,
    # and cannot refute an explicit positive for the same semantic slot.
    opposite_groups = [set(concepts(line)) for line in OPPOSITION_TEXT.splitlines()]
    negators = set(concepts('not except'))
    # Only concepts with authored evidence or a generated logical role should
    # occupy top-J slots. A listed but disconnected word has no useful recall.
    active = set(negators)
    for group in opposite_groups:
        active.update(group)

    for cue, positive, negative, scale in rules:
        qs, ps, ns = concepts(cue), concepts(positive), concepts(negative)
        active.update(qs)
        active.update(ps)
        active.update(ns)
        if not qs:
            raise ValueError('Empty cue: ' + cue)
        guards = condition_guard(qs, selectors)
        anchors = set(concepts(' '.join(ANCHOR.findall(cue))))
        cue_pairs = recall_pairs(qs, negators, guards, anchors)
        if anchors and not cue_pairs:
            raise ValueError('Mandatory condition has no representable pair: ' + cue)
        # Each fact distributes evidence across its answer concepts. Short precise
        # descriptions stay stronger per word than verbose lists.
        inferred = inferred_opposites(set(ps), set(ns), opposite_groups)
        for targets, sign, is_default in ((ps, 1.0, False), (ns, -1.0, False),
                                          (sorted(inferred), -0.55, True)):
            first_destination = default_first if is_default else first
            pair_destination = default_pairs if is_default else pairs
            # Normalize only recallable targets. Masked cue concepts must not
            # dilute usable evidence. Infer contrasts above before filtering:
            # an explicit positive polarity can still oppose its alternative.
            targets = [target for target in targets if target not in qs]
            if not targets:
                continue
            weight = sign * scale / (len(targets) ** 0.35)
            for target in targets:
                for q in qs:
                    # Multiword conditions are not independent facts. A small
                    # backoff is retained, unless pair-only (negative scale) or
                    # negated: "no nucleus" must not teach nucleus -> bacteria.
                    if scale > 0 and not negators.intersection(qs) and not guards and not anchors:
                        add_max(first_destination, q, target, weight / max(1, len(qs)) ** 1.5)
                for pair in cue_pairs:
                    pair_value = sign * abs(scale) / (len(targets) ** 0.35)
                    pair_value /= max(1.0, (len(qs) - 1) ** 0.7)
                    # Keep semantic pair identity until all logical operations
                    # finish; a hash collision is not an authored exception.
                    add_max(pair_destination, pair, target, pair_value)

    # Authoritative signs are now complete, so this cannot depend on fact order.
    # A singleton assertion does not block a pair-specific default, or vice versa.
    merge_opposition_defaults(first, default_first)
    merge_opposition_defaults(pairs, default_pairs)
    del default_first, default_pairs, first_destination, pair_destination
    first = resolve(first)
    for q, targets in first.items():
        # Broad prompts contribute less than a specific scientific noun.
        damping = max(1.0, (len(targets) / 12.0) ** 0.55)
        for target, value in targets.items():
            A[q, target] = value / damping
    # Negated recall questions ("not" / "except") invert the direct definition
    # evidence. This is a symbolic transformation, not fitted from questions.
    # Broad generic concepts are excluded; their senses are too ambiguous.
    authored_slots = {(pair, target) for pair, targets in pairs.items() for target in targets}
    for q, targets in first.items():
        if idf[q] < 0.95:
            continue
        ranked = sorted(targets, key=lambda t: (-abs(float(A[q, t])), t))[:48]
        for neg in negators:
            if q == neg:
                continue
            pair = tuple(sorted((q, neg)))
            for target in ranked:
                # Only an exception for this exact pair outranks its inversion.
                if target != neg and (pair, target) not in authored_slots:
                    add_max(pairs, pair, target, -1.5 * 0.65 * float(A[q, target]) / float(idf[neg]))
    # Hash only at the final storage stage. Preserve independent signed maxima
    # for unavoidable collisions, exactly as for repeated evidence.
    hashed = defaultdict(dict)
    for (a, b), targets in pairs.items():
        row = (a * 1000003 + b) % H
        for target, (positive, negative) in targets.items():
            add_max(hashed, row, target, positive)
            add_max(hashed, row, target, negative)
    pairs = resolve(hashed)
    # These compiler intermediates do not contribute to subsequent allocation.
    # Release their nested dictionaries before creating the dense recall tables;
    # this changes neither evidence nor tensor bytes, and lowers peak live memory.
    del first, hashed, authored_slots, rules
    for c in range(1, C):
        if c not in active:
            idf[c] = 0.0

    # Retain every nonzero recall (including occasional hash unions).
    R = max(1, max(map(len, pairs.values()), default=1))
    if R > 160:
        raise ValueError(f'Unexpectedly wide memory row: {R}')
    row_ids = np.zeros((H, R), dtype=np.int64)
    row_w = np.zeros((H, R), dtype=np.float32)
    for row, targets in pairs.items():
        for j, (target, value) in enumerate(sorted(targets.items())):
            row_ids[row, j] = target
            row_w[row, j] = value
    return {
        'concept': torch.from_numpy(concept),
        'idf': torch.from_numpy(idf),
        'A': torch.from_numpy(A),
        'row_ids': torch.from_numpy(row_ids),
        'row_w': torch.from_numpy(row_w),
        'mix': torch.tensor(0.65, dtype=torch.float32),
    }


if __name__ == '__main__':
    words, _, rules = lexicon()
    (ROOT / 'vocab.txt').write_text('\n'.join(words) + '\n')
    print(f'Wrote {len(words)} vocabulary entries from {len(rules)} authored rules.')
