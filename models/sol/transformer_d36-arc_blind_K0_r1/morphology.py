"""Explicit English inflections and spelling rules; no learned morphology.

Only word-form equivalences are used here. Science associations stay in
knowledge.py. Existing explicit forms take priority over generated forms.
"""

FAMILIES = r'''
leaf leaves
root roots
stem stems
flower flowers
seed seeds
plant plants
animal animals
organism organisms
cell cells
nucleus nuclei
mitochondrion mitochondria
chloroplast chloroplasts
bacterium bacteria
fungus fungi funguses
alga algae
cactus cacti cactuses
larva larvae
pupa pupae
vertebra vertebrae
species
fish fishes
goose geese
mouse mice
tooth teeth
foot feet
child children
person people
chromosome chromosomes
gene genes
allele alleles
artery arteries
vein veins
capillary capillaries
alveolus alveoli
villus villi
hypothesis hypotheses
analysis analyses
analyze analyzes analyzed analyzing analyse analysed analysing
gas gases gasses
axis axes
crisis crises
echo echoes
volcano volcanoes volcanos
potato potatoes
phenomenon phenomena
criterion criteria
index indices indexes
radius radii radiuses
spectrum spectra spectrums
medium media mediums
matrix matrices
formula formulas formulae
calorie calories
center centers centre centres
centimeter centimeters centimetre centimetres
millimeter millimeters millimetre millimetres
kilometer kilometers kilometre kilometres
meter meters metre metres
liter liters litre litres
milliliter milliliters millilitre millilitres
aluminum aluminium
sulfur sulphur
color colors colour colours colored coloured coloring colouring
colored coloured
coloration colouration
behavior behaviors behaviour behaviours
fiber fibers fibre fibres
fertilizer fertilizers fertiliser fertilisers
fertilization fertilisation
vapor vapors vapour vapours
vaporization vaporisation vapourization vapourisation
hemoglobin haemoglobin
paleontology palaeontology
absorb absorbs absorbed absorbing
accelerate accelerates accelerated accelerating
adapt adapts adapted adapting
attract attracts attracted attracting
balance balances balanced balancing
bend bends bent bending
blow blows blew blown blowing
boil boils boiled boiling
breathe breathes breathed breathing
burn burns burned burnt burning
carry carries carried carrying
classify classifies classified classifying
collide collides collided colliding
combine combines combined combining
compress compresses compressed compressing
condense condenses condensed condensing
conduct conducts conducted conducting
conserve conserves conserved conserving
consume consumes consumed consuming
contract contracts contracted contracting
cool cools cooled cooling
create creates created creating
decay decays decayed decaying
destroy destroys destroyed destroying
decompose decomposes decomposed decomposing
decrease decreases decreased decreasing
deposit deposits deposited depositing
digest digests digested digesting
dissolve dissolves dissolved dissolving
divide divides divided dividing
drain drains drained draining
drift drifts drifted drifting
eat eats ate eaten eating
erode erodes eroded eroding
erupt erupts erupted erupting
evaporate evaporates evaporated evaporating
expand expands expanded expanding
fall falls fell fallen falling
feed feeds fed feeding
filter filters filtered filtering
float floats floated floating
flow flows flowed flowing
form forms formed forming
freeze freezes froze frozen freezing
germinate germinates germinated germinating
grow grows grew grown growing
heat heats heated heating
hide hides hid hidden hiding
hibernate hibernates hibernated hibernating
hunt hunts hunted hunting
increase increases increased increasing
insulate insulates insulated insulating
investigate investigates investigated investigating
lift lifts lifted lifting
magnify magnifies magnified magnifying
measure measures measured measuring
melt melts melted melting
migrate migrates migrated migrating
mix mixes mixed mixing
move moves moved moving
observe observes observed observing
orbit orbits orbited orbiting
oscillate oscillates oscillated oscillating
pollinate pollinates pollinated pollinating
pollute pollutes polluted polluting
predict predicts predicted predicting
protect protects protected protecting
pull pulls pulled pulling
pump pumps pumped pumping
push pushes pushed pushing
react reacts reacted reacting
recycle recycles recycled recycling
reflect reflects reflected reflecting
refract refracts refracted refracting
release releases released releasing
repel repels repelled repelling
repeat repeats repeated repeating
reproduce reproduces reproduced reproducing
respire respires respired respiring
revolve revolves revolved revolving
rise rises rose risen rising
rotate rotates rotated rotating
rub rubs rubbed rubbing
rust rusts rusted rusting
separate separates separated separating
settle settles settled settling
sink sinks sank sunk sinking
slide slides slid sliding
spin spins spun spinning
split splits splitting
spread spreads spreading
sprout sprouts sprouted sprouting
stretch stretches stretched stretching
sublimate sublimates sublimated sublimating
support supports supported supporting
survive survives survived surviving
sweat sweats sweated sweating
swim swims swam swum swimming
test tests tested testing
transfer transfers transferred transferring
transform transforms transformed transforming
transmit transmits transmitted transmitting
transport transports transported transporting
trap traps trapped trapping
vibrate vibrates vibrated vibrating
wash washes washed washing
weathering weathered weathers
weigh weighs weighed weighing
'''


def canonical(word, mapping):
    seen = set()
    while word in mapping and mapping[word] != word:
        if word in seen:
            raise ValueError('Cycle in explicit morphology: ' + word)
        seen.add(word)
        word = mapping[word]
    return word


def aliases_for(seed_words, stop_words=()):
    """Map inflections/possessives to canonical forms without examining text."""
    mapping = {}
    for row in FAMILIES.splitlines():
        forms = row.split()
        if forms:
            for form in forms[1:]:
                mapping[form] = forms[0]
    # Only science content words get generated plural and possessive forms.
    # Do not expand generic function words or the entries' concept names.
    candidates = (set(seed_words) | set(mapping) | set(mapping.values())) - set(stop_words)
    for word in sorted(candidates):
        if not word.isalpha() or len(word) < 3:
            continue
        base = canonical(word, mapping)
        if word not in mapping:
            if word.endswith('y') and word[-2] not in 'aeiou':
                plural = word[:-1] + 'ies'
            elif word.endswith(('ch', 'sh', 'ss', 'x')):
                plural = word + 'es'
            elif not word.endswith('s'):
                plural = word + 's'
            else:
                plural = None
            if plural is not None:
                mapping.setdefault(plural, base)
        mapping.setdefault(word + "'s", base)
        if word.endswith('s'):
            mapping.setdefault(word + "'", base)
    # Also cover possessives of generated regular plurals, e.g. plants'.
    for word in list(mapping):
        if word.isalpha():
            base = canonical(word, mapping)
            mapping.setdefault(word + "'s", base)
            if word.endswith('s'):
                mapping.setdefault(word + "'", base)
    return {word: canonical(base, mapping) for word, base in mapping.items()}
