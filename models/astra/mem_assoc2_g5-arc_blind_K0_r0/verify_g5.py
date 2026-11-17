"""Generation-5 source and isolated tensor contracts; no answer labels or fitting."""
from pathlib import Path
import gc
import hashlib
import torch
import build
from audit import audit_rules


# These expectations describe authored facts, not multiple-choice answers.
CONDITIONS = {
    'thermos vacuum': ('insulation', 'conduction convection'),
    'velocity unchanged': ('zero balanced', 'acceleration'),
    'seed boiling': ('embryo dead', 'germination'),
    'moon crater': ('preservation', 'erosion'),
    'moon airless': ('vacuum', 'weather'),
    'moon footprints': ('preservation', 'wind erosion'),
    'soil frozen': ('runoff', 'infiltration'),
    'soil packed': ('runoff', 'infiltration'),
    'soil contour': ('conservation', 'erosion runoff'),
    'soil terracing': ('conservation', 'erosion runoff'),
    'mulch moisture': ('retention', 'evaporation'),
    'leaf petroleum': ('stomata', 'transpiration exchange'),
    'seedling pale': ('etiolation', 'chlorophyll'),
    'compressed pressure': ('increase', 'decrease'),
    'compressed volume': ('decrease', 'increase'),
    'smooth oil': ('lubricant', 'friction'),
    'work stationary': ('zero', 'displacement'),
    'work perpendicular': ('zero', 'transfer'),
    'predator removal': ('prey increase', 'decrease'),
    'prey removal': ('predator decrease', 'increase'),
    'producer removal': ('consumer decrease', 'increase'),
    'herbivore removal': ('vegetation increase', 'decrease'),
    'competitor removal': ('resource increase', 'decrease'),
    'carbonmonoxide poisoning': ('hemoglobin hypoxia', 'oxygen transport'),
    'breaker opened': ('protection', 'current'),
    'circuit completed': ('closed current', ''),
    'thicker resistance': ('decrease', 'increase'),
    'thicker current': ('increase', 'decrease'),
    'longer resistance': ('increase', 'decrease'),
    'longer current': ('decrease', 'increase'),
    'ice salted': ('melting', 'freezing'),
    # Reduced quantities remain recallable along with the correct direction.
    'fish streamlined': ('decrease drag resistance', 'increase'),
    'snail shell': ('conservation decrease dehydration', 'increase'),
    'cactus spines': ('conservation decrease transpiration', 'increase'),
    'resource partitioning': ('coexistence decrease competition', 'increase'),
    'blind experiment': ('masking bias decrease', 'increase'),
    'calorimeter insulation': ('retention decrease exchange', 'increase'),
    'thermal pollution': ('oxygen decrease hypoxia', 'increase'),
    'hot wastewater': ('oxygen decrease', 'increase'),
    'hibernation metabolism': ('slow decrease', 'increase'),
    'dormancy metabolism': ('slow decrease', 'increase'),
    'solute less': ('concentration decrease', 'increase'),
    'solute more': ('concentration increase', 'decrease'),
    'solvent less': ('concentration increase', 'decrease'),
    'solvent more': ('concentration decrease', 'increase'),
    'streamlined drag': ('decrease', 'increase'),
    'streamlined resistance': ('decrease', 'increase'),
    'trophic higher': ('energy decrease', 'increase'),
    'trophic lower': ('energy increase', 'decrease'),
    'dormant living': ('viable respiration', 'dead'),
    'systematic averaging': ('persistent bias', 'eliminated'),
    'systematic repetition': ('persistent bias', 'eliminated'),
    'random averaging': ('uncertainty decrease', 'increase'),
    'dilution moles': ('unchanged solute amount', 'increase decrease'),
    'dilution solvent': ('addition increase', 'decrease'),
    'shell dehydration': ('decrease conservation', 'increase'),
    'spines transpiration': ('decrease conservation', 'increase'),
    # Mechanics: keep motion direction, components, and comparison selectors.
    'uniform straight': ('balanced zero resultant', 'acceleration'),
    'sustained rest': ('balanced zero resultant', 'acceleration'),
    'constant speed': ('magnitude', 'faster slower'),
    'circular speed': ('direction velocity acceleration', ''),
    'speed turning': ('direction acceleration', ''),
    'projectile horizontal': ('constant velocity inertia', 'acceleration'),
    'apex vertical': ('zero velocity', ''),
    'apex acceleration': ('gravity downward', 'zero'),
    'apex force': ('gravity downward', 'zero'),
    'gravity farther': ('attraction weaker decrease', 'stronger increase'),
    'gravity closer': ('attraction stronger increase', 'weaker decrease'),
    'gravity heavier': ('attraction increase', 'decrease'),
    'gravity lighter': ('attraction decrease', 'increase'),
    'pendulum heavier': ('unchanged period', 'increase decrease'),
    'pendulum lighter': ('unchanged period', 'increase decrease'),
    'frequency increase': ('wavelength decrease shorter', 'longer'),
    'frequency decrease': ('wavelength increase longer', 'shorter'),
    'wavelength increase': ('frequency decrease', 'increase'),
    'wavelength decrease': ('frequency increase', 'decrease'),
    'amplitude increase': ('intensity energy loudness increase', 'decrease'),
    'amplitude decrease': ('intensity energy loudness decrease', 'increase'),
    'centripetal work': ('zero perpendicular', 'energy transfer'),
    'centripetal acceleration': ('inward radial center', 'outward'),
    # Matter: retain the process/phase qualifier and distinguish solution ratios.
    'bond breaking': ('energy absorption endothermic', 'release'),
    'bond formation': ('energy release exothermic', 'absorption'),
    'bond dissociation': ('energy absorption endothermic', 'release'),
    'concentrated unsaturated': ('limit additional dissolves', ''),
    'dilute saturated': ('low solubility maximum dissolved', ''),
    'melting plateau': ('constant temperature latent absorption', ''),
    'boiling plateau': ('constant temperature latent absorption', ''),
    'freezing plateau': ('constant temperature latent release', ''),
    'condensation plateau': ('constant temperature latent release', ''),
    'melting intramolecular': ('intact molecules identity', 'decomposition'),
    'boiling intramolecular': ('intact molecules identity', 'decomposition'),
    'boiling intermolecular': ('separation energy absorption', ''),
    'freezing intermolecular': ('ordering energy release', ''),
    'condensation intermolecular': ('attraction energy release', ''),
    # Inquiry: sampling uncertainty, selection bias, controls, and graph conditions.
    'sample larger': ('uncertainty decrease', 'increase'),
    'sample smaller': ('uncertainty increase', 'decrease'),
    'random sampling': ('population probability selection representative', ''),
    'random assignment': ('comparable groups confounding decrease', 'increase'),
    'sample biased': ('unrepresentative selection error', ''),
    'bias larger': ('persistent selection error', 'eliminated'),
    'bias repetition': ('persistent selection error', 'eliminated'),
    'sampling systematic': ('regular interval selection', ''),
    'blinding participants': ('expectation bias decrease', 'increase'),
    'blinding assessors': ('observer measurement bias decrease', 'increase'),
    'control active': ('established treatment comparison', ''),
    'control placebo': ('comparison expectation inactive', ''),
    'control positive': ('known expected response', ''),
    'control negative': ('baseline background', 'response'),
    'correlation causation': ('association', 'proof'),
    'cause correlation': ('association', 'proof'),
    'graph straight': ('linear constant slope', ''),
    'graph rising': ('positive slope', ''),
    'distance slope': ('speed rate', ''),
    'displacement slope': ('velocity', ''),
    'readings clustered': ('precision repeatability', ''),
    'readings offset': ('systematic bias calibration', ''),
    'readings scattered': ('random variation uncertainty', ''),
    # Electrical quantities, ideal transformer selectors, and magnetic state.
    'resistance larger': ('current decrease', 'increase'),
    'resistance smaller': ('current increase', 'decrease'),
    'voltage higher': ('current increase', 'decrease'),
    'voltage lower': ('current decrease', 'increase'),
    'transformer stepup': ('secondary turns voltage increase', 'decrease'),
    'transformer stepdown': ('secondary turns voltage decrease', 'increase'),
    'transformer up': ('secondary voltage increase', 'decrease'),
    'transformer down': ('secondary voltage decrease', 'increase'),
    'transformer ideal': ('power conserved energy', ''),
    'transformer frequency': ('unchanged input output', 'increase decrease'),
    'induction open': ('voltage emf', 'current'),
    'induction closed': ('voltage emf current', ''),
    'unmagnetized iron': ('ferromagnetic attracted domains', ''),
    'unmagnetized domains': ('random orientations', ''),
}


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    assert hashlib.sha256((root / 'arch.py').read_bytes()).hexdigest() == (
        '6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af')
    vocab, aliases, rules = build.lexicon()
    assert vocab == (root / 'vocab.txt').read_text().splitlines()
    assert not audit_rules(vocab, aliases, rules)
    known = set(vocab)

    def terms(text):
        return frozenset(aliases.get(t, t) for t in build.TOKEN.findall(text) if t in known)

    for left, right in (
        ('shrill', 'shriller'), ('swell', 'swollen'),
        ('etiolation', 'etiolated'), ('interlock', 'interlocking'),
        ('striation', 'striated'), ('percolation', 'percolating'),
        ('recharge', 'recharged'), ('lengthen', 'lengthened'),
        ('shorten', 'shortened'), ('tighten', 'tightened'),
        ('retain', 'retention'), ('dilate', 'dilates'),
        ('overgraze', 'overgrazed'), ('starvation', 'starving'),
        ('biomagnification', 'biomagnified'),
        ('acidification', 'acidified'), ('subduct', 'subduction'),
        ('stratosphere', 'stratospheric'), ('criterion', 'criteria'),
        ('introduced', 'introducing'), ('partition', 'partitioning'),
        ('metabolic', 'metabolism'), ('mask', 'masking'),
        ('allocation', 'allocate'), ('blind', 'blinding'),
        ('rectilinear', 'straight'), ('uniform', 'uniformly'),
        ('sustain', 'sustained'), ('instantaneous', 'instantaneously'),
        ('horizontal', 'horizontally'), ('vertical', 'vertically'),
        ('radial', 'radially'), ('concentration', 'concentrations'),
        ('bias', 'biased'), ('assign', 'assignment'),
        ('correlated', 'correlation'), ('representative', 'representativeness'),
        ('uncertainty', 'uncertain'), ('stratified', 'stratification'),
        ('volunteer', 'volunteers'), ('lottery', 'lotteries'),
        ('reading', 'readings'), ('cluster', 'clustered'),
        ('resistor', 'resistors'), ('unmagnetized', 'nonmagnetized'),
        ('unmagnetized', 'unmagnetised'), ('winding', 'windings'),
        ('ferromagnet', 'ferromagnetic'), ('remanence', 'remanent'),
    ):
        assert left in known and right in known
        assert terms(left) == terms(right), (left, right)
    for left, right in (
        ('biomagnification', 'bioaccumulation'),
        ('selection', 'selectivity'), ('sponge', 'spongy'),
        ('spore', 'pollen'), ('gene', 'allele'),
        ('accuracy', 'precision'), ('reproduce', 'reproducible'),
        ('weather', 'weathering'), ('mass', 'weight'),
        ('heat', 'temperature'), ('pressure', 'volume'),
        ('speed', 'velocity'), ('uniform', 'constant'),
        ('sustained', 'instantaneous'), ('horizontal', 'vertical'),
        ('frequency', 'period'), ('pitch', 'amplitude'),
        ('loudness', 'frequency'), ('mass', 'gravity'),
        ('concentration', 'concentrated'), ('concentrated', 'saturated'),
        ('intramolecular', 'intermolecular'),
        ('sample', 'assignment'), ('correlation', 'causation'),
        ('representative', 'representation'), ('uncertainty', 'bias'),
        ('accuracy', 'resolution'), ('precision', 'resolution'),
        ('resistance', 'resistivity'), ('resistance', 'resistor'),
        ('conductance', 'conductivity'), ('nonmagnetic', 'unmagnetized'),
    ):
        assert left in known and right in known
        assert terms(left) != terms(right), (left, right)

    # Do not turn a spatial descriptor into an outcome direction, or an
    # introduced species into an invasive one without the harmful-spread condition.
    for cue, required, forbidden in (
        ('energy pyramid', 'trophic heat', 'increase'),
        ('introduced species', 'nonnative transport', 'invasive decline'),
        ('trophic higher', 'energy decrease', 'biomass'),
        ('trophic lower', 'energy increase', 'biomass'),
    ):
        matching = [r for r in rules if terms(r[0]) == terms(cue)]
        assert matching, cue
        assert any(terms(required) <= terms(p) for _, p, _, _ in matching), cue
        assert all(not (terms(forbidden) & terms(p)) for _, p, _, _ in matching), cue
    assert not any(r[0] == 'foodchain ending' for r in rules)
    assert terms('introduced') != terms('invasive')
    assert terms('nonnative') != terms('native')
    assert terms('random') != terms('systematic')

    # The compiler expands a multi-cue fact into pairs. A qualifier in a third
    # cue must not leak a categorical claim into an unqualified physical pair.
    # These source contracts inspect every fact containing each reviewed pair.
    for pair, forbidden in (
        ('constant speed', 'zero balanced acceleration'),
        ('wave frequency', 'increase decrease'),
        ('wave wavelength', 'increase decrease'),
        ('gravity mass', 'increase decrease'),
        ('gravity distance', 'increase decrease'),
        ('pendulum length', 'faster slower increase decrease'),
        ('ball apex', 'zero'),
        ('sample larger', 'representative bias'),
        ('sample smaller', 'bias unrepresentative unreliable'),
        ('sample size', 'increase decrease confidence'),
        ('sample representative', 'random'),
        ('sampling selection', 'random representative'),
        ('random assignment', 'representative population'),
        ('control group', 'untreated inactive'),
        ('graph straight', 'proportional increase decrease'),
        ('graph rising', 'proportional'),
        ('cause correlation', 'causation'),
        ('readings clustered', 'accuracy'),
        ('current voltage', 'increase decrease'),
        ('current resistance', 'increase decrease smaller larger'),
        ('transformer turns', 'increase decrease more less'),
        ('electromagnet current', 'increase decrease stronger weaker'),
        ('electromagnet coils', 'increase decrease stronger weaker more less'),
        ('electromagnet winding', 'increase decrease stronger weaker more less'),
        ('coil turns', 'increase decrease stronger weaker more less'),
        ('coil current', 'increase decrease stronger weaker'),
    ):
        matches = [r for r in rules if terms(pair) <= terms(r[0])]
        assert matches, pair
        assert all(not (terms(forbidden) & (terms(p) | terms(n)))
                   for _, p, n, _ in matches), pair
    for cue, required, forbidden in (
        ('heat mass', 'energy temperature capacity', 'increase decrease larger smaller'),
        ('resistance unit', 'ohm', 'meter'),
        ('resistivity unit', 'ohm meter', 'ohmmeter'),
        ('conductance unit', 'siemens', 'inverse meter'),
        ('conductivity unit', 'siemens inverse meter', 'ohm'),
        ('resistivity material', 'composition temperature intrinsic', 'length area'),
        ('induction', 'changing flux voltage emf', 'current'),
        ('pendulum highest', 'potential maximum', 'kinetic minimum'),
        ('pendulum lowest', 'kinetic maximum', 'potential minimum'),
        ('light frequency energy', 'photon proportional', 'increase decrease'),
        ('light wavelength energy', 'photon inverse', 'increase decrease'),
    ):
        matches = [r for r in rules if terms(cue) == terms(r[0])]
        assert matches, cue
        assert all(terms(required) <= terms(p) and not (terms(forbidden) & terms(p))
                   for _, p, _, _ in matches), cue
    obsolete = {'constant speed straight', 'constant speed turning',
                'circular speed constant', 'force stationary', 'object rest force',
                'projectile horizontal vacuum', 'wave frequency increase',
                'wave wavelength increase', 'experiment larger sample',
                'experiment small sample', 'sample size increase',
                'graph straight rising', 'graph speed distance',
                'resistance large current', 'resistance small current',
                'transformer more turns'}
    assert not obsolete.intersection(r[0] for r in rules)

    selected = []
    for cue, (positive, negative) in CONDITIONS.items():
        qs = terms(cue)
        assert len(qs) == 2, (cue, qs)
        matches = [r for r in rules if terms(r[0]) == qs]
        assert matches, cue
        assert all(r[3] < 0 for r in matches), ('Condition leaked first-order', cue)
        assert any(terms(positive) <= terms(p) and terms(negative) <= terms(n)
                   for _, p, n, _ in matches), cue
        for _, p, n, _ in matches:
            assert not ((terms(p) - qs) & terms(negative)), cue
            assert not ((terms(n) - qs) & terms(positive)), cue
        selected.extend(matches)
    selected = sorted(set(selected))
    assert not audit_rules(vocab, aliases, selected)
    # Compile only these source conditions, retaining the full concept numbering.
    # The result cannot contain unconditional backoff from any selected fact.
    original = build.lexicon
    try:
        build.lexicon = lambda: (vocab, aliases, selected)
        w = build.build()
        assert w['A'].count_nonzero().item() == 0
        index = {word: i + 2 for i, word in enumerate(vocab)}

        def ids(text):
            return sorted({int(w['concept'][index[t]])
                           for t in build.TOKEN.findall(text) if t in index})

        wanted_rows = set()
        for cue, (positive, negative) in CONDITIONS.items():
            qs = ids(cue)
            row = (qs[0] * 1000003 + qs[1]) % w['row_ids'].shape[0]
            wanted_rows.add(row)
            actual = {int(k): float(v) for k, v in
                      zip(w['row_ids'][row], w['row_w'][row]) if int(k)}
            for text, sign in ((positive, 1), (negative, -1)):
                for target in set(ids(text)) - set(qs):
                    assert actual.get(target, 0) * sign > 0, (cue, text, actual)
        actual_rows = set(torch.where((w['row_w'] != 0).any(dim=1))[0].tolist())
        assert actual_rows == wanted_rows
        build.lexicon = lambda: (vocab, aliases, selected + selected)
        repeated = build.build()
        assert all(torch.equal(w[key], repeated[key]) for key in w)
        del w, repeated
    finally:
        build.lexicon = original
    gc.collect()
    counts = {}
    for path in sorted(root.glob('facts_g5*.txt')):
        counts[path.name] = sum(bool(line.split('#', 1)[0].strip())
                                for line in path.read_text().splitlines())
    print(f'G5 VERIFY OK: {len(CONDITIONS)} signed condition contracts; '
          'aliases, preserved distinctions, zero isolated backoff, duplicate invariance.')
    print(f'Authored rules={len(rules)}; G5 additions={sum(counts.values())}; files={counts}')


if __name__ == '__main__':
    main()
