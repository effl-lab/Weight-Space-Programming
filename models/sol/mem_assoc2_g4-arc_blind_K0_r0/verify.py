"""Label-free checks of representation, compilation, invariance, and resources.

This script has no science-question answers and does not measure accuracy.
Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -I /srv/work/model/verify.py
"""
from pathlib import Path
from collections import defaultdict
from itertools import islice
import hashlib
import json
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import arch
import build

torch.set_num_threads(4)
start = time.monotonic()
assert hashlib.sha256((ROOT/'arch.py').read_bytes()).hexdigest() == (
    '6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af')
G, M, F = build.source()
canonical = build.canonical_facts(M, F)
assert canonical == build.canonical_facts(M, list(reversed(F)))
assert canonical == build.canonical_facts(M, F + F)
vocab = (ROOT/'vocab.txt').read_text().splitlines()
assert vocab == sorted(M)
assert len(set(vocab)) == len(vocab)
assert all(build.TOKEN.fullmatch(word) for word in vocab)

# Shared contrast context must not silently acquire the positive sign.
# Abstract fixtures have no science questions or answer labels.
fixture = {'cue':1,'positive':2,'negative':3,'context':4,'not':5,
           'property':6,'function':7,'contextalias':4}
contrast = [(1.0,['cue'],['positive','context'],['negative','contextalias'])]
assert build.canonical_facts(fixture,contrast) == [(1.0,(1,),(2,),(3,))]
swapped = [(1.0,['cue'],['negative','contextalias'],['positive','context'])]
assert build.canonical_facts(fixture,swapped) == [(1.0,(1,),(3,),(2,))]
assert build.canonical_facts(fixture,[(1.,['cue'],['context'],['contextalias'])]) == []
fixture_A, fixture_pairs = build._associations(fixture,contrast,16)
assert fixture_A[1,4] == 0
assert fixture_A[1,2] > 0 and fixture_A[1,3] < 0
assert all(4 not in targets for targets in fixture_pairs.values())
try:
    build.canonical_facts(fixture,[(1.,['cue'],['positive'],[]),
                                   (1.,['cue'],[],['positive'])])
except ValueError:
    pass
else:
    raise AssertionError('True cross-record contradictions must still fail')
del fixture_A, fixture_pairs

# Direct checks of reviewed declarative claims, not answer predictions.
by_cues = defaultdict(lambda: (set(),set()))
for weight,src,positive,negative in canonical:
    by_cues[src][0].update(positive)
    by_cues[src][1].update(negative)
def claim(cues, positive=(), negative=(), absent=()):
    key = tuple(sorted({M[x] for x in cues.split()}))
    pos,neg = by_cues[key]
    assert {M[x] for x in positive}.issubset(pos), (cues,positive)
    assert {M[x] for x in negative}.issubset(neg), (cues,negative)
    assert not {M[x] for x in absent}.intersection(pos), (cues,absent)
claim('thermos vacuum',positive=['reduce'],negative=['conduction','convection'])
claim('vacuum insulation',negative=['conduction','convection'])
claim('switch open',positive=['zero'],negative=['current'])
claim('circuit open',positive=['zero'],negative=['current'])
claim('tendon',positive=['muscle','bone'],negative=['ligament'])
claim('ligament',positive=['bone'],negative=['muscle'])
claim('seed cooked',negative=['germination'])
claim('plant darkness',positive=['respiration'],negative=['photosynthesis'])
claim('ph below seven',positive=['acid'],negative=['base'])
claim('ph above seven',positive=['base'],negative=['acid'])
claim('fixed composition',positive=['pure','substance'],absent=['compound'])
claim('mineral heaviest',positive=['mass'],absent=['density'])
claim('magma slow cooling',positive=['coarse','crystal'])
claim('sound vacuum',positive=['silence'],negative=['transmission'],absent=['particle'])
claim('balloon squeeze',positive=['compression','decrease','shrink'])
claim('balloon heated',positive=['expansion','enlarge'])
claim('boat cargo',positive=['displacement','draft'],absent=['lower'])
claim('rail gap',negative=['buckling'])
claim('iron paint',positive=['barrier'],negative=['corrosion','rust'])
claim('fuse melted',negative=['current','overheating'])
claim('circuitbreaker overload',positive=['open'],negative=['current'])
claim('seawater drinkable',positive=['distillation','osmosis'])
assert M['filtration'] not in by_cues[tuple(sorted({M[x] for x in 'seawater drinkable'.split()}))][1]
claim('liquid surface molecules escape',positive=['evaporation','vaporization'])
claim('sunflower seedling light',positive=['phototropism'])
claim('iodine',positive=['element','halogen'],absent=['anion'])
claim('iodide',positive=['anion','negative'],absent=['element','metal'])
claim('iodide charge',positive=['negative','one'],negative=['positive','neutral'])
claim('ammonium',positive=['cation','positive','polyatomicion'])
for ion in ['hydroxide','nitrate','nitrite','sulfate','phosphate','carbonate','bicarbonate']:
    claim(ion,positive=['anion','negative','polyatomicion'],absent=['metal','element'])
for name,formula in [('ozone','o3'),('hydrogen peroxide','h2o2'),
                     ('calcium carbonate','caco3'),('sodium bicarbonate','nahco3'),
                     ('sulfur dioxide','so2'),('hydrochloric acid','hcl'),
                     ('sulfuric acid','h2so4'),('nitric acid','hno3'),
                     ('carbonic acid','h2co3'),('silicon dioxide','sio2'),
                     ('sodium hydroxide','naoh'),('potassium chloride','kcl'),
                     ('magnesium oxide','mgo'),('ethanol','c2h5oh'),
                     ('sodium chloride','nacl')]:
    claim(name+' formula',positive=[formula])
claim('o3',positive=['oxygen','element'],negative=['compound'])
claim('hydrogen peroxide decomposition',positive=['h2o','o2'],negative=['h2'])
claim('water distillation products',positive=['h2o'],negative=['h2','o2'])
claim('least',positive=['minimum'],negative=['maximum'])
claim('minimum meaning',positive=['least'],negative=['maximum'])
claim('overheat',positive=['temperature','heat'])
claim('overheat body',positive=['evaporation','cooling'],negative=['shivering'])
claim('energy pyramid top',positive=['least','available'],absent=['organism'])
claim('top predator energy',positive=['least'])
claim('terrestrial producer biomass',positive=['greatest'],negative=['least'])
claim('persistent pesticide top predator',positive=['concentration','highest'],negative=['least'])
claim('pendulum top energy',positive=['potential','greatest'],negative=['kinetic','speed'])
claim('pendulum bottom energy',positive=['kinetic','speed','greatest'],negative=['potential'])
claim('pendulum top kinetic',positive=['minimum','zero'],negative=['maximum'])
claim('pendulum top potential',positive=['maximum'],negative=['minimum'])
claim('pendulum bottom kinetic',positive=['maximum'],negative=['minimum'])
claim('pendulum bottom potential',positive=['minimum'],negative=['maximum'])
claim('energy pyramid shape',positive=['upright'],negative=['inverted'])
claim('aquatic biomass pyramid',positive=['inverted','phytoplankton'])
claim('experiment seed type',positive=['vary','water','light','soil'],absent=['species'])
claim('chemical property',positive=['reactivity'])
claim('react property',positive=['reactivity'])
claim('reactivity classification',positive=['chemical','property'],negative=['physical'])
claim('reflectivity definition',positive=['fraction','incident','reflect'])
claim('absorptivity definition',positive=['fraction','incident','absorb'])
claim('reflectivity unit',positive=['dimensionless','ratio'])
claim('absorptivity unit',positive=['dimensionless','ratio'])
claim('absorbency definition',positive=['capacity','liquid','uptake'])
claim('towel property',positive=['absorbency','capillary'])
claim('mirror property',positive=['reflectivity'])
claim('reflector telescope',positive=['mirror','reflect','focus'])
claim('refractor telescope',positive=['lens','refract','focus'])
claim('telescope objective mirror',positive=['reflector'])
claim('telescope objective lens',positive=['refractor'])
claim('opaque black visible',positive=['absorptivity','high'])
claim('shiny metal reflectivity visible',positive=['high'],negative=['low'])
claim('shiny metal absorptivity visible',positive=['low'],negative=['high'])
claim('graph velocity area',positive=['signed','displacement'],absent=['distance'])
claim('graph speed area',positive=['distance'])
claim('velocity time graph area',positive=['displacement'],absent=['distance'])
claim('speed time graph area',positive=['distance'])
claim('graph horizontal speed',positive=['constant'],absent=['velocity'])
claim('circular constant speed',positive=['acceleration','direction'])
claim('round trip displacement',positive=['zero'])
claim('round trip distance',positive=['path','length'])
claim('nucleus tiny dense',positive=['mass','proton','neutron'],negative=['electron','empty','space'])
claim('atom volume',positive=['cloud','empty','space'],negative=['nucleus'])
claim('mass different planet',positive=['same','invariant'])
assert M['gravity'] not in by_cues[tuple(sorted({M[x] for x in 'mass different planet'.split()}))][1]
claim('isolated momentum collision',positive=['conserved','constant'],negative=['destroy'])
claim('isolated elastic collision',positive=['momentum','kinetic','conserved'])
claim('isolated inelastic collision',positive=['momentum','conserved','thermal'])
claim('isolated elastic collision conservation',positive=['momentum','kinetic','energy'])
claim('isolated inelastic collision conservation',positive=['momentum','energy'],negative=['kinetic'])
claim('net external impulse',positive=['momentum','change'])
claim('impulse unit',positive=['newton','second'])
claim('nerve impulse',positive=['electrical','signal'])
claim('mechanical impulse',positive=['momentum'])
claim('pendulum top',positive=['potential','high'],negative=['kinetic','speed'])
claim('pendulum bottom',positive=['kinetic','speed','high'],negative=['potential'])

# These are semantic representation invariants, not multiple-choice answers.
for group in [('not','least','minimum'), ('hot','overheat'),
              ('react','reactivity'), ('reflect','reflectivity','reflector','refractor'),
              ('absorb','absorptivity','absorbency'),
              ('distance','displacement'), ('speed','velocity'),
              ('closed','isolated'), ('signal','impulse'),
              ('resistance','resistor','resist','resistant'),
              ('two','double'), ('half','halve'), ('left','leave'),
              ('voltage','volt'), ('sun','sunlight'), ('ion','ionization'),
              ('sodium','sodiumion'), ('chlorine','chloride'), ('iodine','iodide'),
              ('ozone','o3'), ('peroxide','h2o2'), ('carbonate','caco3'),
              ('water','h2o'), ('iron','fe'), ('conduction','conductance','conductivity')]:
    assert len({M[x] for x in group}) == len(group), group
for group in [('squeeze','squeezed'), ('double','doubled'), ('half','halves'),
              ('resist','resists'), ('resistant','resistanceability'),
              ('left','leftward'), ('swave','swaves'), ('villus','villi'),
              ('iodide','iodides'), ('enlarge','enlarged'),
              ('signal','signals'), ('impulse','impulses'),
              ('overheat','overheated','overheating'),
              ('react','reaction'), ('reactivity','reactive'),
              ('reflect','reflection'), ('absorb','absorption'),
              ('absorbency','absorbent'), ('reflector','reflectors'),
              ('sulfuric','sulphuric'), ('bicarbonate','hydrogencarbonate')]:
    assert len({M[x] for x in group}) == 1, group

w = build.build()
built_at = time.monotonic()
C = len(G) + 1
H, R = w['row_ids'].shape
assert set(w) == {'concept','idf','A','row_ids','row_w','mix'}
assert w['concept'].shape == (len(vocab)+2,)
assert w['idf'].shape == (C,)
assert w['A'].shape == (C,C)
assert w['row_w'].shape == (H,R)
assert w['mix'].shape == ()
assert C >= 16 and H == 1414217 and R <= 64
assert w['concept'].dtype == w['row_ids'].dtype == torch.int64
for k in ('idf','A','row_w','mix'):
    assert w[k].dtype == torch.float32, k
assert w['concept'][:2].tolist() == [0,0]
assert w['idf'][0].item() == 0
assert w['concept'][2:].tolist() == [M[x] for x in vocab]
for k in ('concept','row_ids'):
    t = w[k]
    for chunk in t.split(4096):
        assert chunk.min().item() >= 0 and chunk.max().item() < C
for k in ('idf','A','row_w'):
    for chunk in w[k].split(4096):
        assert torch.isfinite(chunk).all()
assert torch.isfinite(w['mix'])
for ids, values in zip(w['row_ids'].split(4096), w['row_w'].split(4096)):
    assert (values[ids == 0] == 0).all()

# Independently reconstruct every occupied hashed row and every first-order
# coefficient from the text declarations, without asking science questions.
# A separate reference implementation follows the declared formulas rather
# than calling the model's association compiler to validate its own result.
atomic = defaultdict(dict)
for weight,src,pos,neg in canonical:
    positive_scale = weight / max(1.,len(pos)**0.40)
    negative_scale = -0.78*weight / max(1.,len(neg)**0.40)
    for target,value in [(x,positive_scale) for x in pos]+[(x,negative_scale) for x in neg]:
        if abs(value) > abs(atomic[src].get(target,0.)):
            atomic[src][target] = value
ref_A = np.zeros((C,C),dtype=np.float32)
pairs = defaultdict(dict)
profiles = defaultdict(dict)
for src,targets in sorted(atomic.items()):
    first_scale = 1. if len(src)==1 else .16/len(src)
    for cue in src:
        for target,value in sorted(targets.items()):
            ref_A[cue,target] += first_scale*value
            if len(src)==1:
                profiles[cue][target] = profiles[cue].get(target,0.)+value
    if len(src)>1:
        scale = 1./max(1.,(len(src)*(len(src)-1)/2)**.45)
        for i,left in enumerate(src):
            for right in src[i+1:]:
                row = pairs[left,right]
                for target,value in sorted(targets.items()):
                    row[target] = row.get(target,0.)+value*scale
for anchor,profile in sorted(profiles.items()):
    for focus in sorted({M['property'],M['function']}):
        if anchor in (focus,M['not']):
            continue
        row = pairs[tuple(sorted((anchor,focus)))]
        for target,value in sorted(profile.items()):
            if target not in (anchor,focus):
                row[target] = row.get(target,0.)+.70*3.*np.tanh(value/3.)
ref_A = 3.*np.tanh(ref_A/3.)
for cue in range(1,C):
    if cue==M['not']:
        continue
    active = np.flatnonzero(ref_A[cue])
    if not len(active):
        continue
    row = pairs[tuple(sorted((cue,M['not'])))]
    for target in active:
        if target != M['not']:
            row[int(target)] = row.get(int(target),0.)-1.25*float(ref_A[cue,target])
for i in range(0,C,256):
    np.testing.assert_array_equal(w['A'][i:i+256].numpy(), ref_A[i:i+256])
del ref_A,atomic,profiles
expected = defaultdict(dict)
for (a,b), targets in pairs.items():
    row = expected[(a*1000003+b)%H]
    for t,val in targets.items():
        row[t] = row.get(t,0.)+val
truncated = 0
for h, targets in expected.items():
    nonzero = [(t,val) for t,val in targets.items() if abs(val)>1e-10]
    truncated += len(nonzero)>R
    entries = sorted(nonzero,key=lambda x:(-abs(x[1]),x[0]))[:R]
    ri = torch.zeros(R,dtype=torch.int64)
    rw = torch.zeros(R,dtype=torch.float32)
    for j,(target,value) in enumerate(entries):
        ri[j] = target
        rw[j] = 5.0*np.tanh(value/5.0)
    assert torch.equal(w['row_ids'][h],ri)
    assert torch.equal(w['row_w'][h],rw)
occupied = set(expected)
for offset in range(0,H,4096):
    inds = [i-offset for i in range(offset,min(offset+4096,H)) if i not in occupied]
    if inds:
        assert w['row_ids'][offset:offset+4096][inds].count_nonzero() == 0
        assert w['row_w'][offset:offset+4096][inds].count_nonzero() == 0
pair_count = len(pairs)
del pairs, expected

# Confirm option preservation and synonym exclusion at bag level.
ix = {word:i+2 for i,word in enumerate(vocab)}
for qword, oword in [('two','double'),('half','halve'),('left','leave'),
                     ('resistance','resistor'),('resistance','resistant'),
                     ('voltage','volt'),('water','h2o'),('sun','sunlight'),
                     ('iodine','iodide'),('ozone','o3'),('peroxide','h2o2'),
                     ('carbonate','caco3'),('not','least'),('hot','overheat'),
                     ('react','reactivity'),('reflect','reflectivity'),
                     ('reflect','reflector'),('absorb','absorptivity'),
                     ('absorb','absorbency'),('distance','displacement'),
                     ('speed','velocity'),('closed','isolated'),('signal','impulse')]:
    q = torch.zeros((1,128),dtype=torch.int64)
    o = torch.zeros((1,4,48),dtype=torch.int64)
    q[0,0] = ix[qword]
    o[0,0,0] = ix[oword]
    o[0,1,0] = ix[qword]
    qb,ob = arch.bags(w,q,o)
    assert ob[0,0,M[oword]] > 0
    assert ob[0,1,M[qword]] == 0

rng = np.random.default_rng(71029)
q = torch.from_numpy(rng.integers(0,len(vocab)+2,size=(8,128),dtype=np.int64))
o = torch.from_numpy(rng.integers(0,len(vocab)+2,size=(8,4,48),dtype=np.int64))
# The bags have deliberately varying lengths; these are unlabeled token bags,
# not ARC or authored question/answer examples.
for i,length in enumerate([0,1,4,8,16,24,64,128]):
    q[i,length:] = 0
with torch.no_grad():
    base = arch.forward(w,q,o)
    assert torch.isfinite(base).all()
    perm = torch.tensor([2,0,3,1])
    torch.testing.assert_close(arch.forward(w,q,o[:,perm]),base[:,perm])
    torch.testing.assert_close(arch.forward(w,q.flip(1),o.flip(2)),base)
    torch.testing.assert_close(arch.forward(w,q[:1],o[:1]),base[:1])
    torch.testing.assert_close(arch.forward(w,q[1:2],o[1:2]),base[1:2])
    duplicated_q = q[:2].clone()
    duplicated_q[:,64:] = duplicated_q[:,:64]
    torch.testing.assert_close(arch.forward(w,duplicated_q,o[:2]),base[:2])
    identical_options = o[:1,:1].expand(-1,4,-1)
    tied = arch.forward(w,q[4:5],identical_options)
    assert torch.equal(tied,tied[:,:1].expand(-1,4))
    empty = torch.zeros((1,128),dtype=torch.int64)
    unknown = torch.ones((1,128),dtype=torch.int64)
    assert arch.forward(w,empty,o[:1]).count_nonzero() == 0
    assert arch.forward(w,unknown,o[:1]).count_nonzero() == 0

    inference_start = time.monotonic()
    checksum = 0.0
    for offset in range(0,3530,64):
        n = min(64,3530-offset)
        tq = torch.from_numpy(rng.integers(0,len(vocab)+2,size=(n,128),dtype=np.int64))
        to = torch.from_numpy(rng.integers(0,len(vocab)+2,size=(n,4,48),dtype=np.int64))
        for i in range(n):
            tq[i,8+(i%45):] = 0
        logits = arch.forward(w,tq,to)
        assert logits.shape == (n,4) and torch.isfinite(logits).all()
        checksum += float(logits.sum())
    inference_seconds = time.monotonic()-inference_start

nonzero = sum(int(t.count_nonzero()) for t in w.values())
peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
elapsed = time.monotonic()-start
assert peak_bytes < 2_000_000_000, peak_bytes
assert elapsed < 1200, elapsed
sources = ['facts.txt','earth.txt','life.txt','inquiry.txt','supplement.txt',
           'functions.txt','relations.txt','mechanisms.txt','scenarios.txt',
           'definitions.txt','ecology_extra.txt','earth_extra.txt',
           'physical_extra.txt','facets.txt','context_physics.txt',
           'context_earth.txt','context_life.txt','diagnostic_pairs.txt',
           'context_inquiry.txt','representation.txt']
def records(name):
    return sum(bool(x.split('#',1)[0].strip()) for x in (ROOT/name).read_text().splitlines())
report = dict(status='LABEL-FREE VERIFICATION PASSED',
              architecture_sha256=hashlib.sha256((ROOT/'arch.py').read_bytes()).hexdigest(),
              vocabulary=len(vocab),concepts=C,authored_records=sum(map(records,sources)),
              added_records=sum(map(records,sources[-6:])),
              expanded_records=len(F),canonical_records=len(canonical),
              pair_keys=pair_count,occupied_rows=len(occupied),truncated_rows=truncated,
              shapes={k:list(t.shape) for k,t in w.items()},
              weights=sum(t.numel() for t in w.values()),nonzero=nonzero,
              tensor_bytes=sum(t.numel()*t.element_size() for t in w.values()),
              build_and_source_seconds=built_at-start,
              synthetic_3530_inference_seconds=inference_seconds,
              total_seconds=elapsed,peak_rss_bytes=peak_bytes,
              accuracy_measured=False)
(ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
