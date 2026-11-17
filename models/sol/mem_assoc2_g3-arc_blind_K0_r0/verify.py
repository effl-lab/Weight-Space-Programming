"""Label-free structural, knowledge-layout, and resource checks.

No ARC inputs, answer labels, accuracy calculations, optimization, or fitting.
This is independent of the supplied checker; run that checker separately.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import build
import arch

torch.set_num_threads(4)
start = time.monotonic()
assert hashlib.sha256((ROOT/'arch.py').read_bytes()).hexdigest() == '6945de3b3571d5cb7ee29e2a4c00163c77205a72530dacd4f97023d6185441af'

groups, mapping, facts = build.source()
canonical = build.canonical_facts(mapping, facts)
assert canonical == build.canonical_facts(mapping, list(reversed(facts)))
assert canonical == build.canonical_facts(mapping, facts + facts)
for a,b in [('atom','molecule'), ('proton','electron'), ('neutron','proton'),
            ('chlorine','chloride'), ('carbon','dioxide'), ('cell','molecule'),
            ('sediment','sedimentary'), ('mix','mixture'), ('dissolve','solution'),
            ('endocrine','hormone'), ('boiling','vaporization'),
            ('phosphorus','phosphate'), ('balance','balanced'),
            ('condensation','condensate'), ('amphibian','amphibious'),
            ('sun','sunlight'), ('star','starlight'), ('moon','moonlight'),
            ('heatcapacity','specificheat'), ('amplitude','waveheight'),
            ('sodium','sodiumion'), ('calcium','calciumion'),
            ('ion','ionization'), ('voltage','volt'),
            ('conduct','conduction'), ('conduct','conductance'),
            ('conduct','conductivity'), ('conduction','conductance'),
            ('conduction','conductivity'), ('conductance','conductivity')]:
    assert mapping[a] != mapping[b], (a,b)
for a,b in [('bacterium','bacteria'), ('hypothesis','hypotheses'), ('larva','larvae'),
            ('germinate','germinating'), ('erode','erosion'), ('leaf',"leaf's"),
            ('microorganism','microbes'), ('rotate','rotation'),
            ('balanced','balancing'), ('phosphate','phosphates'),
            ('condensate','condensates'), ('sunlight','sunshine'),
            ('starlight','starlights'), ('moonlight','moonlights'),
            ('waveheight','waveheights'), ('sodiumion','sodiumions'),
            ('calciumion','calciumions'), ('ionization','ionized'),
            ('ionization','ionisation'), ('ionize','ionising'),
            ('volt','volts'), ('conduct','conducting'),
            ('resistivity','resistivities')]:
    assert mapping[a] == mapping[b], (a,b)

# Explicit nomenclature pairs: representation facets, not different substances.
SYMBOLS = [
    ('copper','cu'), ('iron','fe'), ('aluminum','al'), ('silver','ag'),
    ('sodium','na'), ('chlorine','cl'), ('calcium','ca'), ('silicon','si'),
    ('lead','pb'), ('mercury','hg'), ('zinc','zn'), ('gold','au'),
    ('platinum','pt'), ('bromine','br'), ('lithium','li'), ('strontium','sr'),
    ('barium','ba'), ('rubidium','rb'), ('krypton','kr'), ('xenon','xe'),
    ('selenium','se'),
]
FORMULAS = [
    ('water','h2o'), ('hydrogen','h2'), ('oxygen','o2'), ('nitrogen','n2'),
    ('dioxide','co2'), ('monoxide','co'), ('methane','ch4'), ('ammonia','nh3'),
    ('salt','nacl'), ('glucose','c6h12o6'),
]
for name,identifier in SYMBOLS + FORMULAS:
    assert mapping[name] != mapping[identifier], (name,identifier)
assert mapping['carbondioxide'] == mapping['dioxide']

# Scientific source regressions: inspect explicit claims, not answer accuracy.
def claim(cues):
    key = tuple(sorted({mapping[word] for word in cues.split()}))
    yes, no = set(), set()
    for _, src, pos, neg in canonical:
        if src == key:
            yes.update(pos)
            no.update(neg)
    return yes, no

for cues, supported, opposed, unasserted in [
    ('diffusion', 'passive motion', 'atp', 'energy'),
    ('isotonic', 'equilibrium volume', 'shrink swell', 'flow'),
    ('mutation', 'dna variation', '', 'beneficial'),
    ('nocturnal', 'active night', 'day', ''),
    ('population growth', 'birth immigration', 'death emigration', ''),
    ('net force zero', 'constant velocity', 'acceleration', ''),
    ('thrown ball highest', 'gravity acceleration', '', ''),
    ('ball highest point', 'gravity acceleration', '', 'zero'),
    ('terminal speed', 'gravity balanced', 'acceleration', ''),
    ('plant dark photosynthesis', 'stops decrease', 'light increase', ''),
    ('vacuum insulation', 'conduction convection reduced', '', 'particles radiation'),
    ('vacuum thermos', 'reduce conduction convection', '', 'radiation'),
    ('vacuum bottle', 'conduction convection reduced', '', 'radiation'),
    ('moon hammer feather fall', 'vacuum acceleration', 'air resistance', ''),
    ('phosphate composition', 'phosphorus oxygen ion', 'element', ''),
    ('sun', 'star fusion sunlight', 'planet', ''),
    ('star', 'fusion starlight', '', ''),
    ('moon', 'satellite reflected light moonlight', 'fusion', ''),
    ('moon brightness', 'reflected sunlight light', 'fusion', ''),
    ('sunlight origin', 'sun radiation', '', ''),
    ('starlight origin', 'star radiation', '', ''),
    ('moonlight origin', 'reflected sunlight', 'fusion', ''),
    ('sunlight', 'electromagnetic wave vacuum', '', 'star fusion hydrogen helium'),
    ('starlight', 'electromagnetic wave vacuum', '', 'star fusion hydrogen helium'),
    ('moonlight', 'electromagnetic wave vacuum', '', 'satellite crater phase'),
    ('heatcapacity property', 'extensive', 'intensive', ''),
    ('specificheat property', 'intensive', 'extensive', ''),
    ('heatcapacity formula', 'mass multiply specificheat', '', ''),
    ('waveheight definition', 'crest trough distance', '', ''),
    ('waveheight formula', 'two multiply amplitude', '', ''),
    ('amplitude definition', 'maximum displacement equilibrium', '', ''),
    ('sodiumion', 'cation ion positive charged', 'negative neutral', 'metal conductor shiny reactive'),
    ('calciumion', 'cation ion positive charged', 'negative neutral', 'metal conductor shiny'),
    ('chloride', 'anion ion negative', 'positive neutral', ''),
    ('ionization', 'electron transfer charged ion', '', ''),
    ('ionization identity', 'element proton atomicnumber conserved', '', ''),
    ('sodiumion charge', 'positive one', 'negative neutral', ''),
    ('calciumion charge', 'positive two', 'negative neutral', ''),
    ('chloride charge', 'negative one', 'positive neutral', ''),
    ('sodiumion electron count', 'ten 10', 'eleven 11', ''),
    ('sodiumion proton count', 'eleven 11', 'ten 10', ''),
    ('calciumion electron count', 'eighteen 18', 'twenty 20', ''),
    ('calciumion proton count', 'twenty 20', 'eighteen 18', ''),
    ('chloride electron count', 'eighteen 18', 'seventeen 17', ''),
    ('chloride proton count', 'seventeen 17', 'eighteen 18', ''),
    ('safety chemical', 'goggles gloves instructions', 'taste', ''),
    ('unknown plant', 'caution identify', 'taste touch eat', ''),
    ('field investigation', 'observe habitat record', 'destroy', 'disturbance'),
    ('mirror see', 'reflected light image', 'emitted', ''),
    ('voltage unit', 'volt', 'ampere', ''),
    ('voltage', 'volt electric potential', '', ''),
    ('conductance unit', 'siemens', 'ohm', ''),
    ('conductance resistance', 'inverse', 'direct', ''),
    ('conduction', 'contact particle', 'vacuum', 'siemens'),
    ('conduction definition', 'contact heat particle', 'vacuum', ''),
    ('conductivity property', 'intrinsic intensive', 'extensive', ''),
    ('reciprocal resistance', 'conductance', 'conductivity', ''),
    ('reciprocal resistivity', 'conductivity', 'conductance', ''),
    ('magnet cut', 'two north south pole', 'monopole', ''),
    ('magnet divided', 'two north south pole', 'monopole', ''),
    ('magnet fragment', 'two north south pole', 'monopole', ''),
    ('condensation cause', 'cooling vapor liquid', 'evaporation', 'heating'),
    ('melting cause', 'heating solid liquid', 'freezing', 'cooling'),
    ('freezing cause', 'cooling liquid solid', 'melting', 'heating'),
    ('h2o', 'water compound molecule hydrogen oxygen', 'element', ''),
    ('h2', 'hydrogen element molecule diatomic gas', 'compound', ''),
    ('o2', 'oxygen element molecule diatomic gas respiration', 'compound', ''),
    ('n2', 'nitrogen element molecule diatomic gas atmosphere', 'compound', ''),
    ('co2', 'carbon dioxide compound molecule greenhouse', 'element', ''),
    ('co', 'carbon monoxide compound molecule toxic', 'element', ''),
    ('ch4', 'methane compound molecule carbon hydrogen', 'element', ''),
    ('nh3', 'ammonia compound molecule nitrogen hydrogen base', 'element', ''),
    ('nacl', 'salt compound ionic sodium chloride', 'element', 'metal'),
    ('c6h12o6', 'glucose compound carbohydrate sugar', 'element', ''),
]:
    yes, no = claim(cues)
    assert {mapping[x] for x in supported.split()} <= yes, cues
    assert {mapping[x] for x in opposed.split()} <= no, cues
    assert not {mapping[x] for x in unasserted.split()} & (yes | no), cues
assert mapping['element'] not in claim('phosphate')[0]
for name,identifier in SYMBOLS:
    assert mapping[identifier] in claim(name+' symbol')[0], (name,identifier)
    assert mapping[name] in claim(identifier)[0], (name,identifier)
    assert mapping['element'] in claim(identifier)[0], identifier
for cues,identifier in [
    ('water formula','h2o'), ('hydrogen gas formula','h2'),
    ('oxygen gas formula','o2'), ('nitrogen gas formula','n2'),
    ('carbon dioxide formula','co2'), ('carbon monoxide formula','co'),
    ('methane formula','ch4'), ('ammonia formula','nh3'),
    ('salt formula','nacl'), ('glucose formula','c6h12o6'),
]:
    assert mapping[identifier] in claim(cues)[0], (cues,identifier)
for name,identifier in FORMULAS:
    assert mapping[name] in claim(identifier)[0], (name,identifier)

# Atomic idempotence across differently bundled repetitions.
mini = {'anchor':1, 'a':2, 'b':3, 'not':4, 'property':5, 'function':6}
base = [(2., ['anchor'], ['a'], [])]
with_restatement = base + [(1., ['anchor'], ['a','b'], [])]
x,p = build._associations(mini,base,16)
y,r = build._associations(mini,with_restatement,16)
assert x[1,2] == y[1,2]
assert p[(1,5)][2] == r[(1,5)][2]
assert p[(1,4)][2] == r[(1,4)][2]
try:
    build.canonical_facts(mini, base + [(2.,['anchor'],[],['a'])])
except ValueError:
    pass
else:
    raise AssertionError('Contradictory atomic claims must be rejected')

vocab = (ROOT/'vocab.txt').read_text().splitlines()
assert vocab == sorted(mapping) and len(set(vocab)) == len(vocab)
assert all(re.fullmatch(r"[a-z0-9']+",word) for word in vocab)
C = max(16,len(groups)+1)
ref_A, pairs = build._associations(mapping,facts,C)
reference_rows = defaultdict(dict)
H, R = 1414217,64
for (a,b),targets in pairs.items():
    row = reference_rows[(a*1000003+b)%H]
    for target,value in targets.items():
        row[target] = row.get(target,0.) + value

w = build.build()
build_seconds = time.monotonic()-start
assert set(w) == {'concept','idf','A','row_ids','row_w','mix'}
shapes = {'concept':(len(vocab)+2,), 'idf':(C,), 'A':(C,C),
          'row_ids':(H,R), 'row_w':(H,R), 'mix':()}
for name,tensor in w.items():
    assert isinstance(tensor,torch.Tensor)
    assert tuple(tensor.shape) == shapes[name], (name,tensor.shape)
    assert tensor.dtype == (torch.int64 if name in ('concept','row_ids') else torch.float32)
    assert tensor.device.type == 'cpu' and not tensor.requires_grad
    flat = tensor.reshape(-1)
    for at in range(0,flat.numel(),1048576):
        chunk = flat[at:at+1048576]
        if tensor.dtype == torch.float32:
            assert bool(torch.isfinite(chunk).all()),name
        else:
            assert int(chunk.min()) >= 0 and int(chunk.max()) < C,name
    if name == 'row_w':
        # Every unused slot has both zero ID and zero weight, and vice versa.
        ids = w['row_ids'].reshape(-1)
        for at in range(0,flat.numel(),1048576):
            assert torch.equal(flat[at:at+1048576]==0,ids[at:at+1048576]==0)
assert w['concept'][:2].tolist() == [0,0]
assert w['idf'][0] == 0 and bool((w['idf']>=0).all())
assert bool((w['A'][0]==0).all()) and bool((w['A'][:,0]==0).all())
assert np.array_equal(ref_A,w['A'].numpy())
expected_nonzero = 0
for h,targets in reference_rows.items():
    entries = sorted(((t,v) for t,v in targets.items() if abs(v)>1e-10),
                     key=lambda item:(-abs(item[1]),item[0]))[:R]
    ids = np.zeros(R,dtype=np.int64)
    values = np.zeros(R,dtype=np.float32)
    for j,(t,value) in enumerate(entries):
        ids[j]=t; values[j]=5.*np.tanh(value/5.)
    assert np.array_equal(w['row_ids'][h].numpy(),ids)
    assert np.array_equal(w['row_w'][h].numpy(),values)
    expected_nonzero += len(entries)
assert int(w['row_ids'].count_nonzero()) == expected_nonzero
assert int(w['row_w'].count_nonzero()) == expected_nonzero

def encode(text,length):
    index = {word:i+2 for i,word in enumerate(vocab)}
    ids = [index.get(word,1) for word in re.findall(r"[a-z0-9']+",text.lower())][:length]
    return torch.tensor(ids+[0]*(length-len(ids)),dtype=torch.int64)

# Structural tests of the question/answer exclusion mechanism, not answers.
q = encode('mix dissolve',128).reshape(1,-1)
o = torch.stack([encode(text,48) for text in ['mixture','solution','atom','molecule']]).reshape(1,4,48)
qb,ob = arch.bags(w,q,o)
assert ob[0,0,mapping['mixture']] > 0
assert ob[0,1,mapping['solution']] > 0
assert qb[0,mapping['mix']] > 0 and qb[0,mapping['dissolve']] > 0
q = encode('phosphorus balance condensation amphibian',128).reshape(1,-1)
o = torch.stack([encode(text,48) for text in
                 ['phosphate','balanced','condensate','amphibious']]).reshape(1,4,48)
qb,ob = arch.bags(w,q,o)
for k,word in enumerate(['phosphate','balanced','condensate','amphibious']):
    assert ob[0,k,mapping[word]] > 0, word
# Radiation words must survive a question mentioning their source objects.
for sources,targets in [
    ('sun star moon heatcapacity', ['sunlight','starlight','moonlight','specificheat']),
    ('sunlight starlight moonlight amplitude', ['sun','star','moon','waveheight']),
    ('ion sodium calcium chlorine', ['ionization','sodiumion','calciumion','chloride']),
    ('conduct conduction conductance voltage', ['conductivity','contact','siemens','volt']),
]:
    q = encode(sources,128).reshape(1,-1)
    o = torch.stack([encode(text,48) for text in targets]).reshape(1,4,48)
    qb,ob = arch.bags(w,q,o)
    for k,word in enumerate(targets):
        assert ob[0,k,mapping[word]] > 0, word

# A quantity must never erase its distinct unit from an option. These are bag
# invariants only: no logits, answer choices, predictions, or accuracy are scored.
for quantity,unit in [
    ('voltage','volt'), ('current','ampere'), ('resistance','ohm'),
    ('conductance','siemens'), ('energy','joule'), ('power','watt'),
    ('force','newton'), ('mass','kilogram'), ('volume','liter'),
    ('temperature','kelvin'), ('frequency','hertz'), ('time','second'),
    ('charge','coulomb'), ('pressure','pascal'), ('length','meter'),
]:
    assert mapping[quantity] != mapping[unit], (quantity,unit)
    q = encode(quantity,128).reshape(1,-1)
    o = encode(unit,48).reshape(1,1,48).expand(-1,4,-1)
    _,ob = arch.bags(w,q,o)
    assert ob[0,0,mapping[unit]] > 0, (quantity,unit)

# Names and written identifiers must survive exclusion in both directions.
# Upper-case spellings also exercise the task's lower-casing tokenizer.
for name,identifier in SYMBOLS + FORMULAS:
    for source,target in [(name,identifier.upper()), (identifier.upper(),name)]:
        q = encode(source,128).reshape(1,-1)
        o = encode(target,48).reshape(1,1,48).expand(-1,4,-1)
        _,ob = arch.bags(w,q,o)
        concept = mapping[target.lower()]
        assert ob[0,0,concept] > 0, (source,target)

# Arbitrary synthetic inputs: no science questions or answer labels are used.
generator = torch.Generator().manual_seed(72831)
active = torch.nonzero(w['idf'][w['concept']]>0).reshape(-1)
q = torch.zeros(64,128,dtype=torch.int64)
o = torch.zeros(64,4,48,dtype=torch.int64)
q[:,:24] = active[torch.randint(len(active),(64,24),generator=generator)]
o[:,:,:12] = active[torch.randint(len(active),(64,4,12),generator=generator)]
q[:,24]=1; o[:,:,12]=1
with torch.no_grad():
    logits = arch.forward(w,q,o)
    assert logits.shape == (64,4) and bool(torch.isfinite(logits).all())
    chunks = torch.cat([arch.forward(w,q[i:i+8],o[i:i+8]) for i in range(0,64,8)])
    torch.testing.assert_close(logits,chunks,atol=1e-5,rtol=1e-5)
    perm=torch.tensor([2,0,3,1])
    torch.testing.assert_close(arch.forward(w,q,o[:,perm]),logits[:,perm],atol=1e-5,rtol=1e-5)
    # An option's absolute position is never a feature.
    qi=torch.randperm(128,generator=generator); oi=torch.randperm(48,generator=generator)
    torch.testing.assert_close(arch.forward(w,q[:,qi],o[:,:,oi]),logits,atol=1e-5,rtol=1e-5)
    qdupe=q.clone(); odupe=o.clone(); qdupe[:,30:54]=q[:,:24]; odupe[:,:,20:32]=o[:,:,:12]
    torch.testing.assert_close(arch.forward(w,qdupe,odupe),logits,atol=1e-5,rtol=1e-5)
    empty=torch.zeros_like(q)
    assert bool((arch.forward(w,empty,o)==0).all())
    unknown=torch.ones_like(q)
    assert bool((arch.forward(w,unknown,o)==0).all())
    repeated=o[:,0:1,:].expand(-1,4,-1).contiguous()
    tied=arch.forward(w,q,repeated)
    assert torch.equal(tied,tied[:,0:1].expand(-1,4))
    begin=time.monotonic()
    processed=0
    while processed < 3530:
        n=min(64,3530-processed)
        out=arch.forward(w,q[:n],o[:n])
        assert bool(torch.isfinite(out).all())
        processed+=n
    inference_seconds=time.monotonic()-begin

stats = {
    'architecture_sha256':hashlib.sha256((ROOT/'arch.py').read_bytes()).hexdigest(),
    'concepts':C, 'vocabulary':len(vocab), 'expanded_records':len(facts),
    'canonical_records':len(canonical), 'pair_keys':len(pairs),
    'occupied_rows':len(reference_rows), 'row_width':R,
    'weights':sum(t.numel() for t in w.values()),
    'nonzero':sum(int(t.count_nonzero()) for t in w.values()),
    'tensor_bytes':sum(t.numel()*t.element_size() for t in w.values()),
    'build_and_source_reference_seconds':build_seconds,
    'synthetic_3530_inference_seconds':inference_seconds,
    'total_seconds':time.monotonic()-start,
    'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    'official_checker':'Run separately with /srv/work/tools/check_model.py; latest outcome documented in NOTES.md',
    'accuracy_evaluation':False,
}
assert stats['peak_rss_bytes'] < 2_000_000_000, stats
assert stats['total_seconds'] < 1200,stats
(ROOT/'verification.json').write_text(json.dumps(stats,indent=2)+'\n')
print('INDEPENDENT CHECK OK')
print(json.dumps(stats,indent=2))
