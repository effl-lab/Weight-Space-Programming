# A hand-written ARC model (generation 5)

Nothing here is trained or fitted, and nothing is pretrained. All knowledge is in plain-text files I wrote.
`build.py` turns them into tensors with fixed formulas, and `arch.py` is a fixed tensor program. I started from
the parent's folder (generation 4, 53.7% on validation). Its structure is unchanged:

- **Retrieval core.** An option scores high when a fact shares idf-weighted concepts (Porter stems plus
  synonyms) with both the question and the option. The fact rows come in four kinds: whole facts, 4-concept
  windows, single clauses, and facts with their neighbours. There are two routes (the option's own words, and
  its isa.txt categories) and two passes (the second adds the words of the question's best facts). The score
  is divided by its maximum over the four options, and PMI association is added at weight 0.2.
- **Reasoning modules**, each a small explicit rule set on top of retrieval: qualitative reasoning (`qr.txt`:
  quantities, direction words, cause→effect signs), order and extremes (`seq.txt`), arithmetic, experiment
  variables, food webs, and word order for options made of the same words.
- **Negated questions** ("NOT", "EXCEPT", now also "avoid") flip the sign of the score.

The docstrings in the code describe these parts. This file describes what generation 5 changed, and why.

## How I worked

The parent left no dev questions, so I wrote 490 ARC-style questions of my own in 15 rounds
(`/srv/work/dev/b1..b15.txt`, outside the model folder). They cover mixed topics, everyday paraphrase,
sentence-length options, negated questions, technology, direction questions and food webs. Like the parent's
writer, I found my questions are much easier than ARC: the parent scores 78% on them against 53.7% on ARC.
Both the facts and the questions come from the same writer, so I cannot reproduce ARC's vocabulary mismatch.
So I looked for **mechanisms that are wrong in general**, not for facts a question lacked. A dev tool
showed, for each wrong pick, the top facts per option and each module's share of the score. Every change below
is a general rule. Adding a preamble of irrelevant context to every question changed accuracy by about 1%, so
the model is robust to that.

## Changes

### Bugs in the reasoning modules (probably the most valuable on real ARC)
- **Extremes fired on "most likely".** The order-and-extremes rule (weight 2) ranks options that are items of a
  sequence whenever the question has one of the sequence's end words. Several sequences use generic end words:
  `particle_motion` ("most", "least"), `density_state` and `earth_layers` ("least"), `planet_size` ("most").
  So "Which of these resources is **most** important for farming?" pushed "gas" up, and "Which is most likely
  found in lemonade on a hot day?" pushed water over ice, by about ±1.2. Now an end word followed by "likely" or
  "probably" does not count. A sequence can also name **topic words** (a new fourth field in `seq.txt`), and it
  ranks only when the question has one of them. Topics were added to the generic sequences, and `sound_speed`
  got one too. It used to cancel `particle_motion`: it lists gas/liquid/solid in the opposite order with the
  same "fastest", so "In which state do particles move fastest?" got nothing.
- **Experiment variables fired outside experiments.** "change" is both a trigger and a cue word of that rule,
  so "Why do leaves change color in the fall?" gave +0.7 to wrong options. The rule now needs an
  investigation word in the question (experiment, test, measure, variable, student, data, …).
- **Food web.** (a) A direction verb reached organisms across a sentence: in "snakes eat mice. If the snakes
  die out", the mice were marked as dying, so they got no effect. Now a verb reaches an organism only when no
  other organism of the web stands between them. (b) When the question names at least two known organisms,
  only organisms are linked ("Hawks in the field eat rabbits": the hawks eat, not the field). Organisms come
  from the animal and plant lines of `isa.txt` plus a short list in `build.py`, and a link may now span 5 words
  before and 4 after the verb. (c) "fewer seeds grow", "more rabbits die": a comparative before an organism
  and a verb after it combine by sign (not motion verbs: "less rain falls" is less rain). (d) Options naming no
  organism ("They will increase.") speak about the organism the question asks about. My food-web set went
  from 3/10 (parent) to 8/10.
- **Qualitative reasoning.** "more trees are **cut** down" is fewer trees: a comparative plus a passive verb
  combine by sign. "moves **from** a sunny window **to** a dark corner": between "from" and a following "to" is
  the old state, so words with their own direction count reversed there. An event word that is the agent of a
  harm verb ("a disease kills most of the rabbits") keeps its own increase, and the verb reaches its object.
  "causes X" names an increase of X. The pronoun-option rule ("it is lower …") no longer applies when the
  option's direction already belongs to a quantity of its own ("it will stop the plant from needing water").
  Table additions: mass of empty/loaded carts, diffusion/spreading speeds up with temperature, "speeds (up)",
  "sunny" as more light, "dark" as darkness, "disappear/vanish/extinct" as decreases, "colds" as no quantity.

### Negative facts (new channel)
The parent's facts drop every word after a negator ("Magnets do not attract wood, plastic, …" becomes
"Magnets do"). That keeps "X is not Y" from supporting Y, but it throws away what negative facts say.
`neg_channel` uses those words, in separate rows. N is the denied part: the words after the negator, up to
because/so/while/…. P is the rest of the same sentence, up to ";" or ".". A row counts for an option when the
question and the option meet across the negation (question in P and option in N, or the other way round). It
also needs the question and option together to cover most of the denied part (squared coverage, so
"herbivores do not eat meat" says nothing about "eats only plants"). Negations inside conditions ("when a
plant does **not** get enough water, it wilts") are skipped. The sign:
- a plain option the fact denies loses (half weight; this case is the noisiest);
- an option that states the denial ("it does not move") gains;
- in a negated question the global flip turns this around, at full weight there, because a stated denial is
  better evidence than being the least supported option;
- questions with a negator of their own elsewhere are left out.

To use the channel I wrote 98 negative facts (`kb/g5_negative.txt`): non-conductors, things magnets do
not attract, not mammals/insects/plants, not part of a body system, not found in a habitat, not true of
planets, not chemical/physical changes, lab safety. My negated-question set went from 17/25 to 23/25.

### Retrieval core
- **Comparison references.** "Food coloring spreads faster in warm water **than in cold water**" supported
  "cold water". Concepts a fact names only after "than" now count 0.3 on the option side (the same rows with
  other values; the question side is unchanged).
- **Frame words.** "important", "purpose", "role", "main", "affect", "help", … have idf 6–8 because facts
  rarely use them, so a fact that happens to contain one won on that word alone ("helps a fish get oxygen" →
  fins). Their question weight is now 0.5.
- **"only"** is a content word ("eats only plants" vs "eats plants and animals").
- **Option words repeated from the question** keep weight 0.25 instead of 0.4. Echoes like "student" in "it
  moves toward the stronger student" were winning. I tried 0.4/0.25/0.15 by hand on my questions: 0.25 was
  +4/350 over 0.4, and 0.15 was worse.
- Tried and dropped: lower disc_floor (noise), down-weighting isa facts (no effect), linear instead of
  squared negative coverage (−1), a question-relevance factor on negative rows (−1).

### Knowledge
- `kb/g5_technology.txt` (84 facts): hand tools, materials and their uses, manufacturing (casting,
  molding, extrusion, forging), structures (beams, trusses, arch/suspension bridges, tension/compression),
  hydraulic and pneumatic systems, communication and energy technology, bioengineering, design terms.
  Several of these words were not concepts at all before.
- `kb/g5_negative.txt` (above) and `kb/g5_gaps.txt` (8 facts: germination, learned dog behaviours, activity energy,
  transparency).
- `seq.txt`: planets by length of year, sprouting order (root first), density of common objects.

## How I checked it, honestly

These are first-look results: each fresh set was written and run on the parent and on my model as it stood
*before* I looked at that set's failures.

| set | topic | parent | mine at first look |
|---|---|---|---|
| b7 | mixed | 37/40 | 37 |
| b8 | sentence options | 26/30 | 27 |
| b9 | negations, comparisons | 23/34 | 25 |
| b10 | mixed, technology | 26/30 | 28 |
| b13 | mixed | 25/30 | 26 |
| b15 | mixed | 23/30 | 25 |

That is +8 on 194 fresh questions (+4.1 points) at first look. On all 490 questions, after fixing what each
set showed, the model scores 431 (88.0%) and the parent 381 (77.8%). Three questions the parent gets right
are now wrong; all three are near-ties. My questions are easier than ARC, and part of the gain is on question
types I deliberately wrote more of (negations, food webs, technology). So I expect the real gain on ARC to be
smaller than the first-look +4 points: about +1 to +3. It should come mostly from the removed module false
positives ("most likely" and "change" questions happen often in ARC), negated questions, and food-web and
direction questions.

## Running

`build()` takes about 25 s. The forward pass takes about 35 ms per question on 4 cores (about 2.5 minutes for
4,400 questions). Peak memory is about 1.65 GB with long questions (the parent: 1.6 GB). After editing `kb/` or
the word lists, regenerate `vocab.txt` with `python make_vocab.py`, which keeps the old vocabulary. Words that
appear only in `qr.txt`, `seq.txt` or the cue lists in `build.py` must be added to `vocab.txt` by hand.
