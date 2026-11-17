# A hand-written ARC model (generation 3)

Nothing here is trained or fitted, and nothing is pretrained. All knowledge is in plain-text files I wrote;
`build.py` turns them into tensors with fixed formulas, and `arch.py` is a fixed tensor program. I started from
the parent's folder (generation 2, 52.1% on validation) and kept its retrieval core. To it I added
hand-written reasoning modules for question types that word-overlap retrieval cannot answer, more knowledge,
and a few fixes.

## Why these changes

The parent is a retrieval model: an option scores high when some fact shares words with both the question and
the option. On the questions I wrote, it failed in a few recurring ways that more facts do not fix:

- **Direction questions.** "Which change would *increase* the gravitational force?" picked "moving the objects
  farther apart", because the best fact ("as two objects move farther apart, the force of gravity gets
  weaker") shares the most words. Bags of words carry no sign.
- **Order questions.** "egg, larva, pupa, adult" and "egg, pupa, larva, adult" are the same bag of words and get
  the same score. "Which is the largest: cell, tissue, organ, organism" has no fact to retrieve.
- **Calculations.** Speed, density, work, averages and temperature changes: retrieval can only guess.
- **Experiment variables.** "Which is the independent / dependent variable, which must be kept the same?" The
  answer comes from the question's own wording ("the effect of X on Y"), not from a fact. The parent got 4 of
  10 of my questions of this type right.

Each module below is a small, explicit rule set that writes a score on top of the retrieval score. The
retrieval score is divided by its maximum over the four options, so each module's weight means the same thing
in every question.

## Modules (all in arch.py, data written by me)

1. **Qualitative reasoning** (`qr.txt`, `qrbuild.py`, `qr_state`/`qr_score`). About 150 quantities
   (temperature, friction, population, erosion, pitch, ...). Each has trigger words, words that carry their
   own direction ("hotter", "farther", "sandpaper" = more friction), and generic direction words: verbs
   ("increase", "reduces", "grows"; reach 5 tokens both ways), adjective-like words ("more", "less",
   "without"; 3 tokens, forward only), and rate words ("faster", "slowly", which set the direction of a
   process, not of a word that already has one). A wh-word blocks the reach (a new clause starts).
   "same/constant" cancel nearby directions ("the pressure stays the same"), and
   "rate/amount/number of" pass their direction on ("reduce the amount of runoff"). There are about 140
   cause→effect lines with about 330 signs (`R mass > gravity`, `R distance < gravity`, `R slope > erosion`,
   ...), plus two-step chains at half weight. Each text (question, option) gets a direction (−1..1) per
   quantity. Option score:
   - term 1: the option's change, pushed through the cause→effect signs, agrees with the direction that the
     *asked part* of the question states explicitly (from the last wh-word on, or the last 12 tokens, plus 8
     tokens after "wants/needs"). In why/reason questions it is the whole question's stated direction ("The
     rabbit population grows quickly. What is the most likely reason?" → "more food is available");
   - echo: the option itself names the asked change ("placing sandpaper" for "increase the friction");
   - term 2: the change the question describes predicts the direction the option states ("If the Moon were
     farther from Earth, the tides would be …" → "lower"). Off in why/because questions, where the options are
     causes, not effects.

   Lessons: unrestricted self-links let options echo the question ("Metal is colder than wood" for "why does
   metal feel colder"); directions stated in the scenario are not goals ("a new fertilizer doubles growth").
   Hence the asked-part and explicit-direction restrictions.
2. **Order and extremes** (`seq.txt`, `seqbuild.py`, `seq_score`). 47 ordered sequences: life cycles, levels
   of organisation, planets by distance and size, Earth's and the atmosphere's layers, star colour vs
   temperature, Mohs hardness, units, food chains, processes such as forming sedimentary rock. An option that
   lists at least three items of a sequence gains by how well their order agrees (pairwise, by cumulative rank
   counts), reversed by "from largest to smallest". When the question names an end ("smallest", "farthest",
   "hardest") and at least two options are items of a sequence, the options are ranked by position.
   Sequences whose items are often named in either order (states of matter, hardness) are rank-only.
3. **Arithmetic** (`numbers`/`arith_score`). The integers 0–10000 were added to the vocabulary. A 3-digit group
   after a number is read as its thousands part ("1,200"). From the first six numbers of the question it forms
   quotients, products, sums, differences and the mean. An option whose number matches gains, more when the
   asked part names the operation (speed/density → divide; work/how far → multiply; total → add;
   change/difference → subtract; average → mean).
4. **Experiment variables** (`var_spans`/`var_score`). Finds the cause span X and effect span Y from
   "effect of X on Y" and "whether/how X affects Y". It then credits the option about X for
   independent/changed, the option about Y for dependent/measured, and options about neither for
   controlled/kept the same.
5. **Concept association** (PMI over the facts, weight 0.2): weak evidence when no single fact matches well.

Smaller changes: "same" is no longer a stopword ("the same mass" was an empty option); "cannot/can't" negate
the question like "not", but a negator after "why"/"reason" does not ("the main reason people should not
throw batteries in the trash" asks for the reason); "bounce" is related to "reflect"; the dense operand of every sparse product is made
contiguous (with exactly 4096 concepts the strided products became 4× slower).

## Knowledge added

Ten new files `kb/g3_*.txt`, about 1,430 facts in the one-subject-per-clause style the parent recommended:
energy changes in devices and heat transfer; adaptations, behaviour, traits and ecosystem changes; weather,
water, rocks, plate tectonics and Earth's history; matter and chemistry situations; inquiry, tools, units,
lab safety and engineering; space details; body systems and health; forces, electricity and waves; and a
middle-school glossary in two batches (autotroph, osmosis, polygenic, transform boundary, valence electron,
alloy, mass wasting, ...). The glossary was prompted by a grade-8 question set (d13) on which the parent and
my model both scored 63%. On a fresh grade-8 set written afterwards (d14) the result was 96.7% vs 90%.

Writing lesson (found from a failure): a fact clause loses its words after "not/cannot", so "S waves cannot
travel through liquid" supported nothing about S waves. Facts whose content is a negative property are
written positively ("liquids stop S waves", "sand is insoluble in water").

## How I checked it

All my questions are made up and are only in `/srv/work/devtools` (d1–d17, 585 questions; d7, d9, d10 and
d12 are small targeted sets for one module each). On all of them the parent scores 76.5% and this model
87.0%. These numbers flatter the model: after each set was written I fixed the failures it showed.

The honest numbers are first-look results, from sets written before the changes they test (this model vs
parent at the time): d6 78 vs 88 (QR false positives, which led to the asked-part rules), d8 90 vs 90, d11
72.5 vs 72.5, d13 63 vs 63, d14 96.7 vs 90, d15 76.7 vs 76.7, d16 (direction questions) 60 vs 50, d17 75 vs
75. On ordinary questions the model mostly ties the parent at first look. It gains where the modules apply.
Targeted sets: order/extremes 20/20 vs 57.5%, calculations 8/8 vs 1/8, variables 9/10 vs 4/10. Removing QR
alone loses 17 questions and gains 1 on the 585.

My questions are easier than ARC: the parent scored ~77% on them but 52% on validation. So I expect a real
gain of a few points, mostly from the modules on the question types they target (direction, order,
calculation, variables), which I believe are common in ARC, and somewhat from the glossary.

Weights of the modules were set by hand on these questions (qr_w 1, seq_w 0.5, ext_w 2, ar_w 2, var_w 2,
pmi_w 0.2). Tried and removed: a category route through isa.txt for every option (it helped before the new
knowledge, then only added noise); the parent's contrast and length parameters are flat on my sets and are
unchanged.

## Running

`build()` takes about 25 s (vocabulary-level tables for the modules). The forward pass takes about 17 ms per
question on 4 cores (about 1.3 minutes for 4,400 questions), and peak memory is about 1.4 GB. After editing `kb/` or the word lists, regenerate
`vocab.txt` with `python make_vocab.py`. It keeps the old vocabulary, so the numbers and module words stay in. Words that appear only in `qr.txt`,
`seq.txt` or the cue lists in `build.py` must be added to `vocab.txt` by hand (otherwise they are id 1).
