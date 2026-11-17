# A hand-written second-order association memory for ARC science questions (generation 3)

Nothing here is trained or fitted. Every word and every association was written by hand in `kb/`;
`build.py` turns that text into the tensors `arch.py` expects. It uses fixed formulas, and their constants
were chosen by reasoning about what each part should do. They are listed with a comment each at the top of
`build.py`. Generation 3 keeps the generation-2 design and compiler, adds about 2,500 lines of knowledge
(`facts_g3_*.txt`), and changes the mechanism where reading per-option explanations on invented questions showed
a systematic fault. Those changes are listed under "What generation 3 changed" below. The generation-2 sections
follow unchanged except for numbers that have since changed.

## What the knowledge looks like (`kb/`)

| file | content |
|---|---|
| `facts_*.txt` (parent, ≈ 5,400 lines) | grade 3–9 science by topic: Earth, space, life, body, physics, chemistry, inquiry and safety, technology, environment, everyday objects, typical question scenarios |
| `facts_g2_*.txt` (new, ≈ 2,500 lines) | question pattern → answer rules for Earth/weather/water/rocks/soil, space, life science, body, physical science, chemistry, inquiry and engineering, scenarios ("which best explains"), cause → effect chains, organisms, scientists and inventions, earth history, "function of X", and **misconceptions** that ARC distractors use (`seasons cause >! distance closer farther`) |
| `facts_g3_*.txt` (new, ≈ 2,500 lines) | matter and chemistry, energy/forces/waves/electricity, Earth and weather, space, life, ecology and resources, body and health, inquiry/tools/safety/engineering, facts as state tests ask them, classroom investigations and their usual results, paraphrases of frequent facts, regional facts and everyday estimates, rules using the new *in/out* and *without* concepts (`facts_g3_direction.txt`), and **distractor traps** (`facts_g3_traps.txt`: `sound travels fastest >! vacuum air`) |
| `facts_g2_links.txt` | links between base words and the derived words that are now their own concepts (see below) |
| `tax.txt` | taxonomy `parent: child, ...`; generation 2 adds everyday categories (colors, seasons, metals, fuels, storms, conductors and insulators, physical and chemical changes, landforms, lab tools, pollinators, ...) |
| `same.txt` | morphology; `!word` keeps a word as its own concept |
| `stop.txt`, `common.txt`, `adj.txt` | function words, general words with a capped weight, comparatives |

Line formats: `cues > answers` (directed), `term = definition` (both ways), `cues >! answers` (evidence against),
plain lines (topic cluster, symmetric and weaker). `#` starts a comment.

## From text to weights (`build.py`): the parent's pipeline

* A stemmer plus `same.txt` maps words to concepts (`vocab.txt` ≈ 44k words, ≈ 4,150 concepts; run
  `python build.py --vocab` after editing `kb/`).
* idf comes from document frequency over the fact lines. General words are capped at 0.8 and `not`/`except` get 4.5.
* First-order `A`: rule weights scaled by 1/√(#cues) and √(2/#answers); answers support cues at 0.3; taxonomy
  closure; hub damping; and a global two-hop term.
* Pair table (generation 3: H = 2^22 rows × R = 16): every pair of cues in a rule recalls its answers. Recalls that
  add up through one pair grow sublinearly.

## What generation 2 changed, and why

1. **Derived words that are answers are their own concepts.** The architecture drops any option concept that the
   question also has. The parent merged *sediment/sedimentary*, *sun/solar*, *moon/lunar*, *heat/thermal*,
   *digest/digestive*, *respiration/respiratory*, *circulation/circulatory*, *nerve/nervous*,
   *continent/continental*, *tide/tidal*, *metamorphosis/metamorphic*, and others. So in "layers of sediment pressed
   together … this rock is", the right answer *sedimentary* scored exactly 0. These are now separate concepts, and
   `facts_g2_links.txt` links them back to their base words.
2. **Stemmer collisions fixed.** *plane→plan* (a capped general word), *care→car*, *nuclear→nucleus*,
   *invisible→visible*, *insoluble→soluble*, *mate→mat*, *spine→spin*, *age→ag*, … (`same.txt`, last block).
3. **Direction concepts.** *more/increase/greater/high(er)* map to one concept and *less/fewer/decrease/reduce/low(er)*
   to another. The parent made *more* and *less* stop words and capped *increase* low. Direction words carry no
   topic, so they get no first-order associations and act only inside pairs: (*float*, *water*) recalls *less
   dense*, and (*altitude*, up) recalls *down pressure*. A pair that recalls one direction also argues against
   the other (×0.7). Topic clusters and "how high or low" definitions drop direction words.
4. **Negation without the "partner" term.** The parent's (NOT, q) rows also negated q's strongest pair recall.
   That flipped ordinary scenario questions ("the seeds did not sprout"): on my scenario-*not* questions it took
   accuracy from 90% down to 50%, and it didn't help true NOT questions. Now (NOT, q) negates only q's first-order
   answers, with `not` mild (0.4) and `except` strong (2.5).
5. **Category rules fire for members.** A rule whose cues name a taxonomy category (e.g. *herbivore*, *metal*,
   *color*) also fills the pair (member, other cue) at 0.5 with taxonomy decay. So a rule about *metal* + *heat*
   also answers a question about *copper* + *heat*.
6. **Definition chaining.** When a rule's answer is mainly a defined term (*camouflage*), the cues also recall the
   term's definition words (*blend in with surroundings*) at 0.5, because options often paraphrase the term.
7. **Multiword answers share weight.** Words of a multiword name (taxonomy entries, defined terms) recalled as an
   answer get 1/√k each, so *wheel and axle* and *carbon dioxide* don't win just by having two words.

## How it was checked

All checking was done by hand, on questions I invented (`/srv/work/qs/*.txt`, about 400 ARC-style questions,
copied into `/srv/work/my_questions.json`), using `scripts/explain.py` to read per-pair and per-concept
contributions. Each mechanism change was kept only when it was principled *and* not harmful on these questions.
No setting was searched. Accuracy on my sets is about 79%, against 78% for the parent. On my most ARC-like
"Challenge-style" sets it is about 70%. Real ARC is much harder than my own questions: the parent scored 47.7% on
validation while getting 55–90% on its writer's questions.

Known limits: no word order ("light is faster than sound" and "sound is faster than light" are the same bag); no
arithmetic; an option whose words all appear in the question scores 0; true NOT/EXCEPT questions are only partly
handled. A question that mentions both a property and its opposite (*renewable* and *nonrenewable*) is ambiguous.

## What generation 3 changed, and why

Each change comes from a fault seen in `scripts/explain.py` output (per pair and per concept contributions) on
invented questions. Each was kept only if it was principled and did not hurt the invented sets. No constant was
searched. The two constants that were compared against a few alternatives (A_UNIT, and the size of the pair
table) were set to the value the reasoning gave: A_UNIT equal to PAIR_UNIT, and the largest table that fits the
memory budget.

1. **First-order sums saturate (A_UNIT = 1).** The pair table already added recalls sublinearly, but `A` added
   them linearly. So a pair mentioned together in many lines (*atom* and *proton*) outweighed a specific rule
   (*negative charge* → *electron*). `A` now uses the same √ growth above 1.
2. **Bigger, sparser pair table (2^22 × 16 instead of 2^20 × 40).** A pair of question concepts that has no row
   of its own still reads whatever row it hashes to. At 2^20 about 12% of rows were occupied, so explanations kept
   showing recalls from unrelated pairs (e.g. *(not, person) → more*). At 2^22 about 3% are occupied. Only 8% of
   recall entries, the weakest ones of very long generic rows, fall beyond 16 per row. `build()` now frees the
   Python tables before allocating the tensors, so peak memory stays about 1.3 GB, as before. (A table with no
   collisions at all is impossible here: by Dirichlet's approximation theorem it needs more than C² ≈ 17M rows.)
3. **More pair-only concepts, like generation 2's more/less.**
   * *in* (absorb, inhale, intake) and *out* (release, emit, exhale, give **off**): "plants **give off**" and "plants
     **take in**" were the same bag. They are opposites, like more/less.
   * *without* (without, lack, no, none, absence): a rule whose cues contain it fills only the pairs that include it.
     So "animals **without** a backbone → invertebrate" no longer also fires for "animals **with** backbones".
   * *only*, *both*: "eats **only** plants" (herbivore) vs "eats **both** plants and animals" (omnivore).
   * *gain(s)* is now "more" and *lose(s)* "less". *greatest* was mapped to *great* rather than to "more"; fixed.
4. **Negation.** On 15 invented NOT/EXCEPT questions the parent was right once and ranked the right answer **last**
   14 times. A stronger negation helps those but flips scenario questions ("the seeds did not sprout"), as
   generation 2 found. Pair rows can't see whether *not* belongs to the question or to the scenario. So:
   * *except* (almost always a real negation) now also takes back what its concept recalls through a
     **phrase-mate**: a word that names one thing with it (taxonomy names, defined terms, e.g. *simple* +
     *machine*). In "all of these are simple machines EXCEPT", the (simple, machine) pair recall is what lifted
     the wrong options.
   * *not* cancels its concepts' first-order recall (strength 1.0, up from 0.4) but leaves pair recall alone. With
     the same knowledge this took negated questions from 1 to 3 of 15 and kept scenario-not at 10 of 12. The final
     model gets 4 of 15 and 9 of 12 (parent: 2 and 10).
   * *least* is treated like *not*: "which is least likely to conduct electricity" asks for the least supported
     option.
5. **Variable-identification questions.** For "which is the independent (dependent) variable", the answer names
   the factor the stem says was changed (measured). The architecture drops every option concept that the question
   has, so that option scored 0 and lost to any distractor bringing a weakly related word. When the question has
   *independent*, *dependent* or *manipulated*, each new option concept now costs a fixed amount. The usual result
   is a tie among options that only restate the stem, not a wrong pick.
6. **Morphology.** *collide/collision, expand/expansion, divide/division, compress/compression,
   survive/survival, die/death/dead, farm/farmer, hunt/hunter, strong/strength, deep/depth*, … are merged
   (`same.txt`, last block). *ramp* is no longer merged with *inclined*: in "which simple machine is a ramp?" that
   merge deleted *inclined* from the right answer.
7. **No negations on answer sides.** `not` and `cannot` are dropped from fact lines, so `physical change = … no new
   substance` taught that a physical change recalls *new substance*. Generation-3 lines avoid this
   (`nonliving`, `treeless`, `bare ground`, …).

## How generation 3 was checked

Using about 550 invented questions in `/srv/work/qs*/` (also in `/srv/work/my_questions.json`) with `/srv/work/scripts/`:
`ev.py` (accuracy, and gains and losses against the previous run) and `explain.py`:

| set | parent | generation 3 |
|---|---|---|
| all 551 | 77.6% | 87.1% |
| a fresh set written after most changes and never debugged (`qs4/h1.txt`, 50) | 88% | 90% |
| Challenge-style set, every distractor topically linked (`qs4/chal2.txt`, 25) | 61% | 89% (debugged afterwards) |
| NOT/EXCEPT (`qs3/neg.txt`, 15) / scenario-not (`qs3/scen_not.txt`, 12) | 2 / 10 | 4 / 9 |

These numbers say little about ARC. My questions are textbook-like, and on fresh ones both models score about
the same (88–90%), while the parent scored 48.9% on ARC validation. The gains above come mostly from questions
whose failures I read and then wrote knowledge for. I expect the ARC gain to come from the mechanism fixes (1–7),
which apply to every question, and from the breadth of the new rules and traps.

Known limits (in addition to those above): word order is invisible ("liquid to gas" = "gas to liquid", "north
and north" = "north"). Options that restate the question score 0. A topic word can still favor a wrong option
when no trap rule covers it. True NOT questions are still mostly wrong when the right answer is topically close
(*scar* among inherited traits).
