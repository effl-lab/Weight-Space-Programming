# A hand-written second-order association memory for ARC science questions (generation 2)

Nothing here is trained or fitted. Every word and every association was written by hand in `kb/`;
`build.py` turns that text into the tensors `arch.py` expects. It uses fixed formulas, and their constants
were chosen by reasoning about what each part should do. They are listed with a comment each at the top of
`build.py`. This generation keeps the parent's design and compiler, adds about 2,500 lines of knowledge, and fixes
mechanism problems found by reading the model's per-option explanations on invented questions.

## What the knowledge looks like (`kb/`)

| file | content |
|---|---|
| `facts_*.txt` (parent, ≈ 5,400 lines) | grade 3–9 science by topic: Earth, space, life, body, physics, chemistry, inquiry and safety, technology, environment, everyday objects, typical question scenarios |
| `facts_g2_*.txt` (new, ≈ 2,500 lines) | question pattern → answer rules for Earth/weather/water/rocks/soil, space, life science, body, physical science, chemistry, inquiry and engineering, scenarios ("which best explains"), cause → effect chains, organisms, scientists and inventions, earth history, "function of X", and **misconceptions** that ARC distractors use (`seasons cause >! distance closer farther`) |
| `facts_g2_links.txt` | links between base words and the derived words that are now their own concepts (see below) |
| `tax.txt` | taxonomy `parent: child, ...`; generation 2 adds everyday categories (colors, seasons, metals, fuels, storms, conductors and insulators, physical and chemical changes, landforms, lab tools, pollinators, ...) |
| `same.txt` | morphology; `!word` keeps a word as its own concept |
| `stop.txt`, `common.txt`, `adj.txt` | function words, general words with a capped weight, comparatives |

Line formats: `cues > answers` (directed), `term = definition` (both ways), `cues >! answers` (evidence against),
plain lines (topic cluster, symmetric and weaker). `#` starts a comment.

## From text to weights (`build.py`): the parent's pipeline

* A stemmer plus `same.txt` maps words to concepts (`vocab.txt` ≈ 43k words, ≈ 4,100 concepts; run
  `python build.py --vocab` after editing `kb/`).
* idf comes from document frequency over the fact lines. General words are capped at 0.8 and `not`/`except` get 4.5.
* First-order `A`: rule weights scaled by 1/√(#cues) and √(2/#answers); answers support cues at 0.3; taxonomy
  closure; hub damping; and a global two-hop term.
* Pair table (H = 2^20 rows × R = 40): every pair of cues in a rule recalls its answers. Recalls that add up
  through one pair grow sublinearly.

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
