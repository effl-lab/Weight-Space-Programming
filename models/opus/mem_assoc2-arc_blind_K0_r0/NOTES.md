# A hand-written second-order association memory for ARC science questions

Nothing here is trained or fitted. Every word and every association was written by hand in `kb/`;
`build.py` turns that text into the tensors `arch.py` expects, with fixed formulas whose constants
were chosen by reasoning about what each part should do (they are listed, with a comment each, at
the top of `build.py`).

## What the knowledge looks like (`kb/`)

| file | content |
|---|---|
| `facts_*.txt` (≈ 5,400 lines) | grade 3–9 science, organized by topic: Earth, space, life, body, physics, chemistry, inquiry and safety, technology, environment, everyday objects, typical question scenarios |
| `tax.txt` | taxonomy: `parent: child, child, ...` (organisms, rocks, minerals, planets, organ systems, simple machines, conductors/insulators, renewable/nonrenewable resources, ...) |
| `same.txt` | morphology the stemmer can't work out alone: irregular forms (*mice*, *froze*), derivations that mean the same thing (*erosion/erode*, *solar/sun*, *producer/produce*); `!word` keeps a word as its own concept (*weathering* ≠ *weather*) |
| `stop.txt`, `common.txt` | function words (concept 0), and general English words whose weight is capped low (*form*, *process*, *main*, ...) |
| `adj.txt` | adjectives whose comparatives/superlatives map to them (*hotter*, *densest*) |

Line formats in `facts_*.txt`:

* `cues > answers`: a question that mentions the cues supports the answers. Example: `sun rises > east`.
* `term = definition words`: both directions. A question that mentions most of the definition gets the term, and
  the term gets the definition. The term plus one definition word also recall the rest of the definition.
* `cues >! answers`: evidence against. Example: `magnets attract >! plastic wood glass ...`.
* plain lines: a topic cluster, symmetric and weaker. `facts_synonyms.txt` holds near-synonyms and is weaker still.

## From text to weights (`build.py`)

* **Words → concepts.** A small suffix stemmer (plurals, -ed/-ing/-ly, -ation/-ate/-tion, final e, doubled
  consonants) plus `same.txt` maps every word to a concept. `vocab.txt` holds every written word with its
  inflections and base forms (≈ 40k words, ≈ 3,800 concepts). Run `python build.py --vocab` after you edit `kb/`.
* **idf.** Document frequency over the fact lines, scaled to about 1–4. General words are capped at 0.8.
  `not`/`except` get 4.5, so they are always among the J = 16 strongest question concepts.
* **First-order `A`.** Each rule adds cue → answer weight. That weight is scaled by 1/√(#cues), and by √(2/#answers)
  when the rule has many answers, so specific rules beat broad lists. The words of one named thing in the taxonomy
  share a single weight, so *inclined plane* doesn't count twice. Answers also support their cues at 0.3. The
  taxonomy closure links parent ↔ descendant, with 0.5 decay per level and less per child when a parent has
  many children. Rows of hub concepts and columns of hub answers are then damped by 1/√(1 + fan/ref). Last, a
  two-hop term (q → b → c, with the second hop going either way) adds 0.5 of the direct mass. This matches
  options that paraphrase a concept's definition (e.g. *blend in with surroundings* for camouflage).
* **Pair table** (`row_ids`, `row_w`, H = 2^20 rows × R = 40). Every pair of cues in a rule recalls its answers.
  A definition's term paired with one definition word recalls the other definition words. Topic clusters recall
  their other members. When many rules recall the same answer through the same pair, the weights add
  sublinearly (√).
* **Negation.** For each concept q, the pair (NOT, q) recalls q's strongest first-order and pair answers with
  negative weight. In "... EXCEPT" and "which is NOT ..." questions this turns the question's evidence around.
  `except` is strong (×2.5). `not` is mild (×0.4), because in scenarios ("the seed did not sprout") it isn't a
  negated question. My own test questions put both cases at about the same frequency.
* `mix` = 1.5 weighs the first-order read against the pair read.

## How it was checked

All checking was done by hand. I wrote practice questions of my own (`/srv/work/my_questions.json`, 476 questions, ARC-like
but invented) and read per-option contributions with `scripts/explain.py` to find missing knowledge and
systematic problems. Problems found this way:
* multi-word answers counted twice;
* hub words such as *sun*, *energy* and *form* swamping specific facts;
* cause → effect facts that were only written effect → cause;
* scenario *not* being treated as negation.

On a fresh batch of my own questions the model got about 55–90%; it got 8 of the 10 ARC examples in the task.
Real ARC questions (especially Challenge) need more reasoning than association can give, so the test score will
be lower.

Limits: no word order (sequence questions such as life-cycle order are ties); no arithmetic; *increase* and
*decrease* are seen only as associations, not as directions.
