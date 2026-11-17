# A hand-written ARC model (generation 4)

Nothing here is trained or fitted, and nothing is pretrained. All knowledge is in plain-text files I wrote.
`build.py` turns them into tensors with fixed formulas, and `arch.py` is a fixed tensor program. I started from
the parent's folder (generation 3, 53.5% on validation). Its structure is unchanged:

- **Retrieval core.** An option scores high when a fact shares idf-weighted concepts (Porter stems plus
  synonyms) with both the question and the option. The fact rows come in four kinds: whole facts, 4-concept
  windows, single clauses, and facts with their neighbours. There are two routes (the option's own words, and
  its isa.txt categories) and two passes (the second adds the words of the question's best facts). The retrieval
  score is divided by its maximum over the four options, and PMI association is added at weight 0.2.
- **Reasoning modules**, each a small, explicit rule set written on top of retrieval. qualitative reasoning,
  `qr.txt`: quantities, direction words, cause→effect signs. Order and extremes, `seq.txt`. Arithmetic.
  Experiment variables.
- **Negated questions** ("NOT", "EXCEPT") flip the sign of the score.

See the parent's notes, which are summarized in the code docstrings, for those parts. This file describes
what generation 4 changed, and why.

## What I found first

I wrote 515 ARC-style questions of my own (`/srv/work/dev/a1..a12.txt`, written in rounds). I ran the parent on
each round before changing anything. The parent scores about 80% on them, compared with 53.5% on ARC
validation, so my questions are easier than ARC. When I traced the parent's failures, the knowledge base
already contained the needed fact in most cases. A check of ~600 grade-school science terms found only a few
missing from the kb. The failures came from scoring: near-ties between options, list facts that name several
options, synonym merges (solar→sun matches lunar-eclipse facts), and distractors that are associated with the
question's subject ("iron" → "attracted to a magnet" for "a chemical property of iron"). I tried the retrieval
knobs by hand (second pass weight, window weight, question idf power, category route, a superlinear question
match). Each one moved my 300–500 questions by only ±1–3 in either direction, so I left the core parameters as
they were.

So most of the work went into (1) rules for question types that retrieval cannot answer, (2) a few principled
fixes to the core, and (3) about 1,060 new facts.

## Changes

### Qualitative reasoning (the largest gain)
On direction questions ("If X increases, what happens to Y?" / "Which change would make Y smaller?"), tracing
the parent's failures found parsing bugs and missing rules, fixed generally:

- **Constant quantities.** "the temperature increases at constant pressure": the parent's "constant"
  cancelled every direction within 3 tokens, including the temperature's. Now a quantity keeps a direction word
  that stands right next to it. Quantities held constant ("keeping the force the same") are also left out of the
  targets that short options ("It decreases") speak about. Before, "force" cancelled "acceleration" there.
- **Goal vs condition.** In "Which would most likely decrease the number of rabbits?" the decrease is the
  *goal*, not a change that has happened. Directions in the asked part before its first if/when/as are goals.
  They are no longer used as the starting change of term 2 (prediction from the question's change). This does
  not apply to "How does X affect Y" / "What happens", or to questions without a wh-word.
- **Comparatives.** A comparative after a linking verb or a quantity points back ("the population becomes
  larger", "stirred less"), unless it modifies a rate word ("melt more slowly"). Comparatives do not reach past
  "than" ("lower than at sea level": sea level is the reference). A verb with a quantity right before it (its
  subject) reaches only 2 tokens forward ("the wind speed increases over the ocean, the size of the waves …").
- **Option forms.** An option that starts with a pronoun ("it is lower than at sea level") speaks about the
  question's undirected quantities, as short options already did. When the asked part has a direction but
  names no quantity ("which is most likely to decrease?"), that direction applies to each option's quantities.
  When the asked part states no change ("… increases when the water is"), the question's stated change is the
  target at half weight.
- **Echo.** The echo rule rewards an option that names the asked change itself ("sandpaper" for "increase the
  friction"). It now counts only words with their own direction, so "swing it higher" no longer echoes "swing
  faster".
- **Table.** `qr.txt` now supports `! : words`, words that name no quantity although their stem does
  ("dissolved oxygen" is not dissolving). The word before "loss" names the loss ("heat loss"). There are about
  40 new relations: stretch→potential energy, wrapping/foam/lids slow cooling and melting, warm water holds
  less oxygen, melting reduces ice and raises sea level, freezing lowers water's density, winter/summer and
  daylight, muddy water and light, darkness and photosynthesis, producers→consumers, low pressure→storms,
  cars/factories→air quality, process rate→time. Also slope words ("steeper") with their own direction,
  mountains as altitude, heaters/stoves as heat, rubber/mats as friction, "off" and "worse" as decreases. A
  stem collision was removed ("builds" had counted as urban construction).

### New modules
- **Food web** (`foodweb_score`). Reads food relations stated in the question: "frogs eat insects and snakes
  eat frogs", and the passive "mice are eaten by owls". Each verb links the nearest content word before it to
  the nearest one after it. A change to one organism ("a disease killed most of the frogs") spreads: its prey
  change the opposite way, its predators the same way, and organisms two steps away at half weight. An option
  gains when it states an organism's direction correctly. Weight 1.5.
- **Word order for same-word options** (`order_score`). "The Moon orbits Earth" and "Earth orbits the Moon"
  have the same bag of words and tied before. For options whose bag equals another option's, each pair of
  consecutive option words is checked against the 8 best facts: +1 if they stand within 2 places in the same
  order, −1 if in reverse. Weight 0.3.
- **Experiment variables.** Two new patterns: "from different X" (the varied cause) and "measures Y" (the
  effect). A concept in both spans is told apart by its literal form ("different heights" / "how high"). Words
  after a non-generic "of" ("the shape *of a boat*") count half, because every option may mention the object
  studied.

### Core fixes
- **Category opposites** (`opp_rows`). For up to 3 category words of the question (conductor/insulator,
  physical/chemical, predator/prey, igneous/sedimentary/metamorphic, weathering/erosion, mass/weight,
  plant/animal, mammal/reptile/…), a fact row that has the opposite word but not the word itself counts
  exp(−0.7) ≈ 0.5. "Copper is a good conductor" no longer supports copper for "Which is an insulator?". This is
  deliberately not done for amount and direction words (more/less, increase/decrease). Their opposite rows are
  evidence by sign ("fewer bees → fewer fruit" supports "bees → more fruit"); an early version that damped
  them lost such questions.
- **Negation concept.** Facts that state an absence ("the Moon has no atmosphere") carry a concept NEG, and so
  do options with a negator. NEG is removed from the question bag, so "Which is NOT …" does not favour
  negative facts.
- **Stopwords and synonyms.** "toward(s)", "inside", "against", "beneath" and "underneath" carry meaning now
  ("tilted toward the Sun" vs "away from the Sun"). meteor/meteoroid share a concept.
- The succession sequence's first item is "bare" (rock) rather than "rock", so "the soil will turn to rock" is
  not the "first" stage after a forest fire.

### Knowledge
There are five new files, `kb/g4_*.txt`, with about 1,060 facts in the one-relation-per-clause style:
- **Inquiry:** precision vs accuracy, which tool measures what, units, variables, controls, sample size,
  safety, the design process.
- **Physical:** atoms and the periodic table (valence electrons by group), conservation of mass, physical vs
  chemical changes, states of matter, energy transformations of common devices, heat transfer situations,
  forces, simple machines, circuits, magnetism, sound and light.
- **Life:** cells and organelles, body systems, plants, animal groups, adaptations and behaviour, genetics and
  natural selection, ecology and biomes.
- **Earth and space:** rocks and minerals, weathering/erosion/deposition, plates, fossils, the water cycle and
  weather, the atmosphere, resources and pollution, Earth–Sun–Moon, the solar system and stars.
- **Long tail:** technology and inventions, scientists, detailed anatomy, protists and invertebrates,
  elements and materials, glacial and river landforms.

Most of it restates what the parent knew in crisper single clauses, and some of it fills gaps. On fresh
question sets, adding it changed little, as expected from the diagnosis above.

## How I checked it, honestly

These are the first-look results: each fresh set was run on the parent and on my model as it stood *before* I
looked at that set's failures.

| set | topic | parent | mine at first look |
|---|---|---|---|
| a3 | mixed | 49.2/62 | 50 |
| a4 | mixed | 41/51 | 42 |
| a5 | mixed, long tail | 44/50 | 44 |
| a6 | direction | 20/30 | 21 |
| a7 | direction + mixed | 23/30 | 23 |
| a8 | mixed | 34/40 | 34 |
| a9 | direction | 20/25 | 19 |
| a10 | mixed, longer stems | 24/30 | 24 |
| a11 | direction | 14/20 | 15 |
| a12 | direction | 11/15 | 12 |

That is +3.8 questions on 353 fresh questions (+1.1 points) at first look. After fixing what each set showed
(general rules and facts, never question-specific code), the model scores 458/515 (88.9%) on all my
questions, against the parent's 413/515 (80.2%), with no question the parent got right and mine gets wrong. The
real gain on ARC will be between those two numbers and probably near the low end. I expect +1 to +3 points,
mostly on direction, food-web and variable questions and on "which is an insulator/chemical change"-type
questions. General recall questions should be about unchanged.

## Running

`build()` takes about 23 s. The forward pass takes about 34 ms per question on 4 cores (about 2.5 minutes for
4,400 questions), and peak memory is about 1.55 GB. After editing `kb/` or the word lists, regenerate
`vocab.txt` with `python make_vocab.py`. It keeps the old vocabulary, including the numbers 0–10000. Words that
appear only in `qr.txt`, `seq.txt` or the cue lists in `build.py` must be added to `vocab.txt` by hand.
