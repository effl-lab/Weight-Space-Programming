# A hand-written association scorer for ARC

Nothing here is trained or fitted to data. The knowledge is a list of word clusters I wrote by hand
(`knowledge/*.txt`, one cluster per line, ~3,400 lines). Each line is a short fact or theme, written
telegraphically, e.g.

    barometer measures air pressure
    sedimentary rock layers sediments compacted cemented
    ! moon phases new full crescent quarter gibbous

`families.txt` lists derivational families (erode/erosion, magnet/magnetic, ...) that share one stem.
`build.py` turns these files into every weight with closed-form formulas. The only optimization in it is
a geometric design step: spreading code vectors apart so that unrelated clusters collide less. It never looks at
questions or accuracy.

## What the model computes

For each option it computes "how well do the option's words and the question's words share a written cluster",
weighted by how specific the words are. If the question contains a negation word (not / except / least) the
sign of the score is flipped.

## How the fixed transformer does it

Residual layout (64 dims): 4 code slots of 13 dims (one per attention head), then flags:
content, negation word, question/option position, word specificity (IMP), two norm-balancing dims, and
working dims (S, QC, S2, N, two anchor dims).

* **Embeddings.** Every cluster is assigned to one head and gets a 13-d unit code. Codes are spread for low
  mutual coherence, with extra push between clusters that share a topic or a word. A word's code in a head is
  the normalized sum of the (at most 2) clusters it keeps in that head. A word keeps at most 8 clusters: its
  `!` priority lines first (hand-picked for very common words such as water, energy, plant), then a greedy
  cover of distinct partners. Every embedding+position vector has zero mean and the same norm, so the first
  LayerNorm is an exact rescaling. Later writes into the residual sum to zero, and a large constant anchor
  added after block 0 keeps every later LayerNorm a near-constant rescaling as well.
* **Block 0 attention (option → question).** In head h the logit between an option token and a question
  content token is β·(cosine of their slot-h codes − θ). Question stopwords are sinks at logit 0, and option
  tokens are masked. So the attention mass on question words is ≈1 when a cluster-mate is in the question, and
  ≈0 otherwise. The value is the question word's specificity, summed over heads into S. One head also lets question
  tokens find negation words (N).
* **Block 0 MLP.** One neuron computes QC = (question position AND content word); the anchor is added here.
* **Block 1 attention (question → option).** The same matching, run the other way: question tokens attend
  to option content words, with question content masked via QC. The value is (content − QC), so option tokens
  get 0. Head 0 also lets option tokens find negation words in the question (N).
* **Block 1 MLP.** Writes the final per-token score into one dim. For option tokens it is min(S, cap). For
  question tokens it is specificity × matches, computed as a product from GELU's quadratic part. The sign is
  flipped by gate neurons when N≈1, and position gates zero out the other kind of token. A tiny bonus per option
  content word breaks exact ties toward longer options. A pass-through pair removes the old value of the dim.
* **Readout.** ln_f is a fixed rescaling. `score` reads the one dim, and the mean over real tokens gives the
  option's logit. Stopword tokens contribute exactly 0.

A mean shift that LayerNorm would cause (only the single position flag is not zero-sum) is folded back into
every matrix that reads a LayerNorm output.

## Design lessons (from questions I wrote myself)

* What helped most: hand-written priority lines for common words (+8 points); more lines made of specific
  vocabulary; merging near-duplicate lines (fewer clusters per head means fewer code collisions); matches
  weighted by specificity.
* What hurt: lines dense with common words (loose scenario descriptions); definition-style lines that give
  contrasting terms the same words (melting/freezing both "solid liquid"); making common comparatives
  (increase/decrease) stopwords; letting common words take part only through priority lines.
* A different code seed changes ±2–4 points, which is as large as most design choices. So I compared choices
  averaged over 4 seeds and shipped a fixed seed (12345), not the best-scoring one.

## Limits (measured on questions I wrote myself, not ARC)

* Capacity: a 64-d residual lets a word hold only about 8 clusters. A noise-free version of the same scorer
  scores ~13 points higher when every word keeps all of its written clusters. Common words lose most of theirs;
  the `!` lines decide which ones they keep.
* Code collisions add noise. On ~220 self-written questions, accuracy is ~0.53–0.57 depending on the code seed.
  I did not pick the seed by accuracy.
* It is purely associative: no arithmetic, no reasoning about experiments. ARC-Challenge questions were
  selected to defeat exactly this kind of word-association solver, so expect much lower accuracy there.
