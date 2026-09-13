# Abstention: why a similarity threshold was not enough

This records a design decision that was made, measured, found wrong, and
remade. It is the single most consequential correctness change in the project,
so the reasoning is kept in full rather than summarised into the final
architecture.

Reproduce with:

```
python scripts/calibrate_threshold.py      # the original calibration
python scripts/diagnose_abstention.py      # why it failed
python scripts/test_abstention_rule.py     # the replacement
python scripts/recalibrate_threshold.py    # the residual job of the old gate
python -m evaluation.run_retrieval_eval    # the ablation table
```

## 1. The original decision

The system must refuse questions the corpus does not cover, rather than
grounding an answer in the nearest irrelevant policy. The obvious mechanism is a
floor on the best cosine similarity between the query and any chunk.

That floor was calibrated, not guessed. `scripts/calibrate_threshold.py` scored
15 in-corpus and 10 out-of-corpus questions:

| population | min | median | max |
|---|---|---|---|
| in corpus | 0.6882 | 0.7518 | 0.8713 |
| out of corpus | 0.4546 | 0.5360 | 0.6200 |

The distributions separated with a margin of +0.068. The midpoint, **0.65**, gave
0 false accepts and 0 false refusals. On that evidence the decision looked
settled.

## 2. How it failed

The evaluation suite introduced a second, independently written set of
out-of-corpus questions. These were harder in a specific way: they ask about
*plausible HR benefits that do not exist*, rather than about unrelated subjects.

Under the 0.65 floor, **7 of 8 were accepted as grounded**:

| question | best similarity | verdict |
|---|---|---|
| What does the Helios pet insurance policy cover? | 0.7708 | accepted |
| How do I enrol in the Helios company car scheme? | 0.6970 | accepted |
| What is the bereavement travel allowance for a second cousin? | 0.6913 | accepted |
| What is the tuition reimbursement cap for an MBA? | 0.6639 | accepted |
| What is the sabbatical policy after ten years of service? | 0.6632 | accepted |
| What was Helios's Q3 revenue? | 0.6547 | accepted |
| How many shares are in the employee stock purchase plan? | 0.6510 | accepted |
| Which Helios office has a rooftop swimming pool? | 0.5903 | refused |

Against the sixteen genuine evaluation questions, which scored 0.654 to 0.836,
the two populations overlap almost completely. "Pet insurance" scores higher
than ten of the sixteen real questions.

This is not a bug in the embedder. It is the embedder working correctly.
Cosine similarity measures topical adjacency, and a fabricated HR benefit is
topically adjacent to real HR benefits by construction. The first calibration
set was unrepresentative: its negatives were about unrelated subjects, so it
measured a distinction that is easy rather than the one that matters.

**No threshold could have fixed this.** Positives span [0.654, 0.836] and
negatives [0.590, 0.771]; the ranges overlap. `scripts/diagnose_abstention.py`
confirmed the same overlap for BM25 score and for raw term coverage.

## 3. The signal that does work

The diagnostic showed the discriminating information is not in any score, but in
*which* query terms fail to match the corpus:

| population | unmatched terms |
|---|---|
| out of corpus | sabbatical, tuition, pet, stock, shares, revenue, rooftop, swimming, pool, cousin |
| in corpus | maya, rodriguez, weber, silva, brazil, wants, starting, switching, blackouts |

The out-of-corpus questions fail on **topic nouns**. The in-corpus questions fail
on **entity names** and **inflected forms** of words that do occur.

That distinction maps onto something real about the architecture: this system
has two knowledge stores. Policy text is reached by retrieval; employee records,
offices and visa classes are reached by MCP tools. A term missing from the
policy corpus means very little, because it may live in the database. A term
missing from *both* means no combination of retrieval and tool calls can produce
a grounded answer.

`rag/vocabulary.py` implements that test:

- tokenise the question with the same tokenizer the BM25 index was built with
- drop generic scaffolding ("what", "policy", "wants", "exactly")
- drop capitalised non-sentence-initial tokens, which are entity references —
  "Brazil" is absent from every store and yet "can I work from Brazil" is
  answerable, because the international policy applies wherever you go
- normalise inflection so `blackouts`→`blackout` and `switching`→`switch`
- flag any remaining term present in neither the corpus vocabulary nor the
  vocabulary of the mock HR datasets

The proper-noun exemption does not weaken the gate, because an out-of-corpus
question still needs a topic and topics are common nouns. "What is the tuition
reimbursement cap for an MBA?" exempts `MBA` and is still caught by `tuition`;
"What was Helios's Q3 revenue?" exempts `Q3` and is still caught by `revenue`.

Measured on the same sets: **0/16 false refusals, 8/8 correct refusals.**

## 4. Ablation

From `evaluation/results/retrieval_eval.md`, over 16 graded queries and 8
out-of-corpus queries:

| configuration | recall@1 | recall@6 | MRR | false accepts | false refusals |
|---|---|---|---|---|---|
| hybrid + both gates (shipped) | 0.88 | 1.00 | 0.938 | 0/8 | 0/16 |
| dense only | 0.88 | 1.00 | 0.938 | 0/8 | 0/16 |
| bm25 only | 0.69 | 0.94 | 0.797 | 0/8 | 0/16 |
| hybrid, similarity gate only | 0.88 | 1.00 | 0.938 | 8/8 | 0/16 |
| hybrid, lexical gate only | 0.88 | 1.00 | 0.938 | 0/8 | 0/16 |
| hybrid, no gates | 0.88 | 1.00 | 0.938 | 8/8 | 0/16 |

Two honest readings:

1. **The lexical gate does all of the abstention work.** Similarity-only and
   no-gates are indistinguishable at 8/8 false accepts.
2. **Hybrid retrieval does not beat dense retrieval on ranking here.** On 16
   graded queries the two are identical. BM25 alone is worse. So the fusion is
   not earning its place through recall on this sample — it earns it through the
   BM25 vocabulary, which is what the abstention gate is built on, and through
   exact-identifier queries (`POL-INTL-001`, `H-1B`, `§3.1`) that this graded
   set is too small to exercise. Reporting hybrid as a recall win would not be
   supported by the measurement.

## 5. What the similarity floor is still for

Having lost its original job, the floor was recalibrated for the narrow one it
still does: rejecting input that is not a question about anything.
`scripts/recalibrate_threshold.py` measured three populations:

| population | min | median | max |
|---|---|---|---|
| genuine questions (incl. terse and statement-phrased) | 0.5770 | 0.7494 | 0.8011 |
| plausible but non-existent topics | 0.5903 | 0.6632 | 0.7708 |
| nonsense strings | 0.5085 | 0.5376 | 0.5518 |

Genuine and nonsense separate with a margin of +0.025, so the floor moved from
0.65 to **0.56**. It is explicitly *not* expected to separate the middle row;
that is the lexical gate's job.

The floor is kept rather than deleted because three nonsense inputs — `42`,
`?????`, and `the the the the the` — pass the lexical gate. They contain no
unknown content terms because they contain no content terms at all. Each gate
catches what the other cannot.

Lowering the floor also fixed a false refusal the old value caused:
statement-phrased queries ("Maya Rodriguez wants to work from Portugal for 42
days…") score lower than terse keyword queries ("equipment stipend"), and at
0.65 the realistic phrasing was being refused while the keyword phrasing was
not.

## 6. Known limitations

- **Proper nouns are detected by capitalisation.** A question typed entirely in
  lower case loses the exemption, so a novel place name could produce a false
  refusal. The blast radius is bounded: `search()` returns its best passages
  alongside the refusal verdict, so the agent degrades to a hedged answer rather
  than a blank one.
- **The generic-word list is hand-maintained.** Every entry is there because it
  appeared as a false signal in a measured run, but the list is not complete and
  a novel piece of question scaffolding could flag. `tests/test_abstention.py`
  pins the current behaviour so a regression is visible.
- **The negative sets are small** (8 and 7 questions). The claim is that the
  lexical gate is *better* than the similarity gate, which the data supports
  decisively; it is not a claim of a measured error rate on production traffic.
- **A term could exist in the corpus in a different sense** than the question
  intends, and the gate would pass it. Abstention is a floor on nonsense, not a
  guarantee of relevance; correctness of the final answer still depends on the
  model reading the passages it is given.
