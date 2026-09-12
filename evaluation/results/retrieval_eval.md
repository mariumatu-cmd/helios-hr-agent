# Retrieval ablations

Index: 133 chunks / 12 documents, model `BAAI/bge-small-en-v1.5`, k=6.

| configuration | recall@1 | recall@6 | MRR | false accepts | false refusals |
|---|---|---|---|---|---|
| hybrid + both gates (shipped) | 0.88 | 1.00 | 0.938 | 0/8 | 0/16 |
| dense only | 0.88 | 1.00 | 0.938 | 0/8 | 0/16 |
| bm25 only | 0.81 | 0.94 | 0.856 | 0/8 | 0/16 |
| hybrid, similarity gate only | 0.88 | 1.00 | 0.938 | 8/8 | 0/16 |
| hybrid, lexical gate only | 0.88 | 1.00 | 0.938 | 0/8 | 0/16 |
| hybrid, no gates | 0.88 | 1.00 | 0.938 | 8/8 | 0/16 |
