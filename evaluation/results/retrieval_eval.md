# Retrieval ablations

Index: 221 chunks / 16 documents, model `BAAI/bge-small-en-v1.5`, k=6.

The legacy recall columns measure document-level hit rate (any expected document), not complete multi-document evidence coverage. MRR uses deduplicated documents.

| configuration | recall@1 | recall@6 | MRR | false accepts | false refusals |
|---|---|---|---|---|---|
| hybrid + both gates (shipped) | 0.88 | 1.00 | 0.941 | 0/8 | 0/17 |
| dense only | 1.00 | 1.00 | 1.000 | 0/8 | 0/17 |
| bm25 only | 0.59 | 0.94 | 0.728 | 0/8 | 0/17 |
| hybrid, similarity gate only | 0.88 | 1.00 | 0.941 | 8/8 | 0/17 |
| hybrid, lexical gate only | 0.88 | 1.00 | 0.941 | 0/8 | 0/17 |
| hybrid, no gates | 0.88 | 1.00 | 0.941 | 8/8 | 0/17 |
