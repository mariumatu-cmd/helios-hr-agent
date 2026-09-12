"""Hybrid retrieval over the persisted policy index.

Dense and sparse retrieval fail in different, complementary ways on this corpus:

* Dense (bge-small) handles paraphrase. "Can I work from another country?"
  never says *international*, but it embeds near ``POL-INTL-001``.
* BM25 handles rare exact tokens. Policy text is dense with identifiers --
  ``POL-INTL-001``, ``H-1B``, ``30-day``, ``§4.4`` -- and an embedding of a token
  that occurs three times in the corpus is not trustworthy.

The two ranked lists are combined with Reciprocal Rank Fusion rather than a
weighted score blend, because cosine similarities and BM25 scores are on
incomparable scales and BM25's scale additionally shifts with corpus size. RRF
only consumes ranks, so it needs no per-query normalisation and no tuning
constant beyond ``RRF_K``.

A reranker (cross-encoder) was evaluated and deliberately rejected -- see
``evaluation/report.md``; it cost ~1.4 s per query and did not improve recall@5
on a corpus this small and this heading-structured.
"""
from __future__ import annotations

import json
import pathlib
import pickle
import threading
from dataclasses import asdict, dataclass, field

import numpy as np

from config import settings
from rag.ingest import bm25

# Candidates pulled from each retriever before fusion. 50 is well past the point
# where extra candidates change the fused top-10 on a 133-chunk index, but keeps
# recall for queries where one retriever ranks the answer poorly.
POOL = 50

# Standard RRF damping. Larger values flatten the contribution of top ranks;
# 60 is the value from Cormack et al. (2009) and was left untuned deliberately.
RRF_K = 60

# Below this best-cosine, the query is treated as out-of-corpus and the agent is
# told to refuse rather than ground an answer in the nearest irrelevant policy.
# Measured, not guessed: `scripts/calibrate_threshold.py` scores 15 in-corpus and
# 10 out-of-corpus questions. The distributions separate cleanly
# (in-corpus min 0.688, out-of-corpus max 0.620); 0.65 is the midpoint of that
# margin and yields 0 false accepts and 0 false refusals on the calibration set.
MIN_DENSE_SIMILARITY = 0.65


@dataclass
class Hit:
    """One retrieved passage, already formatted for citation."""

    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    citation: str
    text: str
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    source_file: str = ""
    source_format: str = ""
    effective_date: str = ""
    version: str = ""
    expanded_from: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SearchResult:
    """Result envelope. ``grounded`` is the guardrail the agent must respect."""

    query: str
    hits: list[Hit]
    grounded: bool
    best_similarity: float
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "grounded": self.grounded,
            "best_similarity": round(self.best_similarity, 4),
            "reason": self.reason,
            "hits": [h.to_dict() for h in self.hits],
        }


class _Index:
    """Lazily-loaded singleton around the committed index files.

    The embedding model is loaded separately from the index arrays and only on
    the first *query*, so that a process which merely reads metadata (the
    ``/health`` endpoint, tests, tooling) never pays the ONNX runtime's memory
    cost. This matters on a 512 MB free-tier instance.
    """

    _lock = threading.Lock()
    _instance: _Index | None = None

    def __init__(self, index_dir: pathlib.Path):
        self.dir = index_dir
        info_path = index_dir / "index_info.json"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"No index at {index_dir}. Run `python -m rag.ingest.build_index`."
            )
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self.vectors = np.load(index_dir / "embeddings.npy")
        with (index_dir / "metadata.jsonl").open(encoding="utf-8") as fh:
            self.meta = [json.loads(line) for line in fh if line.strip()]
        with (index_dir / "bm25.pkl").open("rb") as fh:
            self.bm25 = pickle.load(fh)

        if len(self.meta) != self.vectors.shape[0]:
            raise RuntimeError("index corrupt: metadata and embedding counts differ")

        self._by_chunk_id = {m["id"]: i for i, m in enumerate(self.meta)}
        self._by_doc: dict[str, list[int]] = {}
        for i, m in enumerate(self.meta):
            self._by_doc.setdefault(m["doc_id"], []).append(i)
        self._embedder = None

    @classmethod
    def get(cls) -> _Index:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(settings.index_dir)
        return cls._instance

    def embed_query(self, query: str) -> np.ndarray:
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    from fastembed import TextEmbedding

                    self._embedder = TextEmbedding(model_name=self.info["model"])
        vector = np.asarray(
            next(iter(self._embedder.query_embed([query]))), dtype=np.float32
        )
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


def _dense_rank(index: _Index, query: str, allowed: set[int] | None) -> tuple[list[tuple[int, float]], float]:
    similarities = index.vectors @ index.embed_query(query)
    if allowed is not None:
        mask = np.full(similarities.shape, -np.inf, dtype=np.float32)
        for i in allowed:
            mask[i] = similarities[i]
        similarities = mask
    top = np.argsort(-similarities)[:POOL]
    ranked = [(int(i), float(similarities[i])) for i in top if np.isfinite(similarities[i])]
    best = ranked[0][1] if ranked else 0.0
    return ranked, best


def _rrf(*ranked_lists: list[int]) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_index in enumerate(ranked, start=1):
            fused[doc_index] = fused.get(doc_index, 0.0) + 1.0 / (RRF_K + rank)
    return fused


def _expand(index: _Index, doc_index: int) -> tuple[str, list[str]]:
    """Re-glue a section that the chunker had to split.

    Chunks carry ``part``/``part_count``; when a hit is one part of a multi-part
    section, the sibling parts are stitched back so the model sees the whole
    rule rather than half of a table. Expansion never crosses a ``doc_id``.
    """
    meta = index.meta[doc_index]
    if meta.get("part_count", 1) <= 1:
        return meta["text"], []

    siblings = [
        j for j in index._by_doc[meta["doc_id"]]
        if index.meta[j]["section"] == meta["section"]
        and index.meta[j].get("part_count", 1) == meta["part_count"]
    ]
    siblings.sort(key=lambda j: index.meta[j].get("part", 1))
    if len(siblings) <= 1:
        return meta["text"], []

    merged: list[str] = []
    for j in siblings:
        text = index.meta[j]["text"]
        # Overlap means consecutive parts share a tail/head; drop the duplicate.
        if merged and text.startswith(merged[-1][-120:]):
            text = text[len(merged[-1][-120:]):]
        merged.append(text)
    return "\n\n".join(merged).strip(), [
        index.meta[j]["id"] for j in siblings if j != doc_index
    ]


def search(
    query: str,
    k: int = 6,
    doc_id: str | None = None,
    expand: bool = True,
) -> SearchResult:
    """Hybrid search. Returns at most ``k`` hits plus a groundedness verdict."""
    query = (query or "").strip()
    if not query:
        return SearchResult(query, [], False, 0.0, "empty query")

    index = _Index.get()

    allowed: set[int] | None = None
    if doc_id:
        if doc_id not in index._by_doc:
            known = ", ".join(sorted(index._by_doc))
            return SearchResult(query, [], False, 0.0, f"unknown doc_id {doc_id!r}; known: {known}")
        allowed = set(index._by_doc[doc_id])

    dense, best_similarity = _dense_rank(index, query, allowed)
    sparse = bm25.score(index.bm25, query, allowed)[:POOL]

    dense_positions = {i: r for r, (i, _) in enumerate(dense, start=1)}
    sparse_positions = {i: r for r, (i, _) in enumerate(sparse, start=1)}
    fused = _rrf([i for i, _ in dense], [i for i, _ in sparse])

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    hits: list[Hit] = []
    for doc_index, score in ordered:
        meta = index.meta[doc_index]
        text, expanded_from = (_expand(index, doc_index) if expand else (meta["text"], []))
        hits.append(
            Hit(
                chunk_id=meta["id"],
                doc_id=meta["doc_id"],
                doc_title=meta["doc_title"],
                section=meta["section"],
                citation=meta["citation"],
                text=text,
                score=round(score, 6),
                dense_rank=dense_positions.get(doc_index),
                bm25_rank=sparse_positions.get(doc_index),
                source_file=meta.get("source_file", ""),
                source_format=meta.get("source_format", ""),
                effective_date=meta.get("effective_date", ""),
                version=meta.get("version", ""),
                expanded_from=expanded_from,
            )
        )

    grounded = bool(hits) and best_similarity >= MIN_DENSE_SIMILARITY
    reason = ""
    if not hits:
        reason = "no passage matched the query"
    elif not grounded:
        reason = (
            f"best passage similarity {best_similarity:.3f} is below the "
            f"{MIN_DENSE_SIMILARITY} grounding threshold; the corpus does not "
            f"appear to cover this question"
        )
    return SearchResult(query, hits, grounded, best_similarity, reason)


def get_section(doc_id: str, section: str) -> list[Hit]:
    """Fetch a named or numbered section verbatim, without ranking.

    ``section`` matches either the section number (``"4.4"``) or a
    case-insensitive substring of the heading.
    """
    index = _Index.get()
    if doc_id not in index._by_doc:
        return []

    wanted = (section or "").strip().lstrip("§").lower()
    matches: list[int] = []
    for i in index._by_doc[doc_id]:
        meta = index.meta[i]
        number = (meta.get("section_number") or "").lower()
        title = (meta.get("section") or "").lower()
        if not wanted or number == wanted or (wanted and wanted in title):
            matches.append(i)

    matches.sort(key=lambda i: index.meta[i].get("ordinal", 0))
    seen: set[str] = set()
    hits: list[Hit] = []
    for i in matches:
        meta = index.meta[i]
        if meta["id"] in seen:
            continue
        text, expanded_from = _expand(index, i)
        seen.update(expanded_from)
        seen.add(meta["id"])
        hits.append(
            Hit(
                chunk_id=meta["id"],
                doc_id=meta["doc_id"],
                doc_title=meta["doc_title"],
                section=meta["section"],
                citation=meta["citation"],
                text=text,
                score=1.0,
                source_file=meta.get("source_file", ""),
                source_format=meta.get("source_format", ""),
                effective_date=meta.get("effective_date", ""),
                version=meta.get("version", ""),
                expanded_from=expanded_from,
            )
        )
    return hits


def list_documents() -> list[dict]:
    """Catalogue of indexed documents, for tool discovery and /health."""
    index = _Index.get()
    out: list[dict] = []
    for doc_id, indices in sorted(index._by_doc.items()):
        first = index.meta[indices[0]]
        out.append({
            "doc_id": doc_id,
            "title": first["doc_title"],
            "version": first.get("version", ""),
            "effective_date": first.get("effective_date", ""),
            "source_file": first.get("source_file", ""),
            "source_format": first.get("source_format", ""),
            "chunks": len(indices),
            "sections": sorted({index.meta[i]["section"] for i in indices}),
        })
    return out


def index_info() -> dict:
    return dict(_Index.get().info)


def warm() -> dict:
    """Force the index and the embedding model to load now.

    `_Index` defers the ~164 MB embedder until the first dense query, which
    keeps short-lived CLI processes cheap. The deployed service calls this
    during startup instead, so the cost is paid before the first user request
    rather than during it.
    """
    index = _Index.get()
    index.embed_query("warmup")
    return dict(index.info)
