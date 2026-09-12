"""Build the persisted RAG index.

    python -m rag.ingest.build_index          # build into rag/index/
    python -m rag.ingest.build_index --check  # verify the index matches the corpus

Outputs (all committed to the repository so the deployed free-tier service never
has to embed at boot -- see `deployed.md`):

    rag/index/index_info.json   model, dimensions, chunk config, corpus fingerprint
    rag/index/metadata.jsonl    one JSON record per chunk, with citation metadata
    rag/index/embeddings.npy    float32 [n_chunks, dim], L2-normalised
    rag/index/bm25.pkl          keyword index built with the same tokenizer

The build is deterministic: documents are parsed in sorted filename order,
chunking is a pure function, and the embedding model is pinned by name and
recorded in ``index_info.json`` together with a SHA-256 fingerprint of the
corpus. ``--check`` re-computes the fingerprint, so CI fails if the corpus is
edited without rebuilding the index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import pickle
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import apply_seeds, settings  # noqa: E402
from rag.ingest import bm25  # noqa: E402
from rag.ingest.chunk import MAX_TOKENS, OVERLAP_TOKENS, TARGET_TOKENS, chunk_corpus  # noqa: E402
from rag.ingest.parse import SOURCE_PREFIX, SUPPORTED_SUFFIXES, load_corpus  # noqa: E402

EMBED_BATCH = 32

# Fields persisted per chunk. `embed_text` is intentionally dropped: it is
# reconstructible and doubles the on-disk metadata size for no retrieval value.
_PERSIST_FIELDS = (
    "id", "doc_id", "doc_title", "version", "effective_date", "owner",
    "applies_to", "related", "source_file", "source_format",
    "heading_path", "section", "section_number", "part", "part_count",
    "text", "token_estimate", "citation", "ordinal",
)


def corpus_fingerprint(corpus_dir: pathlib.Path) -> str:
    """Stable SHA-256 over the ingestable corpus files."""
    digest = hashlib.sha256()
    paths = sorted(
        p for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not p.name.startswith(SOURCE_PREFIX)
    )
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _embedder(model_name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


def build(index_dir: pathlib.Path | None = None) -> dict:
    apply_seeds()
    index_dir = index_dir or settings.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    docs = load_corpus(settings.corpus_dir)
    chunks = chunk_corpus(docs)
    if not chunks:
        raise RuntimeError("no chunks produced; is corpus/ empty?")

    print(f"parsed   {len(docs)} documents")
    print(f"chunked  {len(chunks)} chunks "
          f"(target={TARGET_TOKENS} max={MAX_TOKENS} overlap={OVERLAP_TOKENS} tokens)")

    model = _embedder(settings.embed_model)
    texts = [c["embed_text"] for c in chunks]
    vectors = np.asarray(list(model.embed(texts, batch_size=EMBED_BATCH)), dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms
    print(f"embedded {vectors.shape[0]} chunks -> {vectors.shape[1]} dimensions")

    keyword_index = bm25.build([c["text"] for c in chunks])
    print(f"bm25     {len(keyword_index['postings']):,} unique terms")

    np.save(index_dir / "embeddings.npy", vectors)
    with (index_dir / "bm25.pkl").open("wb") as fh:
        pickle.dump(keyword_index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            record = {k: chunk[k] for k in _PERSIST_FIELDS}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    info = {
        "model": settings.embed_model,
        "dimensions": int(vectors.shape[1]),
        "chunks": len(chunks),
        "documents": len(docs),
        "doc_ids": [d.doc_id for d in docs],
        "source_formats": sorted({d.source_format for d in docs}),
        "chunking": {
            "strategy": "heading_aware_with_sentence_aligned_overlap",
            "target_tokens": TARGET_TOKENS,
            "max_tokens": MAX_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
        },
        "corpus_fingerprint": corpus_fingerprint(settings.corpus_dir),
        "seed": settings.seed,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.perf_counter() - started, 2),
    }
    (index_dir / "index_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote    {index_dir.relative_to(ROOT)} in {info['build_seconds']}s")
    return info


def check(index_dir: pathlib.Path | None = None) -> int:
    """Verify the persisted index is present, complete, and current."""
    index_dir = index_dir or settings.index_dir
    info_path = index_dir / "index_info.json"
    if not info_path.is_file():
        print("FAIL: no index found; run `python -m rag.ingest.build_index`", file=sys.stderr)
        return 1

    info = json.loads(info_path.read_text(encoding="utf-8"))
    problems: list[str] = []

    for name in ("embeddings.npy", "metadata.jsonl", "bm25.pkl"):
        if not (index_dir / name).is_file():
            problems.append(f"missing {name}")

    if not problems:
        vectors = np.load(index_dir / "embeddings.npy")
        n_meta = sum(1 for _ in (index_dir / "metadata.jsonl").open(encoding="utf-8"))
        if vectors.shape[0] != info["chunks"]:
            problems.append(f"embeddings rows {vectors.shape[0]} != info chunks {info['chunks']}")
        if n_meta != info["chunks"]:
            problems.append(f"metadata rows {n_meta} != info chunks {info['chunks']}")
        if info["model"] != settings.embed_model:
            problems.append(f"index model {info['model']} != configured {settings.embed_model}")

    current = corpus_fingerprint(settings.corpus_dir)
    if current != info.get("corpus_fingerprint"):
        problems.append("corpus has changed since the index was built; rebuild required")

    if problems:
        for problem in problems:
            print(f"FAIL: {problem}", file=sys.stderr)
        return 1

    print(
        f"OK: {info['chunks']} chunks from {info['documents']} documents "
        f"({', '.join(info['source_formats'])}), model {info['model']}, "
        f"built {info['built_at']}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify the policy RAG index.")
    parser.add_argument("--check", action="store_true", help="verify without rebuilding")
    args = parser.parse_args()

    if args.check:
        return check()
    build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
