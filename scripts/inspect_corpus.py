"""Developer utility: show how each corpus document parses and chunks.

    python scripts/inspect_corpus.py            # per-document summary
    python scripts/inspect_corpus.py POL-EXP-001  # dump one document's chunks
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.ingest.parse import load_corpus  # noqa: E402


def main() -> int:
    docs = load_corpus(ROOT / "corpus")
    wanted = sys.argv[1] if len(sys.argv) > 1 else None

    if wanted:
        doc = next((d for d in docs if d.doc_id == wanted), None)
        if doc is None:
            print(f"no such doc_id: {wanted}", file=sys.stderr)
            return 1
        try:
            from rag.ingest.chunk import chunk_corpus
        except ImportError:
            print(doc.markdown)
            return 0
        chunks = [c for c in chunk_corpus(docs) if c["doc_id"] == wanted]
        print(f"{doc.doc_id}  {doc.title}  ({len(chunks)} chunks)\n")
        for c in chunks:
            print(f"--- {c['id']}  [{c['token_estimate']} tok]  {c['citation']}")
            print(c["text"][:400].rstrip())
            print()
        return 0

    header = f"{'doc_id':<18} {'fmt':<5} {'chars':>6} {'h1':>3} {'h2':>3} {'h3':>3}  title"
    print(f"{len(docs)} documents parsed\n")
    print(header)
    print("-" * len(header))
    total = 0
    for d in docs:
        h1 = len(re.findall(r"^# ", d.markdown, re.M))
        h2 = len(re.findall(r"^## ", d.markdown, re.M))
        h3 = len(re.findall(r"^### ", d.markdown, re.M))
        total += len(d.markdown)
        print(
            f"{d.doc_id:<18} {d.source_format:<5} {len(d.markdown):>6} "
            f"{h1:>3} {h2:>3} {h3:>3}  {d.title[:44]}"
        )
        missing = [
            k for k in ("version", "effective_date", "owner") if not getattr(d, k)
        ]
        if missing:
            print(f"   !! missing metadata: {missing}")
    print(f"\ntotal characters: {total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
