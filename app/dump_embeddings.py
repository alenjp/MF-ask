"""
Audit dump: writes every stored chunk AND its 384-dim embedding to
data/chunks_with_embeddings.txt.

Reads from the persisted Chroma collection (data/chroma/) so the numbers shown
are exactly what retrieval uses, not a re-embed. Vectors are rounded to 4
decimals for readability; the stored vectors are full float32.

Run:  python app/dump_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest import get_client  # noqa: E402
from config import CHROMA_DIR, CHUNKS_TXT, COLLECTION, EMBED_MODEL  # noqa: E402

TARGET = CHROMA_DIR.parent / "chunks_with_embeddings.txt"
DECIMALS = 4


def main() -> int:
    n_chunks = (CHUNKS_TXT.read_text(encoding="utf-8").count("chunk_id : ")
                if CHUNKS_TXT.exists() else 0)

    client = get_client()
    try:
        collection = client.get_collection(COLLECTION)
    except Exception:
        print("[dump] collection not found - run `python app/ingest.py` first",
              file=sys.stderr)
        return 1

    got = collection.get(
        include=["documents", "metadatas", "embeddings"],
        limit=collection.count(),
    )
    docs, metas, vecs = got["documents"], got["metadatas"], got["embeddings"]
    ids = got["ids"]

    lines = [
        "=" * 78,
        "RAG CHUNK + EMBEDDING DUMP - HDFC Mutual Fund FAQ corpus",
        f"chunks: {len(docs)}   model: {EMBED_MODEL}",
        f"vector dim: {len(vecs[0])}   decimals rounded: {DECIMALS}",
        f"source chunks.txt also at: {CHUNKS_TXT}",
        "=" * 78,
        "",
    ]
    for i, cid in enumerate(ids):
        m = metas[i] or {}
        vec = "  ".join(f"{v:.{DECIMALS}f}" for v in vecs[i])
        lines += [
            "-" * 78,
            f"chunk_id : {cid}",
            f"scheme   : {m.get('scheme', '')}",
            f"section  : {m.get('section', '')}",
            f"url      : {m.get('source_url', '')}",
            f"doc      : {docs[i][:100]}{'...' if len(docs[i]) > 100 else ''}",
            "-" * 78,
            "embedding[384]:",
            vec,
            "",
        ]

    TARGET.write_text("\n".join(lines), encoding="utf-8")
    print(f"[dump] wrote {len(docs)} chunks + embeddings -> {TARGET}")
    print(f"[dump] (chunks.txt holds {n_chunks} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())