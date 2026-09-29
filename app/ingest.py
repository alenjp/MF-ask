"""
Stages 3 and 4 of ingestion: EMBED -> STORE (ChromaDB, persisted to disk).

Run once:  python app/ingest.py [--rebuild]

Embedding: sentence-transformers/all-MiniLM-L6-v2 (local, 384-dim, no API key).
The same model instance encodes both chunks and user questions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from chunking import build_all_chunks, dump_chunks
from config import CHROMA_DIR, COLLECTION, EMBED_MODEL, RAW_DIR, SOURCES_CSV

import csv


def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def get_client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )


def retrieved_on() -> str:
    with SOURCES_CSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[0]["retrieved_on"] if rows else ""


def ingest(rebuild: bool = False) -> int:
    if not list(RAW_DIR.glob("*.txt")):
        print("[ingest] no raw files - run `python app/fetch_sources.py` first", file=sys.stderr)
        return 1

    chunks = build_all_chunks(retrieved_on())
    if not chunks:
        print("[ingest] chunker produced 0 chunks", file=sys.stderr)
        return 1
    dump_chunks(chunks)
    print(f"[chunk] {len(chunks)} chunks -> data/chunks.txt")

    client = get_client()
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:  # noqa: BLE001
            pass
    try:
        collection = client.get_collection(COLLECTION)
        existing = collection.count()
    except Exception:  # noqa: BLE001
        collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        existing = 0

    if existing >= len(chunks) and not rebuild:
        print(f"[store] chroma already has {existing} chunks - skipping (use --rebuild to redo)")
        return 0

    model = get_model()
    # Every raw chunk repeats the scheme name, which lets "HDFC ... Fund" match
    # dominate the cosine score and drown out the *section* term the user asked
    # about. Prefixing a "<section> — <scheme>:" header restores the
    # question-relevant words to the top of the vector.
    #
    # The SAME prefixed text is stored as the Chroma document, not just the
    # vector, so what the LLM reads matches what we embedded. Feeding it the bare
    # body instead made the model read an unlabelled label/value list ("Nil /
    # Stamp duty on investment: / 0.005% ...") and answer "not stated in the
    # sources" even though the fact was right there.
    docs = [f"{c.section} — {c.scheme}: {c.text}" for c in chunks]
    vectors = model.encode(docs, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=[v.tolist() for v in vectors],
        documents=docs,
        metadatas=[c.to_dict()["metadata"] | {"section": c.section} for c in chunks],
    )
    print(f"[embed] {len(vectors)} vectors, dim={len(vectors[0])} via {EMBED_MODEL}")
    print(f"[store] chroma collection '{COLLECTION}' now holds {collection.count()} chunks")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    raise SystemExit(ingest(rebuild=args.rebuild))
