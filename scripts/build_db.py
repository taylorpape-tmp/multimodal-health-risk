"""Build the local SQLite store (data/hurdle.db) from the cleaned interim files.

Rebuilds from scratch (drop, create, ingest). The db is gitignored; this is the reproducible
recipe that recreates it.

    python scripts/build_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hurdle.db import ingest, queries, schema  #noqa: E402


def main():
    engine = ingest.get_engine()
    schema.drop_all(engine)
    schema.create_all(engine)

    stats = ingest.ingest_omics(engine)
    print(f"omics: {stats['subjects']} subjects, {stats['omics_cells']} feature cells")
    print(f"total subjects in db: {queries.subject_count(engine)}")

    for panel in ("S8_sspg", "S9_isir"):
        m = queries.omics_matrix(engine, panel)
        print(f"  {panel}: matrix {m.shape[0]} subjects x {m.shape[1] - 1} analytes (+label)")


if __name__ == "__main__":
    main()
