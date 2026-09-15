# SQL layer

The project stores every cleaned modality in one relational database rather than
loose CSVs. This is deliberate: the person specification lists database / SQL
skills as essential, and a normalized store is genuinely the right tool for
joining four modalities keyed on subject.

## Engine

`src/hurdle/db/ingest.py::get_engine()` returns a SQLAlchemy engine. It defaults
to a local SQLite file (`sqlite:///data/hurdle.db`) but reads `HURDLE_DB_URL`, so
the identical ingest and query code runs against BigQuery in the cloud
(`bigquery://<project>/<dataset>`) — the GCP half of the cloud demonstration.
SQLAlchemy Core (not the ORM) keeps the SQL explicit and portable.

## Schema

Defined in `src/hurdle/db/schema.py`. A small star-ish design keyed on subject:

| table | grain | source |
|---|---|---|
| `subjects` | one row per iPOP patient | S1 labels + S8/S9 (IRIS, SSPG, demographics) |
| `omics_features` | subject × panel × analyte (long) | cleaned S8 / S9 matrices |
| `wearable_features` | subject × feature (long) | wearable feature builder |
| `cgm_features` | subject × feature (long) | CGM variability builder |
| `subject_crosswalk` | id map across studies | computed 22-patient overlap |
| `predictions` | subject × target × model | model runs |
| `metrics` | run × target × model × metric | model runs |

The feature tables are **long** (subject, feature, value) rather than one column
per analyte. That keeps the schema stable when the feature set changes and turns
"give me these analytes for these subjects" into a clean `WHERE ... IN` query.

The **crosswalk** table is the honest core of the fusion design: it encodes that
CGM id `1636-69-001` maps to omics Z-code `ZOZOW1T` via site code `69-001`, so a
JOIN assembles the real 22-patient overlap cohort. Subjects with no cross-study
id are filled by the virtual-cohort layer, disclosed as such.

## Build it

```bash
cd ~/Desktop/resume/HURDLE
python scripts/build_db.py     # drop + create + ingest -> data/hurdle.db
```

The `.db` file is gitignored; `build_db.py` is the reproducible recipe. On the
real cleaned data this materializes 60 subjects and 14,135 omics feature cells.

## Query it

`src/hurdle/db/queries.py` holds the JOINs that turn the normalized store back
into modeling matrices — the payoff of normalizing in the first place:

```python
from hurdle.db import ingest, queries
engine = ingest.get_engine()

# pivot one omics panel back to a subjects x analytes matrix with its label
s9 = queries.omics_matrix(engine, "S9_isir")   # 60 x (IRIS + 152 analytes)
s8 = queries.omics_matrix(engine, "S8_sspg")   # 59 x (SSPG + 85 analytes)

# the honest real-overlap cohort (omics AND CGM), joined through the crosswalk
overlap = queries.overlap_cohort(engine)

# a long feature table pivoted wide, optionally restricted to a subject set
wear = queries.feature_wide(engine, "wearable_features")
```

The matrices come straight out of SQL ready to hand to an `MLModel` or to
`run_nested_consensus`.

## Tests

`tests/test_db.py` runs on an in-memory SQLite engine (no disk), covering schema
creation, omics ingest, the pivot round-trip, the SSPG/IRIS label join, the
crosswalk overlap JOIN, and a feature-table round-trip.
