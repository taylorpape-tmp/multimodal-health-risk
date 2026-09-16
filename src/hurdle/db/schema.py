"""Database schema for the multimodal diabetes-risk store.

Engine-agnostic SQLAlchemy Core so the same schema/ingest/query code runs on local
SQLite and BigQuery: a small star design keyed on subject, with feature tables
(omics, wearable, CGM) in long format so the schema stays stable as the feature set
changes and 'these features for these subjects' is a clean WHERE ... IN query.
"""
from sqlalchemy import Column, Float, ForeignKey, Integer, MetaData, String, Table, UniqueConstraint

metadata = MetaData()

subjects = Table(
    "subjects", metadata,
    Column("subject_id", String, primary_key=True),   #omics Z-code, e.g. ZOZOW1T
    Column("iris", String),                            #IR / IS / Unknown
    Column("sspg", Float),                             #steady-state plasma glucose
    Column("fpg", Float),
    Column("clinical_class", String),                  #Diabetic / prediabetic / ...
    Column("gender", String),
    Column("ethnicity", String),
    Column("age", Float),
    Column("bmi", Float),
)

omics_features = Table(
    "omics_features", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("subject_id", String, ForeignKey("subjects.subject_id"), nullable=False),
    Column("panel", String, nullable=False),           #'S8_sspg' | 'S9_isir'
    Column("analyte", String, nullable=False),
    Column("value", Float),
    UniqueConstraint("subject_id", "panel", "analyte", name="uq_omics_cell"),
)

wearable_features = Table(
    "wearable_features", metadata,
    Column("subject_id", String, primary_key=True),    #wearable SubjectN id
    Column("feature", String, primary_key=True),
    Column("value", Float),
)

cgm_features = Table(
    "cgm_features", metadata,
    Column("subject_id", String, primary_key=True),    #CGM id, e.g. 1636-69-001
    Column("feature", String, primary_key=True),
    Column("value", Float),
)

subject_crosswalk = Table(
    "subject_crosswalk", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("omics_subject_id", String),                #Z-code
    Column("cgm_subject_id", String),                  #e.g. 1636-69-001
    Column("site_code", String),                       #e.g. 69-001
    Column("wearable_subject_id", String),             #e.g. Subject12 (usually null)
)

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String, nullable=False),
    Column("subject_id", String, nullable=False),
    Column("target", String, nullable=False),          #'SSPG' | 'IRIS'
    Column("model", String, nullable=False),
    Column("y_true", Float),
    Column("y_pred", Float),
)

metrics = Table(
    "metrics", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String, nullable=False),
    Column("target", String, nullable=False),
    Column("model", String, nullable=False),
    Column("metric", String, nullable=False),          #'R2' | 'AUROC' | ...
    Column("value", Float),
)


def create_all(engine):
    #create every table on the given engine (idempotent)
    metadata.create_all(engine)


def drop_all(engine):
    metadata.drop_all(engine)
