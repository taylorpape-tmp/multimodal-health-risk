#gcp.tf, the GCP half of the "AWS + GCP" competence.
#
#BigQuery holds the structured/omics tables (the SQL analytics layer); a GCS
#bucket mirrors AWS S3 for data/checkpoint staging on the Google side. This
#demonstrates cross-cloud fluency without duplicating the whole training path.

#BigQuery

resource "google_bigquery_dataset" "hurdle" {
  dataset_id  = "${replace(var.name_prefix, "-", "_")}_omics"
  description = "HURDLE structured + omics tables (proteomics, metabolomics, clinical labels)."
  location    = var.bigquery_location

  #Guardrail: don't let `terraform destroy` silently drop tables with data.
  #Flip to true only for a throwaway demo dataset.
  delete_contents_on_destroy = false

  labels = {
    project     = "hurdle"
    managed_by  = "terraform"
    environment = var.environment
  }
}

#An example table so the dataset is not empty and the schema is version-
#controlled. Real omics tables get loaded via `bq load` from the GCS bucket.
resource "google_bigquery_table" "omics_features" {
  dataset_id          = google_bigquery_dataset.hurdle.dataset_id
  table_id            = "omics_features"
  description         = "Wide feature table: one row per subject, omics + clinical columns."
  deletion_protection = false #example table; set true for real data

  schema = jsonencode([
    { name = "subject_id", type = "STRING", mode = "REQUIRED", description = "De-identified subject key" },
    { name = "sspg", type = "FLOAT", mode = "NULLABLE", description = "Steady-state plasma glucose (insulin resistance target)" },
    { name = "iris_label", type = "INTEGER", mode = "NULLABLE", description = "Insulin-resistant (1) vs sensitive (0)" },
    { name = "modality", type = "STRING", mode = "NULLABLE", description = "omics | wearable | cgm | imaging" },
    { name = "ingested_at", type = "TIMESTAMP", mode = "NULLABLE", description = "Load timestamp" },
  ])
}

#GCS

resource "google_storage_bucket" "data" {
  name     = "${var.name_prefix}-data-${var.gcp_project_id}"
  location = var.gcp_region

  #Private by default; enforce uniform bucket-level IAM (no per-object ACLs).
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  #Expire noncurrent versions after 90 days to bound storage cost.
  lifecycle_rule {
    condition {
      days_since_noncurrent_time = 90
    }
    action {
      type = "Delete"
    }
  }

  force_destroy = var.s3_force_destroy #reuse the same "demo-friendly cleanup" flag
}
