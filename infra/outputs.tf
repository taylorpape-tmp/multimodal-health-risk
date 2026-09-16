# outputs.tf — values the ML code and the runbook need after apply.

# ---------- AWS ----------

output "s3_data_bucket" {
  description = "S3 bucket for raw data + model checkpoints. Set HURDLE_DATA_BUCKET to this."
  value       = aws_s3_bucket.data.bucket
}

output "s3_data_bucket_arn" {
  description = "ARN of the data bucket."
  value       = aws_s3_bucket.data.arn
}

output "ecr_repository_url" {
  description = "ECR repo URL to push/pull the serving image. Used by the deploy workflow."
  value       = aws_ecr_repository.serving.repository_url
}

output "github_actions_role_arn" {
  description = "IAM role ARN the deploy workflow assumes via OIDC. Put in the workflow's role-to-assume."
  value       = aws_iam_role.github_actions_deploy.arn
}

output "training_launch_template_id" {
  description = "Launch template for the RETFound GPU spot instance (null if GPU path disabled)."
  value       = var.enable_gpu_training ? aws_launch_template.training[0].id : null
}

output "training_instance_profile" {
  description = "IAM instance profile name for the training box (null if disabled)."
  value       = var.enable_gpu_training ? aws_iam_instance_profile.training[0].name : null
}

# ---------- GCP ----------

output "bigquery_dataset_id" {
  description = "BigQuery dataset holding the structured/omics tables."
  value       = google_bigquery_dataset.hurdle.dataset_id
}

output "gcs_data_bucket" {
  description = "GCS bucket for data/checkpoint staging on the Google side."
  value       = google_storage_bucket.data.name
}
