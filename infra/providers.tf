# providers.tf — provider configuration.
#
# No credentials live here. The AWS provider reads the standard credential chain
# (env vars / ~/.aws/credentials from `aws configure` / an assumed role); the
# Google provider reads Application Default Credentials from `gcloud auth
# application-default login`. See infra/README.md.

provider "aws" {
  region = var.aws_region

  # Tag everything so cost reports and cleanup can filter by project.
  default_tags {
    tags = {
      Project     = "hurdle"
      ManagedBy   = "terraform"
      Environment = var.environment
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}
