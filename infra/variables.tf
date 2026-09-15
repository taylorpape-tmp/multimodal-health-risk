# variables.tf — all inputs. Real values go in terraform.tfvars (gitignored);
# a committed terraform.tfvars.example shows the shape. No secrets or account
# IDs are hardcoded anywhere in this repo.

# ---------- Global ----------

variable "environment" {
  description = "Deployment environment label (dev | staging | prod). Tags + name suffixes."
  type        = string
  default     = "dev"
}

variable "name_prefix" {
  description = "Prefix for resource names. Keep short; must be S3/GCS-name-safe (lowercase, hyphens)."
  type        = string
  default     = "hurdle"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,20}$", var.name_prefix))
    error_message = "name_prefix must be lowercase alphanumeric/hyphen, 2-21 chars, starting with a letter."
  }
}

# ---------- AWS ----------

variable "aws_region" {
  description = "AWS region for S3, ECR, IAM, and GPU training. us-east-1 = N. Virginia (cheapest, widest GPU availability)."
  type        = string
  default     = "us-east-1"
}

variable "aws_account_id" {
  description = "Your 12-digit AWS account ID. Used to scope IAM trust + ECR ARNs. Placeholder only in .example."
  type        = string

  validation {
    condition     = can(regex("^[0-9]{12}$", var.aws_account_id))
    error_message = "aws_account_id must be exactly 12 digits."
  }
}

variable "s3_force_destroy" {
  description = "If true, `terraform destroy` empties the bucket first. Convenient for a portfolio/demo; set false for anything you care about."
  type        = bool
  default     = true
}

# ---- GPU training path (RETFound ViT-L on a spot instance) ----

variable "enable_gpu_training" {
  description = "Create the training IAM instance profile + security group + launch template. RETFound trains here; CPU-tier models run locally."
  type        = bool
  default     = true
}

variable "gpu_instance_type" {
  description = "GPU instance for RETFound ViT-L fine-tuning. g4dn.xlarge (T4, 16GB) is the cheapest that fits; g5.xlarge (A10G, 24GB) is faster."
  type        = string
  default     = "g4dn.xlarge"
}

variable "gpu_spot_max_price" {
  description = "Max USD/hour you'll pay for the spot instance. Leave empty to cap at the on-demand price. e.g. \"0.40\"."
  type        = string
  default     = ""
}

variable "training_vpc_id" {
  description = "VPC to place the training security group in. Empty string => use the account's default VPC (looked up)."
  type        = string
  default     = ""
}

variable "ssh_ingress_cidr" {
  description = "CIDR allowed to SSH to the training instance. Set to YOUR.IP/32. Default is a deny-by-nothing placeholder you MUST override."
  type        = string
  default     = "127.0.0.1/32"
}

# ---------- CI/CD (GitHub OIDC -> AWS) ----------

variable "github_owner" {
  description = "GitHub org/user that owns the repo, e.g. \"taylorpape\". Scopes the OIDC trust policy."
  type        = string
  default     = ""
}

variable "github_repo" {
  description = "Repository name, e.g. \"HURDLE\". The deploy workflow assumes the CI role only for this repo."
  type        = string
  default     = ""
}

variable "create_github_oidc_provider" {
  description = "Create the GitHub OIDC provider in IAM. Set false if your account already has token.actions.githubusercontent.com registered (only one per account)."
  type        = bool
  default     = true
}

# ---------- GCP ----------

variable "gcp_project_id" {
  description = "Your GCP project ID (not the number). Placeholder only in .example."
  type        = string
}

variable "gcp_region" {
  description = "GCP region for the GCS bucket. europe-west2 = London."
  type        = string
  default     = "europe-west2"
}

variable "bigquery_location" {
  description = "BigQuery dataset location. Multi-region \"EU\" or a region like \"europe-west2\". Cannot be changed after creation."
  type        = string
  default     = "EU"
}
