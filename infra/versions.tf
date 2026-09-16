#versions.tf, Terraform + provider version constraints for the HURDLE cloud layer.
#
#Pins keep `terraform init` reproducible: the same provider builds resolve on
#any machine that runs this code. Bump deliberately, not implicitly.

terraform {
  required_version = ">= 1.6.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.40"
    }
  }

  #Remote state (recommended before you run this with a team or from CI).
  #
  #Local state (the default) is fine for a solo first run. Once more than one
  #person or CI touches the infra, move state to a versioned, locked backend so
  #two applies can't corrupt each other.
  #
  #To enable: create the bucket + DynamoDB lock table ONCE (by hand or with a
  #tiny bootstrap config), then uncomment and fill in real names, and run
  #`terraform init -migrate-state`. Do NOT hardcode a bucket that doesn't exist.
  #backend "s3" {
  #bucket         = "REPLACE-ME-hurdle-tfstate"   # a bucket you created first
  #key            = "hurdle/infra/terraform.tfstate"
  #region         = "eu-west-2"
  #dynamodb_table = "REPLACE-ME-hurdle-tf-lock"   # for state locking
  #encrypt        = true
  #}
}
