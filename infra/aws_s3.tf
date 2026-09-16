# aws_s3.tf — one private, versioned, encrypted bucket for raw data + model
# checkpoints. Raw data stays out of git (see .gitignore); this is where it
# lives in the cloud. RETFound checkpoints written by the GPU training job land
# under models/.

resource "aws_s3_bucket" "data" {
  # Bucket names are globally unique; account ID suffix avoids collisions.
  bucket        = "${var.name_prefix}-data-${var.aws_account_id}"
  force_destroy = var.s3_force_destroy
}

# Keep old versions of checkpoints/data so an overwrite is never a silent loss.
resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Encrypt at rest. SSE-S3 (AES256) is free and needs no key management; switch
# to aws:kms if you need audited, rot-able keys.
resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# Private: block every form of public access.
resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle: expire noncurrent versions after 90 days and abort stale multipart
# uploads, so old checkpoints don't accrue storage cost forever.
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}
