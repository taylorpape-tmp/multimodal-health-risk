# aws_ecr.tf — private container registry for the FastAPI serving image built by
# the deploy workflow (.github/workflows/deploy.yml).

resource "aws_ecr_repository" "serving" {
  name                 = "${var.name_prefix}-serving"
  image_tag_mutability = "MUTABLE" # allow :latest to move; pin by digest in prod

  image_scanning_configuration {
    scan_on_push = true # free basic CVE scan on every push
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  force_delete = true # destroy removes the repo even if images remain (demo-friendly)
}

# Keep only the last 10 images so old CI builds don't accumulate storage cost.
resource "aws_ecr_lifecycle_policy" "serving" {
  repository = aws_ecr_repository.serving.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}
