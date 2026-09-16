# aws_iam_training.tf — least-privilege IAM for the GPU training instance.
#
# The training EC2 instance assumes this role (via an instance profile) so it can
# read data and write checkpoints to exactly one bucket and pull the serving
# base image from exactly one ECR repo — nothing else. No user keys on the box.

data "aws_iam_policy_document" "training_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "training" {
  count              = var.enable_gpu_training ? 1 : 0
  name               = "${var.name_prefix}-training-role"
  assume_role_policy = data.aws_iam_policy_document.training_assume.json
}

# Scope: list the bucket, read/write only under data/ and models/, pull from the
# one ECR repo. Least privilege — no s3:* and no wildcard resource.
data "aws_iam_policy_document" "training_permissions" {
  statement {
    sid       = "ListDataBucket"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
  }
  statement {
    sid     = "ReadWriteDataAndModels"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = [
      "${aws_s3_bucket.data.arn}/data/*",
      "${aws_s3_bucket.data.arn}/models/*",
    ]
  }
  statement {
    sid       = "EcrAuthToken"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"] # this action does not support resource scoping
  }
  statement {
    sid = "EcrPull"
    actions = [
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchCheckLayerAvailability",
    ]
    resources = [aws_ecr_repository.serving.arn]
  }
}

resource "aws_iam_role_policy" "training" {
  count  = var.enable_gpu_training ? 1 : 0
  name   = "${var.name_prefix}-training-policy"
  role   = aws_iam_role.training[0].id
  policy = data.aws_iam_policy_document.training_permissions.json
}

# Attach SSM so you can open a shell via Session Manager without opening SSH.
resource "aws_iam_role_policy_attachment" "training_ssm" {
  count      = var.enable_gpu_training ? 1 : 0
  role       = aws_iam_role.training[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "training" {
  count = var.enable_gpu_training ? 1 : 0
  name  = "${var.name_prefix}-training-profile"
  role  = aws_iam_role.training[0].name
}
