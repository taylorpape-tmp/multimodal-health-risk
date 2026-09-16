# aws_gpu_training.tf — the RETFound (ViT-L) GPU training path.
#
# Strategy: a LAUNCH TEMPLATE that requests a g4dn/g5 SPOT instance from the
# AWS Deep Learning AMI (CUDA + PyTorch preinstalled). You launch it on demand
# for a fine-tuning run and terminate it after — you are NOT running a GPU 24/7.
# CPU-tier HURDLE models train locally; only RETFound needs this.
#
# Why a launch template and not an aws_instance: it keeps the expensive resource
# OFF by default. `terraform apply` creates the template (free); you spin up the
# instance from it with one AWS CLI command (in the README) only when training,
# and terminate it the moment the run finishes. No idle GPU cost.

# Look up the current Deep Learning AMI (Ubuntu 22.04, PyTorch) owned by Amazon,
# so the template always points at a maintained, GPU-ready image.
data "aws_ami" "dlami" {
  count       = var.enable_gpu_training ? 1 : 0
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch*Ubuntu 22.04*"]
  }
  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Resolve the VPC to place the security group in: explicit var, else default VPC.
data "aws_vpc" "default" {
  count   = var.enable_gpu_training && var.training_vpc_id == "" ? 1 : 0
  default = true
}

locals {
  training_vpc_id = var.enable_gpu_training ? (
    var.training_vpc_id != "" ? var.training_vpc_id : data.aws_vpc.default[0].id
  ) : ""
}

# Security group: egress open (to reach S3/ECR/pip), ingress SSH only from your
# IP. Prefer SSM Session Manager (attached in IAM) and leave SSH closed.
resource "aws_security_group" "training" {
  count       = var.enable_gpu_training ? 1 : 0
  name        = "${var.name_prefix}-training-sg"
  description = "HURDLE GPU training: SSH from operator IP only, all egress."
  vpc_id      = local.training_vpc_id

  ingress {
    description = "SSH from operator IP (override ssh_ingress_cidr to YOUR.IP/32)"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.ssh_ingress_cidr]
  }

  egress {
    description = "All outbound (S3, ECR, package registries)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# The launch template: spot GPU instance, our IAM profile, our SG, a root volume
# big enough for the DLAMI + RETFound weights + a retinal image shard.
resource "aws_launch_template" "training" {
  count       = var.enable_gpu_training ? 1 : 0
  name        = "${var.name_prefix}-training"
  description = "RETFound ViT-L spot training. Launch on demand, terminate after the run."

  image_id      = data.aws_ami.dlami[0].id
  instance_type = var.gpu_instance_type

  iam_instance_profile {
    arn = aws_iam_instance_profile.training[0].arn
  }

  vpc_security_group_ids = [aws_security_group.training[0].id]

  instance_market_options {
    market_type = "spot"
    spot_options {
      # Empty max_price => AWS caps at the on-demand price (recommended).
      max_price                      = var.gpu_spot_max_price != "" ? var.gpu_spot_max_price : null
      spot_instance_type             = "one-time"
      instance_interruption_behavior = "terminate"
    }
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = 120 # GB: DLAMI ~50GB + weights + a data shard
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted             = true
    }
  }

  # Bootstrap: point the run at the data bucket. Your actual training command
  # (git clone, pip install -e ., python -m hurdle.train_retfound ...) goes here
  # or is run interactively over SSM. Kept minimal so this stays a template.
  user_data = base64encode(<<-EOT
    #!/bin/bash
    set -euo pipefail
    echo "HURDLE_DATA_BUCKET=${aws_s3_bucket.data.bucket}" >> /etc/environment
    echo "RETFound training host ready. Data bucket: ${aws_s3_bucket.data.bucket}"
  EOT
  )

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name = "${var.name_prefix}-retfound-training"
      Role = "gpu-training"
    }
  }
}
