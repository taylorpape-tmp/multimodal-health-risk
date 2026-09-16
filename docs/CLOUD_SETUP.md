# HURDLE cloud setup — file manifest & next commands

This document lists every file added for the cloud infrastructure layer (Phase G)
and the exact commands you run to stand it up. **Nothing was provisioned** — this
is code you run yourself with your own AWS + GCP accounts.

## Files created

### Terraform (`infra/`)
- `infra/versions.tf` — Terraform + pinned provider versions (AWS `~>5.60`, Google `~>5.40`); commented S3 remote-state backend
- `infra/providers.tf` — AWS + Google provider config (credential chains, default tags); no keys in code
- `infra/variables.tf` — all inputs with validation (account ID, name prefix)
- `infra/terraform.tfvars.example` — template; copy to `terraform.tfvars` (gitignored) and fill in your IDs
- `infra/aws_s3.tf` — private, versioned, AES256-encrypted data/checkpoint bucket + lifecycle rules
- `infra/aws_ecr.tf` — private ECR repo for the serving image + keep-last-10 lifecycle + scan-on-push
- `infra/aws_iam_training.tf` — least-privilege GPU-training role/instance-profile (scoped S3 + ECR pull + SSM)
- `infra/aws_gpu_training.tf` — Deep Learning AMI lookup, security group, and RETFound **spot launch template**
- `infra/aws_iam_cicd.tf` — GitHub OIDC provider + CI deploy role (ECR push only, no long-lived keys)
- `infra/gcp.tf` — BigQuery dataset + example omics schema, and a private versioned GCS bucket
- `infra/outputs.tf` — bucket names, ECR URL, CI role ARN, launch-template ID
- `infra/README.md` — full runbook (auth, init/plan/apply/destroy, GPU run, cost estimate, ML wiring)
- `infra/.gitignore` — keeps state files and real `terraform.tfvars` out of git

### Serving image
- `Dockerfile` — multi-stage (Python 3.11-slim builder + runtime), installs `src/hurdle`, non-root, serves `hurdle.serving.app:app` on 8080
- `.dockerignore` — keeps data, git, infra, tests out of the build context
- `requirements-serving.txt` — pinned runtime deps for the API image (FastAPI, uvicorn, sklearn, boto3)
- `src/hurdle/serving/__init__.py` — serving subpackage
- `src/hurdle/serving/app.py` — FastAPI app: `/health`, `/`, `/predict`; degrades gracefully with no checkpoint

### CI/CD
- `.github/workflows/deploy.yml` — builds the image and pushes to ECR via GitHub OIDC (separate from test CI)

## Validation status (from the authoring sandbox)

| Check | Result |
|-------|--------|
| `terraform fmt -check` | **PASS** (clean after `terraform fmt`) |
| `terraform init -backend=false` | **PASS** — HCL parses, provider versions resolve, dependency graph builds, lock file written |
| `terraform providers` | **PASS** — resolves `hashicorp/aws ~>5.60` + `hashicorp/google ~>5.40` |
| `terraform validate` | **Not completed in sandbox** — requires launching provider plugin binaries to load schemas; the sandbox blocks the terraform↔plugin gRPC handshake (the plugin binary runs fine standalone). **Run it yourself:** `cd infra && terraform init && terraform validate` |
| Dockerfile | Reviewed manually — base `python:3.11-slim`, `COPY src/ ./src/` + `pip install .` matches the repo's `src/`-layout `pyproject.toml`; entrypoint `hurdle.serving.app:app` matches `src/hurdle/serving/app.py`. Not built (no Docker in sandbox). |

So: formatting and initialization/graph checks passed for real; schema-level
`validate` and the Docker build are the two steps to run locally to fully close
the loop.

## Commands you run next

```bash
# --- Authenticate (your own accounts; nothing stored in the repo) ---
aws configure
gcloud auth application-default login
gcloud config set project <your-gcp-project-id>
gcloud services enable bigquery.googleapis.com storage.googleapis.com

# --- Terraform ---
cd infra
cp terraform.tfvars.example terraform.tfvars      # then edit in your IDs + your IP
terraform init
terraform validate                                # the step pending from the sandbox
terraform plan
terraform apply                                   # creates the resources
terraform output                                  # capture bucket/ECR/role values

# --- Wire CI (GitHub repo variables) ---
#   AWS_REGION         = <your region>
#   AWS_ROLE_TO_ASSUME = terraform output -raw github_actions_role_arn
#   ECR_REPOSITORY     = terraform output -raw ecr_repository_url

# --- Serving image (optional local build/test before CI does it) ---
cd ..
docker build -t hurdle-serving .
docker run -p 8080:8080 hurdle-serving            # then: curl localhost:8080/health

# --- RETFound GPU training run (spot; cents per run) ---
aws ec2 run-instances --launch-template LaunchTemplateId=$(terraform -chdir=infra output -raw training_launch_template_id)
aws ssm start-session --target <instance-id>      # train, push checkpoint to s3://.../models/
aws ec2 terminate-instances --instance-ids <id>   # ALWAYS terminate when done

# --- Tear down to avoid cost ---
cd infra && terraform destroy
```

See `infra/README.md` for the full runbook, GPU cost table, and how each cloud
resource connects to the ML code.
