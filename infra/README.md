# HURDLE cloud infrastructure (Terraform)

Infrastructure-as-code for the HURDLE multimodal diabetes-risk portfolio. This
provisions the **cloud half** of the "build locally first, add cloud later"
strategy:

- **AWS** — S3 (raw data + model checkpoints), ECR (serving image), IAM
  (least-privilege training role + GitHub-OIDC CI role), and a **GPU training
  path** (a spot-instance launch template on the Deep Learning AMI) for
  fine-tuning **RETFound (ViT-L)**. CPU-tier HURDLE models train on your laptop;
  only RETFound needs the GPU box.
- **GCP** — a BigQuery dataset for the structured/omics SQL layer and a GCS
  bucket, demonstrating the AWS+GCP competence the role's person-spec marks
  Essential.

> Nothing here is applied for you. You run it, with your own accounts. No
> secrets, account IDs, or keys are committed — everything sensitive is a
> variable supplied via `terraform.tfvars` (gitignored).

---

## Files

| File | What it defines |
|------|-----------------|
| `versions.tf` | Terraform ≥1.6 + pinned AWS `~>5.60` / Google `~>5.40` providers; commented S3 remote-state backend |
| `providers.tf` | AWS + Google provider config (region, default tags). Reads standard credential chains — no keys in code |
| `variables.tf` | All inputs, with validation on account ID / name prefix |
| `terraform.tfvars.example` | Copy to `terraform.tfvars` and fill in your IDs |
| `aws_s3.tf` | Private, versioned, encrypted data/checkpoint bucket + lifecycle |
| `aws_ecr.tf` | Private ECR repo for the serving image + keep-last-10 lifecycle |
| `aws_iam_training.tf` | Least-privilege role/instance-profile for the GPU box (scoped S3 + ECR pull + SSM) |
| `aws_gpu_training.tf` | DLAMI lookup, security group, and the **spot launch template** for RETFound |
| `aws_iam_cicd.tf` | GitHub OIDC provider + CI deploy role (ECR push only, no long-lived keys) |
| `gcp.tf` | BigQuery dataset + example schema, and a private versioned GCS bucket |
| `outputs.tf` | Bucket names, ECR URL, CI role ARN, launch-template ID — wired into the ML code + CI |

---

## Prerequisites (install once)

- **Terraform** ≥ 1.6 — <https://developer.hashicorp.com/terraform/downloads>
- **AWS CLI v2** — `aws configure` (or SSO) with a profile that can create IAM/S3/ECR/EC2
- **gcloud CLI** — `gcloud auth application-default login` and an existing GCP project with billing enabled
- Enable the required GCP APIs once:
  ```bash
  gcloud services enable bigquery.googleapis.com storage.googleapis.com
  ```

---

## Step-by-step

```bash
# 0. Authenticate (uses YOUR credentials; nothing is stored in this repo)
aws configure                                   # or: aws sso login --profile <p>
gcloud auth application-default login
gcloud config set project <your-gcp-project-id>

# 1. Enter the infra dir and set your values
cd infra
cp terraform.tfvars.example terraform.tfvars
$EDITOR terraform.tfvars                         # fill in aws_account_id, gcp_project_id, github_owner, your IP, ...

# 2. Initialise providers (downloads AWS + Google plugins)
terraform init

# 3. Review the plan — READ IT before applying
terraform plan

# 4. Create the resources
terraform apply                                 # type "yes" to confirm

# 5. Capture the outputs (used by CI + the ML code)
terraform output
```

After `apply`, wire the outputs in:

- Set env var `HURDLE_DATA_BUCKET` = `terraform output -raw s3_data_bucket` for
  training/serving code that reads/writes S3.
- In GitHub → **Settings → Secrets and variables → Actions → Variables**, set
  repository variables used by `.github/workflows/deploy.yml`:
  - `AWS_REGION` = your region
  - `AWS_ROLE_TO_ASSUME` = `terraform output -raw github_actions_role_arn`
  - `ECR_REPOSITORY` = `terraform output -raw ecr_repository_url`

### Tear everything down (avoid ongoing cost)

```bash
cd infra
terraform destroy                               # type "yes"
```

`s3_force_destroy`/`force_destroy` default to `true` so the buckets and ECR repo
empty themselves on destroy — convenient for a portfolio demo. Set them `false`
for anything you actually care about keeping.

---

## Running a RETFound GPU training job

`terraform apply` creates the launch **template** (free); it does **not** run a
GPU. You start an instance from the template only when training, and terminate
it the instant the run finishes:

```bash
# Launch one spot GPU instance from the template
# (--region defaults to your AWS CLI profile; pass --region <your-region> to override)
aws ec2 run-instances \
  --launch-template LaunchTemplateId=$(terraform output -raw training_launch_template_id)

# Connect WITHOUT opening SSH, via SSM Session Manager (IAM already allows it):
aws ssm start-session --target <instance-id>

# On the box: clone the repo, install, pull data from S3, fine-tune RETFound,
# push the checkpoint back to s3://$HURDLE_DATA_BUCKET/models/, then:
sudo shutdown -h now      # or: aws ec2 terminate-instances --instance-ids <id>
```

Prefer **SSM Session Manager** over SSH — the IAM role already grants it, so you
can leave `ssh_ingress_cidr` closed. If you do use SSH, set `ssh_ingress_cidr`
to `YOUR.IP/32` and attach a key pair.

### Cost estimate (GPU spot approach)

Spot pricing varies by region/AZ/time; these are representative and let you
sanity-check the order of magnitude — **not** a quote.

| Instance | GPU | ~Spot $/hr | Cost of a ~30–60 min RETFound fine-tune |
|----------|-----|-----------|------------------------------------------|
| `g4dn.xlarge` | T4 16GB | ~$0.15–0.25 | **~$0.08–0.25** (single-digit to low-tens of cents) |
| `g5.xlarge` | A10G 24GB | ~$0.30–0.50 | ~$0.15–0.50 |

So a training run is **cents, not dollars**, provided you terminate the instance
afterward. Storage: S3 Standard is ~$0.023/GB-month; a handful of checkpoints is
pennies. **The one way to be surprised by cost is leaving a GPU instance
running** — always terminate, and run `terraform destroy` when done with the
whole stack.

---

## How this connects to the ML code

- **S3 bucket** (`s3_data_bucket`) — the cloud home for `data/raw` (kept out of
  git by `.gitignore`) and for model checkpoints under `models/`. The GPU
  training job reads data from and writes RETFound weights to this bucket.
- **ECR repo** (`ecr_repository_url`) — holds the serving image built from the
  repo `Dockerfile`, which packages `src/hurdle` and serves
  `hurdle.serving.app:app` (FastAPI) on port 8080.
- **GitHub OIDC role** (`github_actions_role_arn`) — lets
  `.github/workflows/deploy.yml` push that image to ECR with no long-lived keys.
- **BigQuery dataset** (`bigquery_dataset_id`) — the SQL analytics layer for the
  structured/omics feature tables (`bq load` from the GCS bucket).
- **Training IAM role** — least-privilege: the GPU box can touch only this
  bucket's `data/` + `models/` prefixes and pull only this ECR repo.

---

## Remote state (when you outgrow local state)

Local state (default) is fine for a solo first run. Before sharing with a team
or running Terraform from CI, migrate to the S3 backend: create a state bucket +
DynamoDB lock table once, then uncomment and fill the `backend "s3"` block in
`versions.tf` and run `terraform init -migrate-state`. Do not point the backend
at a bucket that doesn't exist yet.

## Notes on validation

`terraform fmt` and `terraform init` were run against this config during
authoring. `terraform validate` (schema-level attribute checking) should be run
locally after `terraform init` — see `docs/CLOUD_SETUP.md` for the exact
validation status from the authoring sandbox.
