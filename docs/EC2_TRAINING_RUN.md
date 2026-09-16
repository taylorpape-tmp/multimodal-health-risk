# Running the training job on EC2 (real cloud compute)

This runs the retinal-CNN training on a real **EC2 instance**, pulling data from S3 and pushing the
trained model + embeddings back to S3. It completes the end-to-end cloud training loop:
**data in S3 → compute on EC2 → artifacts back to S3.**

Honest framing for the writeup: GPU spot quota was pending on the new account, so this job ran on an
**EC2 CPU instance** (`c5.2xlarge`) to demonstrate the full cloud training pipeline. Swapping to the
`g4dn.xlarge` GPU launch template (already provisioned) to fine-tune RETFound is a one-line instance-type
change once quota clears.

All commands run in **your** terminal (the assistant cannot reach it). Region: `us-east-1`.

---

## 0. Prerequisites (already done)

- S3 bucket `hurdle-data-791973676965` exists and holds `retinamnist_224.npz`, `train_imaging.py`,
  and `src/hurdle/imaging/*` (you uploaded these earlier).
- IAM instance profile `hurdle-training-profile` exists (from `terraform apply`) — gives the instance
  S3 read/write without embedding keys.
- Default VPC + a security group already exist.

## 1. Pick a CPU AMI and launch an instance

```bash
# latest Amazon Linux 2023 AMI (CPU, free-tier-family base)
AMI=$(aws ssm get-parameters --region us-east-1 \
  --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query 'Parameters[0].Value' --output text)
echo "AMI=$AMI"

# launch a c5.2xlarge (8 vCPU) on-demand instance with the training IAM profile
aws ec2 run-instances --region us-east-1 \
  --image-id "$AMI" \
  --instance-type c5.2xlarge \
  --iam-instance-profile Name=hurdle-training-profile \
  --instance-initiated-shutdown-behavior terminate \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hurdle-cpu-training}]' \
  --user-data file://ec2_train_bootstrap.sh \
  --query 'Instances[0].InstanceId' --output text
```

The `--user-data` script (next section) runs automatically on boot: it installs deps, pulls code+data
from S3, trains, uploads results, and self-terminates. `instance-initiated-shutdown-behavior terminate`
means the box deletes itself when done — no lingering cost.

## 2. The bootstrap script (`ec2_train_bootstrap.sh`)

Save this file in the same directory before launching (the launch command references it):

```bash
#!/bin/bash
set -euxo pipefail
BUCKET=hurdle-data-791973676965
exec > >(tee /var/log/hurdle-train.log|logger -t hurdle -s 2>/dev/console) 2>&1

dnf install -y python3.11 python3.11-pip git
python3.11 -m pip install --upgrade pip
python3.11 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python3.11 -m pip install numpy==1.26.4 transformers==4.40.2 scikit-learn pandas medmnist

cd /home/ec2-user
mkdir -p run/src/hurdle/imaging run/data run/models && cd run
aws s3 cp s3://$BUCKET/train_imaging.py ./train_imaging.py
aws s3 cp s3://$BUCKET/code/imaging/ ./src/hurdle/imaging/ --recursive
aws s3 cp s3://$BUCKET/retinamnist_224.npz ./data/retinamnist_224.npz

# make the package importable and run
touch src/hurdle/__init__.py src/hurdle/imaging/__init__.py
PYTHONPATH=src python3.11 train_imaging.py --backbone resnet50 --epochs 8 --res 224 \
  --data data/retinamnist_224.npz --out models/ || echo "TRAIN FAILED"

# push results back to S3
aws s3 cp models/ s3://$BUCKET/trained/ --recursive
echo "DONE — uploaded to s3://$BUCKET/trained/"
shutdown -h now
```

## 3. Watch progress

```bash
# find the instance
aws ec2 describe-instances --region us-east-1 \
  --filters "Name=tag:Name,Values=hurdle-cpu-training" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].[InstanceId,PublicIpName,State.Name]' --output table

# check whether results landed (poll every few min; training ~30 min on c5.2xlarge)
aws s3 ls s3://hurdle-data-791973676965/trained/
```

When `imaging_resnet50.pt` and `imaging_embeddings.npz` appear under `trained/`, the run is done and the
instance has self-terminated.

## 4. Pull the trained artifacts back locally (optional)

```bash
aws s3 cp s3://hurdle-data-791973676965/trained/ ./models_ec2/ --recursive
```

## 5. Cost + cleanup

- `c5.2xlarge` on-demand ≈ **$0.34/hr**; a 30-40 min run ≈ **$0.20**.
- The instance self-terminates (`shutdown -h now` + terminate-on-shutdown), so there's nothing to clean
  up manually. Confirm with:

```bash
aws ec2 describe-instances --region us-east-1 \
  --filters "Name=tag:Name,Values=hurdle-cpu-training" \
  --query 'Reservations[].Instances[].State.Name' --output text
# should show 'terminated'
```

## For the GPU version later

Once the `g4dn.xlarge` spot quota clears, the same bootstrap works — launch from the provisioned launch
template instead (`--launch-template LaunchTemplateId=lt-0150fb14b0f1bd990`), install the CUDA torch build,
and point the script at `--backbone retfound`. Everything else is identical.
