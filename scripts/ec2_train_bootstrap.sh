#!/bin/bash
#EC2 user-data bootstrap: pulls code+data from S3, trains the retinal CNN, pushes results back, self-terminates
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

touch src/hurdle/__init__.py src/hurdle/imaging/__init__.py
PYTHONPATH=src python3.11 train_imaging.py --backbone resnet50 --epochs 8 --res 224 \
  --data data/retinamnist_224.npz --out models/ || echo "TRAIN FAILED"

aws s3 cp models/ s3://$BUCKET/trained/ --recursive
echo "DONE — uploaded to s3://$BUCKET/trained/"
shutdown -h now
