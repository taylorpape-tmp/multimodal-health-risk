"""Fine-tune the retinal CNN on RetinaMNIST and export embeddings for fusion.

Designed to run on the AWS GPU instance (infra/aws_gpu_training.tf) but falls
back to CPU. Two outputs:
  1. a fine-tuned checkpoint (models/imaging_<backbone>.pt)
  2. per-image embeddings for train/val/test (models/imaging_embeddings.npz) --
     these are the REAL feature vectors that anchor the fusion imaging block

Usage (from repo root):
  python scripts/train_imaging.py --backbone resnet50 --epochs 15 --res 224
  python scripts/train_imaging.py --backbone retfound --epochs 20 --res 224  # GPU

On AWS the wrapper (scripts/aws_train.sh) pulls data from S3, runs this, and
pushes models/ back to S3, then the instance is terminated.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hurdle.imaging import dataset, transfer  #noqa: E402


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(backbone="resnet50", epochs=15, res=224, lr=1e-3, batch=64,
          data_dir="data/raw/imaging", out_dir="models", seed=0):
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(seed)
    dev = _device()
    print(f"device={dev} backbone={backbone} epochs={epochs} res={res}", flush=True)

    npz = f"{data_dir}/retinamnist_224.npz" if res == 224 else f"{data_dir}/retinamnist.npz"
    data = dataset.load_retinamnist(npz, resolution=res)

    train_ds = dataset.RetinaMNIST(data.train_images, data.train_labels)
    val_ds = dataset.RetinaMNIST(data.val_images, data.val_labels)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch)

    #class-weighted loss for the imbalanced DR grades (grade 0 ~45%, grade 4 ~6%)
    weights = torch.tensor(dataset.compute_class_weights(data.train_labels),
                           dtype=torch.float32, device=dev)
    model = transfer.build_transfer_model(backbone=backbone, pretrained=True,
                                          freeze_backbone=True).to(dev)
    #two-stage: warm the head with the backbone frozen, then unfreeze and fine-tune
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def run_epoch(dl, training):
        model.train(training)
        tot, correct, losssum = 0, 0, 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            with torch.set_grad_enabled(training):
                out = model(xb)
                loss = loss_fn(out, yb)
                if training:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            losssum += float(loss) * len(yb)
            correct += int((out.argmax(1) == yb).sum())
            tot += len(yb)
        return losssum / tot, correct / tot

    for ep in range(epochs):
        if ep == epochs // 3:
            #stage 2: unfreeze backbone with a smaller lr (discriminative fine-tune)
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.1)
        tl, ta = run_epoch(train_dl, True)
        vl, va = run_epoch(val_dl, False)
        print(f"epoch {ep+1}/{epochs}  train_acc={ta:.3f}  val_acc={va:.3f}", flush=True)

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    ckpt = out / f"imaging_{backbone}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"saved checkpoint {ckpt}", flush=True)

    _export_embeddings(model, data, dev, out / "imaging_embeddings.npz")
    return str(ckpt)


def _export_embeddings(model, data, dev, path):
    #real per-image embeddings (penultimate features) -> anchor the fusion block
    import torch
    model.eval()
    feats = {}
    extractor = getattr(model, "features", None) or model
    for split, imgs in (("train", data.train_images), ("val", data.val_images),
                        ("test", data.test_images)):
        ds = dataset.RetinaMNIST(imgs, np.zeros(len(imgs), int))
        outs = []
        with torch.no_grad():
            for i in range(0, len(ds), 64):
                xb = torch.stack([ds[j][0] for j in range(i, min(i + 64, len(ds)))]).to(dev)
                z = model(xb) if extractor is model else extractor(xb).flatten(1)
                outs.append(z.cpu().numpy())
        feats[split] = np.concatenate(outs)
    np.savez(path, **feats)
    print(f"saved embeddings {path}  shapes={[(k, v.shape) for k, v in feats.items()]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50", "retfound", "dinov2"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--res", type=int, default=224, choices=[28, 224])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    train(backbone=args.backbone, epochs=args.epochs, res=args.res,
          lr=args.lr, batch=args.batch)
