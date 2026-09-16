"""Transfer-learning scaffold for RetinaMNIST DR grading.

This builds models but does not train them. It gives the training script a fresh
5-class head on a chosen backbone, a freeze/unfreeze toggle (linear-probe vs full
fine-tune), discriminative-LR param groups (small LR on the backbone, larger on
the head), and a from-scratch torch-only CNN baseline for comparison.

Backbones are limited to a verified shortlist (retfound, resnet50, dinov2); see
BACKBONES for ids and licenses. torch (and transformers, for the pretrained
backbones) are optional: the module imports without them and the builder raises
a clear 'install torch' message so a torch-free environment fails at build time.
"""
from dataclasses import dataclass

NUM_CLASSES = 5


@dataclass(frozen=True)
class BackboneSpec:
    #repo_id: the verified Hugging Face id; kind selects how features are pooled
    key: str
    repo_id: str
    kind: str          #'hf_vit' | 'hf_resnet'
    feat_dim: int      #pooled feature width feeding the fresh head
    gated: bool
    license: str


#the verified shortlist; do not invent other ids
BACKBONES = {
    "retfound": BackboneSpec("retfound", "YukunZhou/RETFound_mae_natureCFP",
                             "hf_vit", 1024, gated=True, license="CC-BY-NC"),
    "resnet50": BackboneSpec("resnet50", "microsoft/resnet-50",
                             "hf_resnet", 2048, gated=False, license="Apache-2.0"),
    "dinov2":   BackboneSpec("dinov2", "facebook/dinov2-small",
                             "hf_vit", 384, gated=False, license="Apache-2.0"),
}

_INSTALL_TORCH_MSG = (
    "the transfer-learning scaffold requires torch (and transformers for the "
    "pretrained backbones). install with: pip install torch transformers. "
    "the from-scratch baseline needs only torch."
)


def _require_torch():
    #lazy probe so the module imports cleanly without torch; raise a clear message
    import importlib.util
    if importlib.util.find_spec("torch") is None:
        raise ImportError(_INSTALL_TORCH_MSG)
    import torch
    return torch


def _make_from_scratch(num_classes):
    #small torch-only CNN baseline, no pretrained weights, no downloads
    _require_torch()
    import torch.nn as nn

    class FromScratchCNN(nn.Module):
        #3 conv blocks -> global avg pool -> fresh linear head; the 'train from
        #zero' control against a pretrained backbone
        def __init__(self, n_out):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.feat_dim = 64
            self.head = nn.Linear(64, n_out)

        def forward(self, x):
            f = self.backbone(x).flatten(1)
            return self.head(f)

        def param_groups(self, base_lr, head_lr_mult=10.0):
            #from-scratch: backbone and head both fresh, so a single LR is fine,
            #but keep the two-group shape so callers treat all models uniformly
            return [
                {"params": [p for p in self.backbone.parameters() if p.requires_grad],
                 "lr": base_lr},
                {"params": [p for p in self.head.parameters() if p.requires_grad],
                 "lr": base_lr * head_lr_mult},
            ]

    return FromScratchCNN(num_classes)


def _make_hf_transfer(spec, num_classes, pretrained, freeze_backbone):
    #wrap a Hugging Face backbone with a fresh linear head
    _require_torch()
    import importlib.util
    if importlib.util.find_spec("transformers") is None:
        raise ImportError(_INSTALL_TORCH_MSG)
    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    if spec.gated and pretrained:
        #do not silently attempt a gated download; make the requirement explicit
        raise PermissionError(
            f"{spec.repo_id} is gated ({spec.license}); accept the license on "
            "Hugging Face and pass an auth token (huggingface-cli login), or use "
            "pretrained=False to build the architecture with random weights, or "
            "choose an ungated backbone (resnet50, dinov2)."
        )

    if pretrained:
        backbone = AutoModel.from_pretrained(spec.repo_id)
    else:
        backbone = AutoModel.from_config(AutoConfig.from_pretrained(spec.repo_id))

    class HFTransferModel(nn.Module):
        #backbone features -> pooled vector -> fresh num_classes head
        def __init__(self, bb, kind, feat_dim, n_out):
            super().__init__()
            self.backbone = bb
            self.kind = kind
            self.feat_dim = feat_dim
            self.head = nn.Linear(feat_dim, n_out)

        def _pool(self, out):
            #hf_resnet exposes pooler_output; vit models pool the CLS/mean token
            if getattr(out, "pooler_output", None) is not None:
                p = out.pooler_output
                return p.flatten(1) if p.dim() > 2 else p
            return out.last_hidden_state.mean(dim=1)

        def forward(self, x):
            return self.head(self._pool(self.backbone(x)))

        def param_groups(self, base_lr, head_lr_mult=10.0):
            #discriminative LR: small on the pretrained backbone, larger on the
            #fresh head; frozen params are dropped so the optimizer ignores them
            return [
                {"params": [p for p in self.backbone.parameters() if p.requires_grad],
                 "lr": base_lr},
                {"params": [p for p in self.head.parameters() if p.requires_grad],
                 "lr": base_lr * head_lr_mult},
            ]

    model = HFTransferModel(backbone, spec.kind, spec.feat_dim, num_classes)
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    return model


def build_transfer_model(backbone="resnet50", num_classes=NUM_CLASSES,
                         pretrained=True, freeze_backbone=True, from_scratch=False):
    """Build a DR-grading model with a fresh num_classes head.

    backbone is one of BACKBONES (ignored when from_scratch=True). pretrained
    loads backbone weights, but gated backbones (retfound) refuse a silent
    download, so pass pretrained=False to get the architecture only.
    freeze_backbone gives a linear-probe; unfreeze for full fine-tuning.
    from_scratch builds the torch-only baseline CNN instead.

    Returns a torch.nn.Module whose .head is a Linear of width num_classes and
    which exposes .param_groups(base_lr, head_lr_mult) for discriminative-LR
    optimisation. Raises ImportError when torch (or transformers) is missing.
    """
    _require_torch()
    if from_scratch:
        return _make_from_scratch(num_classes)
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from "
                         f"{tuple(BACKBONES)} or pass from_scratch=True")
    return _make_hf_transfer(BACKBONES[backbone], num_classes, pretrained,
                             freeze_backbone)
