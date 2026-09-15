"""Tests for the retinal-imaging modality: RetinaMNIST loader + transfer scaffold.

These use a tiny synthetic .npz written to tmp_path so they run in milliseconds
and never depend on the gitignored raw data. The torch path is guarded with
importorskip and only builds the torch-only from-scratch baseline, so no weights
(least of all the gated RETFound weights) are ever downloaded.
"""
import numpy as np
import pytest

from hurdle.imaging.dataset import (
    NUM_CLASSES,
    RetinaMNISTData,
    compute_class_weights,
    load_retinamnist,
)
from hurdle.imaging.transfer import BACKBONES, build_transfer_model


def _write_synthetic_npz(path, resolution=28, counts=(6, 2, 3, 2, 1)):
    #tiny RetinaMNIST-shaped file: imbalanced grades matching the real 0..4 layout
    rng = np.random.default_rng(0)

    def split(n_per_class):
        labels = np.concatenate([np.full(c, g, np.uint8)
                                 for g, c in enumerate(n_per_class)])
        n = len(labels)
        images = rng.integers(0, 256, size=(n, resolution, resolution, 3),
                              dtype=np.uint8)
        return images, labels.reshape(-1, 1)

    tr_i, tr_l = split(counts)
    va_i, va_l = split((2, 1, 1, 1, 1))
    te_i, te_l = split((2, 1, 1, 1, 1))
    np.savez(path,
             train_images=tr_i, train_labels=tr_l,
             val_images=va_i, val_labels=va_l,
             test_images=te_i, test_labels=te_l)


@pytest.fixture
def synthetic_npz(tmp_path):
    p = tmp_path / "retinamnist_tiny.npz"
    _write_synthetic_npz(p, resolution=28)
    return p


def test_load_returns_expected_shapes_and_label_range(synthetic_npz):
    data = load_retinamnist(synthetic_npz, resolution=28)
    assert isinstance(data, RetinaMNISTData)
    assert data.resolution == 28
    #14 train images across the 5 imbalanced grades
    assert data.train_images.shape == (14, 28, 28, 3)
    assert data.train_images.dtype == np.uint8
    for split in ("train", "val", "test"):
        images, labels = data.split(split)
        assert images.shape[0] == labels.shape[0]
        assert images.shape[1:] == (28, 28, 3)
        assert labels.ndim == 1
        assert labels.min() >= 0 and labels.max() <= 4


def test_resolution_mismatch_raises(synthetic_npz):
    #passing the wrong resolution for the file must fail loudly
    with pytest.raises(ValueError, match="resolution"):
        load_retinamnist(synthetic_npz, resolution=224)


def test_class_weights_computed_and_sum_normalised(synthetic_npz):
    data = load_retinamnist(synthetic_npz, resolution=28)
    w = data.class_weights
    assert w.shape == (NUM_CLASSES,)
    assert np.all(np.isfinite(w))
    #sum-normalised to NUM_CLASSES so an all-equal distribution would give ones
    assert np.isclose(w.sum(), NUM_CLASSES)
    #inverse-frequency: the rarest grade (4, count 1) outweighs the commonest (0, count 6)
    assert w[4] > w[0]


def test_class_weights_uniform_gives_ones():
    #equal counts per class -> all weights == 1 (matches an unweighted loss)
    labels = np.repeat(np.arange(NUM_CLASSES), 4)
    w = compute_class_weights(labels)
    assert np.allclose(w, 1.0)


def test_class_weights_reject_out_of_range():
    with pytest.raises(ValueError):
        compute_class_weights(np.array([0, 1, 9]))


def test_backbone_shortlist_ids_are_the_verified_ones():
    #guard against drift: the scout's verified ids, no invented repos
    assert BACKBONES["retfound"].repo_id == "YukunZhou/RETFound_mae_natureCFP"
    assert BACKBONES["retfound"].gated is True
    assert BACKBONES["resnet50"].repo_id == "microsoft/resnet-50"
    assert BACKBONES["dinov2"].repo_id == "facebook/dinov2-small"


def test_builder_from_scratch_has_five_unit_head_when_torch_present():
    #torch-guarded: only the torch-only from-scratch baseline, no downloads
    torch = pytest.importorskip("torch")
    model = build_transfer_model(from_scratch=True, num_classes=NUM_CLASSES)
    assert isinstance(model, torch.nn.Module)
    assert model.head.out_features == NUM_CLASSES
    #a tiny CPU forward pass yields one logit row per image over the 5 grades
    x = torch.zeros(2, 3, 28, 28)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, NUM_CLASSES)
    #discriminative param groups: head LR above backbone LR
    groups = model.param_groups(base_lr=1e-4, head_lr_mult=10.0)
    assert len(groups) == 2
    assert groups[1]["lr"] > groups[0]["lr"]


def test_builder_without_torch_raises_install_message(monkeypatch):
    #simulate a torch-free env: the builder must raise a clear install hint
    import importlib.util as ilu
    real_find_spec = ilu.find_spec

    def fake_find_spec(name, *a, **k):
        if name == "torch":
            return None
        return real_find_spec(name, *a, **k)

    #_require_torch calls importlib.util.find_spec; patch it on the module itself
    monkeypatch.setattr(ilu, "find_spec", fake_find_spec)
    with pytest.raises(ImportError, match="install"):
        build_transfer_model(from_scratch=True)
