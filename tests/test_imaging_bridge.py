"""Tests for the CNN-embedding -> fusion bridge, using a synthetic embedding
matrix so no trained CNN is needed to exercise the plumbing."""
import numpy as np
import pytest

from hurdle.fusion import imaging_bridge as ib
from hurdle.fusion import virtual_cohort as vc


@pytest.fixture
def fake_embeddings(tmp_path):
    #write an npz shaped like train_imaging.py's export: 3 splits x (n x dim)
    rng = np.random.default_rng(0)
    path = tmp_path / "imaging_embeddings.npz"
    np.savez(path,
             train=rng.normal(size=(80, 32)),
             val=rng.normal(size=(20, 32)),
             test=rng.normal(size=(40, 32)))
    return str(path)


def test_load_embeddings_split(fake_embeddings):
    emb = ib.load_embeddings(fake_embeddings, split="train")
    assert emb.shape == (80, 32)
    with pytest.raises(KeyError):
        ib.load_embeddings(fake_embeddings, split="nope")


def test_reduce_embeddings_shape(fake_embeddings):
    emb = ib.load_embeddings(fake_embeddings, split="train")
    reduced, scaler, pca = ib.reduce_embeddings(emb, n_components=6)
    assert reduced.shape == (80, 6)
    #the fitted pca can reapply to another split with the same dim
    val = ib.load_embeddings(fake_embeddings, split="val")
    assert pca.transform(scaler.transform(val)).shape == (20, 6)


def test_imaging_features_frame_columns(fake_embeddings):
    df = ib.imaging_features_frame(fake_embeddings, n_components=6)
    assert list(df.columns) == [f"imaging_f{i}" for i in range(6)]
    assert df.shape == (80, 6)
    assert np.isfinite(df.to_numpy()).all()


def test_real_frames_includes_imaging_when_given(fake_embeddings):
    frames = ib.real_frames_with_imaging(imaging_npz=fake_embeddings, n_components=6)
    assert "imaging" in frames and frames["imaging"].shape[1] == 6
    #omitted modalities are simply absent
    assert "omics" not in frames


def test_bridge_feeds_calibration(fake_embeddings):
    #the real imaging block flows into the virtual-cohort calibration path
    frames = ib.real_frames_with_imaging(imaging_npz=fake_embeddings, n_components=6)
    specs = vc.calibrate_from_real(frames)
    by_name = {s.name: s for s in specs}
    #imaging loading is set from the real embeddings, within the clamp range
    assert 0.1 <= by_name["imaging"].loading <= 1.5
