"""RetinaMNIST loader for the retinal-imaging modality.

RetinaMNIST is diabetic-retinopathy grading: RGB fundus crops with an ordinal
label 0 to 4 (0 = no DR, 4 = proliferative). The grades are imbalanced (grade 0
is ~45% of train, grade 4 ~6%), so the loader also computes sum-normalised class
weights for a weighted loss.

The .npz ships two resolutions (28px and 224px), each with images (N, H, W, 3)
uint8 and labels (N, 1) uint8 in {0..4} for train/val/test. torch is optional:
when present, load_retinamnist can build a torch Dataset per split; when absent,
it still returns the numpy arrays and class weights.
"""
from dataclasses import dataclass, field

import numpy as np

#ordinal DR grades, index == label value
RETINAMNIST_CLASSES = (
    "no_DR",
    "mild",
    "moderate",
    "severe",
    "proliferative",
)
NUM_CLASSES = len(RETINAMNIST_CLASSES)

_SPLITS = ("train", "val", "test")


def _torch_available():
    #cheap import probe; kept in a helper so callers/tests can monkeypatch
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def compute_class_weights(labels, num_classes=NUM_CLASSES):
    """Inverse-frequency class weights, rescaled to sum to num_classes.

    An all-equal distribution gives all-ones (matching an unweighted loss), and
    classes absent from `labels` get weight 0 rather than infinity.
    """
    labels = np.asarray(labels).ravel().astype(int)
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(f"labels must be in [0, {num_classes - 1}], got "
                         f"[{labels.min()}, {labels.max()}]")
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    inv = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
    total = inv.sum()
    if total == 0:
        raise ValueError("no labels supplied; cannot compute class weights")
    return (inv / total * num_classes).astype(np.float64)


@dataclass
class RetinaMNISTData:
    """The three splits plus class weights.

    Images are (N, H, W, 3) uint8; labels are 1-D int arrays in {0..4}. Class
    weights come from the train split only, so val/test never inform the loss.
    """
    resolution: int
    train_images: np.ndarray
    train_labels: np.ndarray
    val_images: np.ndarray
    val_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    class_weights: np.ndarray
    class_names: tuple = field(default=RETINAMNIST_CLASSES)

    def split(self, name):
        #return (images, labels) for 'train' | 'val' | 'test'
        if name not in _SPLITS:
            raise ValueError(f"unknown split {name!r}; choose from {_SPLITS}")
        return getattr(self, f"{name}_images"), getattr(self, f"{name}_labels")

    def torch_dataset(self, name, normalize=True):
        #build a torch Dataset for one split; raises if torch is absent
        images, labels = self.split(name)
        return RetinaMNIST(images, labels, normalize=normalize)


def load_retinamnist(npz_path, resolution):
    """Load a RetinaMNIST .npz and return a RetinaMNISTData container.

    resolution (28 or 224) is validated against the file so a path/resolution
    mismatch fails loudly. Class weights come from the train split. torch is not
    required; the container carries numpy arrays and can build a torch Dataset
    later via .torch_dataset(...) when torch is installed.
    """
    with np.load(npz_path) as z:
        missing = [k for s in _SPLITS
                   for k in (f"{s}_images", f"{s}_labels") if k not in z.files]
        if missing:
            raise KeyError(f"{npz_path} is missing expected arrays: {missing}")
        arrs = {k: z[k] for s in _SPLITS
                for k in (f"{s}_images", f"{s}_labels")}

    #validate image contract on the train split and check the requested resolution
    ti = arrs["train_images"]
    if ti.ndim != 4 or ti.shape[-1] != 3:
        raise ValueError(f"expected (N, H, W, 3) images, got {ti.shape}")
    h, w = ti.shape[1], ti.shape[2]
    if h != w:
        raise ValueError(f"expected square images, got {h}x{w}")
    if int(resolution) != h:
        raise ValueError(f"resolution={resolution} does not match file images "
                         f"({h}px); use the matching .npz")

    labels = {s: arrs[f"{s}_labels"].ravel().astype(np.int64) for s in _SPLITS}
    lo = min(int(labels[s].min()) for s in _SPLITS)
    hi = max(int(labels[s].max()) for s in _SPLITS)
    if lo < 0 or hi >= NUM_CLASSES:
        raise ValueError(f"labels out of range: found [{lo}, {hi}], "
                         f"expected [0, {NUM_CLASSES - 1}]")

    return RetinaMNISTData(
        resolution=h,
        train_images=arrs["train_images"], train_labels=labels["train"],
        val_images=arrs["val_images"],     val_labels=labels["val"],
        test_images=arrs["test_images"],   test_labels=labels["test"],
        class_weights=compute_class_weights(labels["train"]),
    )


#imagenet channel stats, the standard normalisation for RGB fundus transfer models
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class RetinaMNIST:
    """torch Dataset over one RetinaMNIST split.

    Yields (image, label) where image is a float32 CHW tensor scaled to [0, 1]
    (imagenet-normalised when normalize=True) and label is a long scalar. It
    only constructs when torch is installed, raising a clear ImportError
    otherwise so a torch-free environment fails at build time, not import time.
    """

    def __init__(self, images, labels, normalize=True):
        if not _torch_available():
            raise ImportError(
                "RetinaMNIST torch Dataset requires torch; install torch "
                "(and torchvision) or use the numpy arrays on RetinaMNISTData "
                "instead."
            )
        import torch

        self._torch = torch
        imgs = np.asarray(images)
        if imgs.ndim != 4 or imgs.shape[-1] != 3:
            raise ValueError(f"expected (N, H, W, 3) images, got {imgs.shape}")
        self.images = imgs
        self.labels = np.asarray(labels).ravel().astype(np.int64)
        if len(self.images) != len(self.labels):
            raise ValueError("images and labels length mismatch: "
                             f"{len(self.images)} vs {len(self.labels)}")
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        torch = self._torch
        img = self.images[idx].astype(np.float32) / 255.0
        if self.normalize:
            img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        #HWC -> CHW
        chw = np.transpose(img, (2, 0, 1)).copy()
        return torch.from_numpy(chw), torch.tensor(int(self.labels[idx]))
