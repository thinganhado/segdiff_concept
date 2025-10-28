# datasets/voc_v4.py
from pathlib import Path
import random
import numpy as np
import torch
from mpi4py import MPI
from torch.utils.data import DataLoader, Dataset

# ---------- helpers ----------
def _db_to_m1p1(x_db: np.ndarray) -> np.ndarray:
    # input saved as dB in [-60, 0] -> map to [-1, 1]
    return (x_db + 60.0) / 30.0 - 1.0

def _mask_to_m1p1(m: np.ndarray) -> np.ndarray:
    # uint8 {0,1} -> [-1, 1]
    return 2.0 * m.astype(np.float32) - 1.0

def _pad_to_min_size(arr: np.ndarray, min_h: int, min_w: int, pad_value=0.0) -> np.ndarray:
    h, w = arr.shape
    ph = max(0, min_h - h)
    pw = max(0, min_w - w)
    if ph == 0 and pw == 0:
        return arr
    out = np.full((h + ph, w + pw), pad_value, dtype=arr.dtype)
    out[:h, :w] = arr
    return out

def _random_or_center_crop(arr: np.ndarray, size: int, train: bool) -> np.ndarray:
    h, w = arr.shape
    if h == size and w == size:
        return arr
    if train:
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
    else:
        top = (h - size) // 2
        left = (w - size) // 2
    return arr[top:top+size, left:left+size]

def _find_pairs_monu_like(root: Path, split: str):
    """Expect voc.v4/<split>/{img,mask} with npy files named Ms_smooth__*.npy and mask95_smoothed__*.npy."""
    base = root / split
    img_dir = base / "img"
    mask_dir = base / "mask"
    pairs = []
    if not img_dir.exists() or not mask_dir.exists():
        return pairs
    for ms_path in img_dir.glob("Ms_smooth__*.npy"):
        stem_part = ms_path.stem.split("Ms_smooth__")[-1]
        mask_path = mask_dir / f"mask95_smoothed__{stem_part}.npy"
        if mask_path.exists():
            pairs.append((ms_path, mask_path, stem_part))
    return pairs

# ---------- dataset ----------
class VocV4Dataset(Dataset):
    """
    Returns:
      mask: 1×H×W float tensor in [-1,1]
      out_dict["conditioned_image"]: 3×H×W float tensor in [-1,1] (grayscale triplicated)
      id_str: string
    """
    def __init__(self, root: Path, split="Training", image_size=256, train=False):
        self.root = Path(root)
        self.train = bool(train)
        self.image_size = int(image_size)
        self.split = split

        pairs = _find_pairs_monu_like(self.root, split)
        if not pairs:
            raise RuntimeError(
                f"[voc.v4] No pairs found under {self.root}/{split}. "
                f"Expected {self.root}/{split}/img and /mask with matching npy files."
            )

        # MPI shard
        shard = MPI.COMM_WORLD.Get_rank()
        num_shards = MPI.COMM_WORLD.Get_size()
        self.pairs = pairs[shard::num_shards]

        print(f"[voc.v4:{split}] total {len(pairs)} | rank {shard}/{num_shards} -> {len(self.pairs)} items")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ms_path, mask_path, pid = self.pairs[idx]
        S_db = np.load(ms_path).astype(np.float32)   # F×T, in dB [-60,0]
        M01  = np.load(mask_path).astype(np.uint8)   # F×T, {0,1}

        # normalize
        S = _db_to_m1p1(S_db)                        # [-1,1]
        M = _mask_to_m1p1(M01)                       # [-1,1]

        # pad then crop to square image_size
        S = _pad_to_min_size(S, self.image_size, self.image_size, pad_value=-1.0)
        M = _pad_to_min_size(M, self.image_size, self.image_size, pad_value=-1.0)
        S = _random_or_center_crop(S, self.image_size, train=self.train)
        M = _random_or_center_crop(M, self.image_size, train=self.train)

        # replicate conditioning to 3 channels for RRDB
        cond = np.stack([S, S, S], axis=0)          # 3×H×W
        mask = M[None, ...]                          # 1×H×W

        out_dict = {"conditioned_image": torch.from_numpy(cond)}
        return torch.from_numpy(mask), out_dict, f"{Path(pid).stem}_{idx}"

# ---------- API (mirrors monu.py signatures) ----------
def create_dataset(mode="train", image_size=256, data_dir=None):
    # use CLI path when provided, else fall back
    datadir = Path(data_dir) if data_dir else Path(__file__).absolute().parents[2] / "data/voc.v4"
    return VocV4Dataset(
        datadir,
        split=("Training" if mode == "train" else "Test"),
        image_size=image_size,
        train=(mode == "train"),
    )

def load_data(*, data_dir, batch_size, image_size, class_name,
              class_cond=False, expansion=None, deterministic=False):
    dataset = create_dataset(mode="train", image_size=image_size, data_dir=data_dir)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=not deterministic,
                        num_workers=0,
                        drop_last=True)
    while True:
        yield from loader