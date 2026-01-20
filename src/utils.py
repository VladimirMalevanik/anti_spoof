import os
import glob
from pathlib import Path
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_kaggle() -> bool:
    return (
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
        or Path("/kaggle").exists()
    )

def find_la_root_kaggle() -> Path:
    candidates = glob.glob("/kaggle/input/*/LA/LA")
    if not candidates:
        raise FileNotFoundError(
            "ASVspoof2019 LA not found. Attach dataset under /kaggle/input/<dataset>/LA/LA"
        )
    return Path(candidates[0])

def resolve_la_root(data_root: str | None) -> Path:
    if data_root:
        p = Path(data_root)
        if not p.exists():
            raise FileNotFoundError(f"--data_root not found: {p}")
        return p
    if is_kaggle():
        return find_la_root_kaggle()
    raise ValueError("Provide --data_root when running locally (path to .../LA/LA).")

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
