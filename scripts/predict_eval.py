from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import get_device, resolve_la_root
from src.data import ASVspoofDataset
from src.model import LightCNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None, help="Path to .../LA/LA (local). On Kaggle it's auto.")
    parser.add_argument("--ckpt", type=str, default="artifacts/best_eval.pth")
    parser.add_argument("--out", type=str, default="vdmalevanik.csv")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--header", action="store_true", help="Write header to CSV (default: NO header)")
    args = parser.parse_args()

    device = get_device()
    la_root = resolve_la_root(args.data_root)

    proto_dir = la_root / "ASVspoof2019_LA_cm_protocols"
    eval_proto  = proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt"
    eval_dir  = la_root / "ASVspoof2019_LA_eval" / "flac"

    # protocol order
    proto_ids = [ln.split()[1] for ln in open(eval_proto, "r", encoding="utf-8") if len(ln.split()) >= 2]
    eval_paths = [str(eval_dir / f"{u}.flac") for u in proto_ids]

    ds = ASVspoofDataset(eval_paths, labels=None)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = LightCNN().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    scores = {}
    use_amp = (device.type == "cuda")

    with torch.no_grad():
        for x, _, utts in tqdm(dl, desc="Infer eval", leave=False):
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                probs = F.softmax(model(x), dim=1)[:, 1].detach().cpu().numpy()
            scores.update(dict(zip(utts, probs.tolist())))

    # build CSV exactly in protocol order
    rows = [(u, scores[u]) for u in proto_ids]
    out_path = Path(args.out)
    pd.DataFrame(rows).to_csv(out_path, index=False, header=args.header)
    print(f"CSV saved -> {out_path.resolve()}")

if __name__ == "__main__":
    main()
