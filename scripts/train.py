from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import set_seed, get_device, resolve_la_root, ensure_dir
from src.data import load_split, ASVspoofDataset
from src.model import LightCNN
from src.metrics import compute_eer
from src.wandb_utils import init_wandb

def build_loaders(la_root: Path, batch_size: int, num_workers: int):
    proto_dir = la_root / "ASVspoof2019_LA_cm_protocols"
    train_proto = proto_dir / "ASVspoof2019.LA.cm.train.trn.txt"
    dev_proto   = proto_dir / "ASVspoof2019.LA.cm.dev.trl.txt"
    eval_proto  = proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt"

    train_dir = la_root / "ASVspoof2019_LA_train" / "flac"
    dev_dir   = la_root / "ASVspoof2019_LA_dev"   / "flac"
    eval_dir  = la_root / "ASVspoof2019_LA_eval"  / "flac"

    assert train_proto.is_file() and dev_proto.is_file() and eval_proto.is_file(), "Protocol files not found"
    assert train_dir.exists() and dev_dir.exists() and eval_dir.exists(), "Audio dirs not found"

    train_files, train_labels = load_split(train_proto, train_dir, is_eval=False)
    dev_files,   dev_labels   = load_split(dev_proto,   dev_dir,   is_eval=False)

    train_ds = ASVspoofDataset(train_files, train_labels)
    dev_ds   = ASVspoofDataset(dev_files, dev_labels)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    dev_dl = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    paths = {
        "train_proto": train_proto, "dev_proto": dev_proto, "eval_proto": eval_proto,
        "train_dir": train_dir, "dev_dir": dev_dir, "eval_dir": eval_dir
    }
    return train_dl, dev_dl, paths

@torch.no_grad()
def evaluate(model, loader, device, crit) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    ys = []
    ps = []

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = crit(logits, y)
        total_loss += loss.item() * x.size(0)

        prob_spoof = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        ys.append(y.detach().cpu().numpy())
        ps.append(prob_spoof)

    y_true = np.concatenate(ys, axis=0)
    y_score = np.concatenate(ps, axis=0)
    eer = compute_eer(y_true, y_score)

    avg_loss = total_loss / len(loader.dataset)
    return float(avg_loss), float(eer)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None, help="Path to .../LA/LA (local). On Kaggle it's auto.")
    parser.add_argument("--epochs", type=int, default=33)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)  # Windows friendly
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="asvspoof-lightcnn")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="artifacts")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    la_root = resolve_la_root(args.data_root)
    out_dir = ensure_dir(args.out_dir)

    config = vars(args) | {"device": str(device), "la_root": str(la_root)}
    wandb_on = init_wandb(args.project, args.run_name, config)

    # silence AMP deprecation warnings
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r"`torch\.cuda\.amp\..*` is deprecated",
    )

    train_dl, dev_dl, _ = build_loaders(la_root, args.batch_size, args.num_workers)

    model = LightCNN().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, verbose=True)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_eer = 1e9
    best_path = out_dir / "best_eval.pth"

    import wandb
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch:02d}/{args.epochs}", leave=False)
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item() * x.size(0)

        tr_loss = running / len(train_dl.dataset)
        sch.step(tr_loss)

        if wandb_on:
            wandb.log({"epoch": epoch, "loss_train": tr_loss, "lr": opt.param_groups[0]["lr"]})
        else:
            try:
                wandb.log({"epoch": epoch, "loss_train": tr_loss, "lr": opt.param_groups[0]["lr"]})
            except Exception:
                pass

        # dev every 3 epochs + final
        if epoch % 3 == 0 or epoch == args.epochs:
            dv_loss, dv_eer = evaluate(model, dev_dl, device, crit)
            if wandb_on:
                wandb.log({"loss_dev": dv_loss, "eer_dev": dv_eer})

            if dv_eer < best_eer:
                best_eer = dv_eer
                torch.save(model.state_dict(), best_path)

            print(f"[{epoch:02d}/{args.epochs}] train_loss={tr_loss:.4f} | dev_loss={dv_loss:.4f} | dev_eer={dv_eer*100:.2f}%")
        else:
            print(f"[{epoch:02d}/{args.epochs}] train_loss={tr_loss:.4f}")

    print(f"Best EER: {best_eer*100:.2f}%")
    print(f"Best ckpt saved: {best_path}")

if __name__ == "__main__":
    main()
