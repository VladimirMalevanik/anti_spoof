from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

def load_split(proto_path: Path, audio_dir: Path, is_eval: bool = False) -> Tuple[List[str], List[Optional[int]]]:
    files, labels = [], []
    with open(proto_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            # протокол: ... utt_id ...
            utt = parts[1]
            lab = None
            if not is_eval:
                # в оригинале метка в конце строки: bonafide/spoof
                if len(parts) >= 5:
                    lab = 0 if parts[-1] == "bonafide" else 1
                else:
                    continue

            path = audio_dir / f"{utt}.flac"
            if path.is_file():
                files.append(str(path))
                labels.append(lab)
    return files, labels

class ASVspoofDataset(Dataset):
    """
    Возвращает:
      x: FloatTensor [1, 867, 600]
      y: LongTensor scalar (0 bonafide / 1 spoof) или -1 для eval
      utt: str (stem filename)
    """
    def __init__(self, files: List[str], labels: Optional[List[Optional[int]]] = None):
        self.files = files
        self.labels = labels

        self.n_fft = 1724
        self.hop_length = 128
        self.win_length = 1724
        self.window = torch.blackman_window(self.win_length)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        wav, sr = torchaudio.load(self.files[idx])  # [C,T]
        wav = wav.mean(0)  # mono [T]
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(wav.device),
            center=True,
            pad_mode="reflect",
            onesided=True,
            return_complex=True,
        )
        power = spec.abs() ** 2
        logp = torch.log10(power + 1e-10)  # [F,T]

        # fix frequency bins to 867
        f, t = logp.shape
        if f < 867:
            pad = 867 - f
            logp = F.pad(logp, (0, 0, pad // 2, pad - pad // 2))
        else:
            logp = logp[:867, :]

        # fix time frames to 600
        if t < 600:
            logp = F.pad(logp, (0, 600 - t, 0, 0))
        else:
            logp = logp[:, :600]

        x = logp.unsqueeze(0).float()  # [1,867,600]

        if self.labels is None:
            y = -1
        else:
            lab = self.labels[idx]
            y = -1 if lab is None else int(lab)

        utt = Path(self.files[idx]).stem
        return x, torch.tensor(y, dtype=torch.long), utt
