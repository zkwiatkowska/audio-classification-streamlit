from pathlib import Path
from torch.utils.data import Dataset
import torch
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm
import torchopenl3


class ESC50Reader(Dataset):

    def __init__(self, path: Path):
        super().__init__()
        self.path = path / "audio"
        self.gt = pd.read_csv(path / "meta/esc50.csv")

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, item):
        info = self.gt.iloc[item]
        audio, sr = sf.read(self.path / info["filename"])
        target = torch.one_hot(torch.tensor(info['target']), num_classes=50)
        return audio, target, info["filename"]


class ESC50OpenL3Reader(Dataset):

    def __init__(self, orig_path: Path, emb_path: Path, *, train=True, test_fold=5):
        super().__init__()
        self.path = emb_path
        gt = pd.read_csv(orig_path / "meta/esc50.csv")
        self.gt = gt[gt["fold"] != test_fold] if train else gt[gt["fold"] == test_fold]
        self.gt.reset_index(inplace=True)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, item):
        info = self.gt.iloc[item]
        with open(self.path / f"{info['filename'].replace('.wav', '.npy')}", "rb") as f:
            emb, target = np.load(f, allow_pickle=True)
        emb = emb.squeeze(axis=0)
        emb = (emb - emb.min()) / (emb.max() - emb.min())
        return emb, target


def process_esc50_to_openl3(data_path: Path, embedding_size=512):
    reader = ESC50Reader(data_path)
    output_path = Path(f"esc50_openl3_{embedding_size}")
    for sample in tqdm(reader):
        audio, target, filename = sample
        emb, _ = torchopenl3.get_audio_embedding(audio, 44100, content_type="env", embedding_size=embedding_size)
        emb = emb.mean(axis=1).cpu().numpy()
        with open(output_path / f"{filename.replace('wav', 'npy')}", 'wb') as f:
            np.save(f, [emb, target], allow_pickle=True)
