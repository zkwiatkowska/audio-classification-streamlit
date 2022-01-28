import numpy as np

import torch
from torch import tensor as T

import torchopenl3


def get_audio_embedding(
    audio,
    sr,
    model,
    embedding_size=6144,
    center=True,
    hop_size=0.5,
    batch_size=16,
    sampler="resampy",
):

    device = "cpu"

    if isinstance(audio, np.ndarray):
        audio = T(audio, device=device, dtype=torch.float64)

    if audio.ndim == 1:
        audio = audio.view(1, -1)
    elif audio.ndim == 2 and audio.shape[1] == 2:
        audio = audio.view(1, audio.shape[0], audio.shape[1])
    assert audio.ndim == 2 or audio.ndim == 3

    audio = torchopenl3.core.preprocess_audio_batch(audio, sr, center, hop_size, sampler=sampler).to(
        torch.float32
    )

    total_size = audio.size()[0]
    audio_embedding = torch.zeros(size=(1, embedding_size))
    ctr = 0

    for i in range(total_size):
        y = model(audio[i:i+1, :, :])
        ctr += y.shape[0]
        audio_embedding += y.sum(axis=0)

    audio_embedding /= ctr

    return audio_embedding
