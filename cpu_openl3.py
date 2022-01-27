import numpy as np

import torch
from torch import tensor as T

import torchopenl3


def get_audio_embedding(
    audio,
    sr,
    model=None,
    input_repr="mel256",
    content_type="music",
    embedding_size=6144,
    center=True,
    hop_size=0.1,
    batch_size=32,
    sampler="resampy",
):

    if model is None:
        model = torchopenl3.core.load_audio_embedding_model(input_repr, content_type, embedding_size)

    device = "cpu"

    if isinstance(audio, np.ndarray):
        audio = T(audio, device=device, dtype=torch.float64)

    if audio.ndim == 1:
        audio = audio.view(1, -1)
    elif audio.ndim == 2 and audio.shape[1] == 2:
        audio = audio.view(1, audio.shape[0], audio.shape[1])
    assert audio.ndim == 2 or audio.ndim == 3
    nsounds = audio.shape[0]

    audio = torchopenl3.core.preprocess_audio_batch(audio, sr, center, hop_size, sampler=sampler).to(
        torch.float32
    )
    total_size = audio.size()[0]
    audio_embedding = []
    with torch.set_grad_enabled(False):
        for i in range((total_size // batch_size) + 1):
            small_batch = audio[i * batch_size : (i + 1) * batch_size]
            if small_batch.shape[0] > 0:
                # print("small_batch.shape", small_batch.shape)
                audio_embedding.append(model(small_batch))
    audio_embedding = torch.vstack(audio_embedding)
    # This is broken, doesn't use hop-size or center
    ts_list = torch.arange(audio_embedding.size()[0] // nsounds) * hop_size
    ts_list = ts_list.expand(nsounds, audio_embedding.size()[0] // nsounds)

    return (
        audio_embedding.view(nsounds, audio_embedding.shape[0] // nsounds, -1),
        ts_list,
    )
