from argparse import ArgumentParser
from pathlib import Path

from librosa import display
import matplotlib.pyplot as plt
import numpy as np

from audio_features import make_spectrogram, make_log_spectrogram, make_mel_spectrogram, make_logmel_spectrogram
from data_processors import ESC50Reader

EXAMPLES = {
    "Dog": 496,
    "Vacuum Cleaner": 1061,
    "Siren": 641,
    "Church Bells": 1841
}
SR = 44100


def make_plot(func: callable, func_args: dict, data: dict, name: str):
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    ax = [i for sublist in ax for i in sublist]
    for ix, sound in enumerate(data):
        func(data[sound], ax=ax[ix], sr=SR, **func_args)
        ax[ix].set_title(sound, fontdict={'size': 10})
        ax[ix].set_xlabel('Time [s]', fontdict={'size': 10})
    fig.subplots_adjust(hspace=0.5, wspace=0.25)
    fig.savefig(f"plots/{name}.png", dpi=800)


if __name__ == '__main__':

    plt.style.use('ggplot')
    Path("plots").mkdir(exist_ok=True)

    parser = ArgumentParser()
    parser.add_argument("--esc50-path", required=True, type=Path)
    args = parser.parse_args()

    dataset = ESC50Reader(args.esc50_path)

    # FEATURE EXTRACTION
    waves = {k: dataset[v][0] for k, v in EXAMPLES.items()}
    normed = {k: v / np.abs(v).max() for k, v in waves.items()}

    spectrogram_small_window = {k: make_spectrogram(v, n_fft=128, hop=32) for k, v in normed.items()}
    spectrogram_wide_window = {k: make_spectrogram(v, n_fft=4096, hop=1024) for k, v in normed.items()}
    log_spectrogram = {k: make_log_spectrogram(v, n_fft=4096, hop=1024) for k, v in normed.items()}

    mel_spectrogram = {k: make_mel_spectrogram(v, n_fft=4096, hop=1024, n_mels=128) for k, v in normed.items()}
    log_mel_spectrogram = {k: make_logmel_spectrogram(v, n_fft=4096, hop=1024, n_mels=128) for k, v in normed.items()}

    # PLOTS

    # spectrogram low window linear scale magnitude
    make_plot(func=display.specshow, func_args=dict(y_axis='linear', x_axis='time'), data=spectrogram_small_window,
              name='stft_small_window_linear')

    log_y_stft = {
        'stft_small_window_log': spectrogram_small_window,
        'stft_wide_window_log': spectrogram_wide_window,
        'log_stft': log_spectrogram,
    }

    for filename, features in log_y_stft.items():
        make_plot(func=display.specshow, func_args=dict(y_axis='log', x_axis='time'), data=features,
                  name=filename)

    mel_y_stft = {
        'stft_mel': mel_spectrogram,
        'log_stft_mel': log_mel_spectrogram,
    }

    for filename, features in mel_y_stft.items():
        make_plot(func=display.specshow, func_args=dict(y_axis='mel', x_axis='time'), data=features,
                  name=filename)
