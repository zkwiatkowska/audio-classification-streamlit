import librosa
import numpy as np


def make_spectrogram(audio, n_fft, hop_length):
    return np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)) ** 2


def make_log_spectrogram(audio, n_fft, hop_length):
    spectrogram = make_spectrogram(audio, n_fft, hop_length)
    return librosa.power_to_db(spectrogram, ref=np.max)


def make_mel_spectrogram(audio, n_fft, hop_length):
    spectrogram = make_spectrogram(audio, n_fft, hop_length)
    return librosa.feature.melspectrogram(S=spectrogram, n_mels=128)


def make_logmel_spectrogram(audio, n_fft, hop_length):
    mel_spectrogram = make_mel_spectrogram(audio, n_fft, hop_length)
    return librosa.power_to_db(mel_spectrogram)
