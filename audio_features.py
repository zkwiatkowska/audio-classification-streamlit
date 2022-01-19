import librosa
import numpy as np


def make_spectrogram(audio, n_fft, hop):
    return np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop)) ** 2


def make_log_spectrogram(audio, n_fft, hop):
    spectrogram = make_spectrogram(audio, n_fft, hop)
    return librosa.power_to_db(spectrogram, ref=np.max)


def make_mel_spectrogram(audio, n_fft, hop, n_mels):
    spectrogram = make_spectrogram(audio, n_fft, hop)
    return librosa.feature.melspectrogram(S=spectrogram, n_mels=n_mels)


def make_logmel_spectrogram(audio, n_fft, hop, n_mels):
    mel_spectrogram = make_mel_spectrogram(audio, n_fft, hop, n_mels)
    return librosa.power_to_db(mel_spectrogram)
