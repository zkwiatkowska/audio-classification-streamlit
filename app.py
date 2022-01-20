import streamlit as st
import soundfile as sf
import matplotlib.pyplot as plt
from librosa import display
from audio_features import make_spectrogram, make_log_spectrogram, make_logmel_spectrogram, make_mel_spectrogram


def make_plot(feature, sr: int, name: str, y_axis: str):
    st.markdown(name)
    figs = plt.figure(figsize=(10, 5))
    display.specshow(
        feature,
        sr=sr,
        y_axis=y_axis,
        x_axis='time'
    )
    st.pyplot(fig=figs)


if __name__ == '__main__':

    st.header("Upload .wav audio file")
    uploaded_file = st.file_uploader("", type="wav")
    plt.style.use("ggplot")

    if uploaded_file is not None:
        audio, sample_rate = sf.read(uploaded_file)
        window_options = [32, 128, 1024, 2048, 4096]
        window_options = [x for x in window_options if x < len(audio)]

        st.header("Listen to your audio file")
        st.audio(uploaded_file)

        st.header("Waveform")
        fig = plt.figure(figsize=(10, 5))
        display.waveplot(audio, sr=sample_rate)
        st.pyplot(fig=fig)

        st.header("Feature plots")

        window = st.selectbox("Choose window size for STFT", options=window_options)
        melbins = st.selectbox("Choose number of Mel bins", options=[32, 64, 128, 256])

        st.subheader("Short-Term Fourier Transform")
        col1, col2 = st.columns(2)

        with col1:
            make_plot(
                feature=make_spectrogram(audio, n_fft=window, hop=window // 4),
                sr=sample_rate,
                name="**STFT with log y-axis**",
                y_axis="log"
            )

        with col2:
            make_plot(
                feature=make_log_spectrogram(audio, n_fft=window, hop=window // 4),
                sr=sample_rate,
                name="**Log-scaled STFT with log y-axis**",
                y_axis="log"
            )

        st.subheader("Mel Spectrogram")
        col1, col2 = st.columns(2)

        with col1:
            make_plot(
                feature=make_mel_spectrogram(audio, n_fft=window, hop=window // 4, n_mels=melbins),
                sr=sample_rate,
                name="**Mel-spectrogram with mel y-axis**",
                y_axis="mel"
            )

        with col2:
            make_plot(
                feature=make_logmel_spectrogram(audio, n_fft=window, hop=window // 4, n_mels=melbins),
                sr=sample_rate,
                name="**Log-scaled Mel-spectrogram with mel y-axis**",
                y_axis="mel"
            )