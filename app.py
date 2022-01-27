import streamlit as st
import soundfile as sf
import matplotlib.pyplot as plt
import torchopenl3
from librosa import display
from audio_features import make_spectrogram, make_log_spectrogram, make_logmel_spectrogram, make_mel_spectrogram
from train import OpenL3Classifier
import pandas as pd
import numpy as np

plt.style.use("ggplot")


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


def make_prediction(audio_file, sampling_rate, net):
    feature, _ = torchopenl3.get_audio_embedding(audio_file, sampling_rate, content_type="env", embedding_size=512)
    feature = feature.mean(axis=1).cpu()
    return net(feature).detach().numpy()


def process_prediction(prediction, mapping, top_k=5):
    indices = prediction.squeeze().argsort()[-top_k:]
    classes = list(mapping.loc[indices]["category"])
    classes = [x.replace("_", " ").title() for x in classes]
    prediction = list(zip(classes, np.take(prediction, indices)))
    return prediction


def setup_environment():
    loaded_model = OpenL3Classifier(512, 50, 1e-3, 1).load_from_checkpoint(
        "models/fold4/checkpoints/epoch=149-step=599.ckpt",
        map_location={"cuda:0": "cpu"}
    )
    loaded_model.eval()

    classes = pd.read_csv("class_map.csv")
    return loaded_model, classes


if __name__ == '__main__':

    model, class_map = setup_environment()

    st.header("Upload .wav audio file")
    uploaded_file = st.file_uploader("", type="wav")

    window = st.sidebar.selectbox("Choose window size for STFT", options=[128, 1024, 2048, 4096])
    melbins = st.sidebar.selectbox("Choose number of Mel bins", options=[32, 64, 128])
    top_k_classes = st.sidebar.slider("Choose number of top K classes", min_value=1, max_value=50, value=5)

    if uploaded_file is not None:
        if uploaded_file.type != "audio/wav":
            st.warning("Wrong file type")
        else:
            audio, sample_rate = sf.read(uploaded_file)

            st.header("Listen to your audio file")
            st.audio(uploaded_file)

            st.header("Waveform")
            fig = plt.figure(figsize=(10, 5))
            display.waveplot(audio, sr=sample_rate)
            st.pyplot(fig=fig)

            st.header("Feature plots")
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

            st.header("Classification")
            answer = make_prediction(audio, sample_rate, model)
            top_k_predictions = pd.DataFrame(
                process_prediction(answer, class_map, top_k=top_k_classes),
                columns=["Class", "Probability"]
            )

            fig = plt.figure(figsize=(10, 5))
            plt.barh(top_k_predictions["Class"], top_k_predictions["Probability"])
            st.pyplot(fig=fig)
