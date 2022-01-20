# Audio classification with PyTorch Lightning and OpenL3

This is a source code for the tutorial about audio processing, audio classification and simple audio app building.
The tutorial will be published soon and the link will be updated.

## Data

In the project I'm using ESC-50 dataset.
It can be downloaded [here](https://github.com/karolpiczak/ESC-50).

## Plot generation

Plots used in the tutorial where generated using 2 files:
1. [feature plots](generate_feature_plots.py) - plots of STFT and mel spectrograms,
2. [explanation plots](generate_feature_plots.py) - plots of waves and FFT result.

## Audio classification

Simple classifier was trained using [train.py](train.py). For list of args run:
```shell script
python train.py -h
```

## Streamlit app

The outcomes of the project are summarised in the Streamlit App where you can plot features and run classification based on your own files.
In order to open app run:
```shell script
streamlit run app.py
``` 