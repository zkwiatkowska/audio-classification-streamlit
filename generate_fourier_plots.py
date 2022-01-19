from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq

if __name__ == '__main__':

    plt.style.use("ggplot")
    Path("plots").mkdir(exist_ok=True)

    A = 2 * np.pi
    x = np.linspace(0, 10, 10 * 50)
    sinex = np.sin(A * x)
    sine2x = np.sin(A * 2 * x)
    sine3x = np.sin(A * 3 * x)
    sine_sum = sinex + sine2x + sine3x
    sine_sum /= sine_sum.max()
    plots = {
        "y = sin(2πx)": sinex,
        "y = sin(4πx)": sine2x,
        "y = sin(6πx)": sine3x,
        "y = peak_norm[sin(2πx) + sin(4πx) + sin(6πx)]": sine_sum
    }

    fig, ax = plt.subplots(4, 1, figsize=(10, 5), sharex=True)

    for ix, (name, plot) in enumerate(plots.items()):
        ax[ix].plot(x, plot)
        ax[ix].set_title(name, fontdict={'size': 10})
        ax[ix].set_ylabel("Amplitude", fontdict={'size': 8})

    plt.xlabel("Time", fontdict={'size': 8})
    fig.subplots_adjust(hspace=0.9)
    fig.savefig("plots/sine.png", dpi=800)

    y = rfft(sine_sum)
    x = rfftfreq(len(sine_sum), 1 / 50)

    fig = plt.figure(figsize=(10, 5))
    plt.plot(x[:50], np.abs(y)[:50])
    plt.xlabel("Frequency", fontdict={'size': 8})
    plt.ylabel("Magnitude", fontdict={'size': 8})
    plt.savefig("plots/fft.png", dpi=800)
