import soundfile as sf
import soxr
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
from librosa.feature import melspectrogram
from matplotlib.ticker import ScalarFormatter
from scipy.signal.windows import *
from scipy.signal import ShortTimeFFT

# %% Parameters
SR = 22050
HOP = 128
FFT = 512
WIN = 512
N_MELS = 80
F_MAX_HZ = SR / 2
CMAP = 'viridis'

# %% Conversion function
def hz_to_mels(freq):
    return 2595 * np.log10(1 + freq/700)

def mels_to_hz(freq):
    return 700 * (10**(freq/2595) - 1)

# %% Mel Filterbank function
def construct_filterbank():
    # Retrieve the center frequencies of mel banks, then convert them to FFT bins
    mel_banks = np.linspace(0, hz_to_mels(F_MAX_HZ), N_MELS+2, dtype=np.float32)
    mel_banks = np.floor((FFT+1) * mels_to_hz(mel_banks) / SR).astype(int)

    # Construct a mel transformation matrix
    weights = np.zeros((N_MELS, int(FFT//2 + 1)), dtype=np.float32)
    for i in range(1, N_MELS+1):
        for j in range(1, int(FFT//2 + 1)):
            if mel_banks[i - 1] <= j <= mel_banks[i]:
                den = mel_banks[i] - mel_banks[i - 1]
                if den != 0:
                    weights[i - 1, j] = (j - mel_banks[i - 1]) / den
            elif mel_banks[i] < j < mel_banks[i + 1]:
                den = mel_banks[i + 1] - mel_banks[i]
                if den != 0:
                    weights[i - 1, j] = (mel_banks[i + 1] - j) / den

    return weights / (np.sum(weights, axis=1, keepdims=True)+1e-10)


# %% Spectrogram function
def show_spectrogram(y, scale, axis):
    w = hann(WIN)
    SFT = ShortTimeFFT(win=w, hop=HOP, fs=SR, fft_mode='onesided', mfft=FFT)
    Sx = np.abs(SFT.stft(y, p0=0, p1=int(np.ceil(len(y)/HOP)))) ** 2
    Sx_db = 10 * np.log10(np.fmax(Sx, 1e-5))
    Sx_db = Sx_db - Sx_db.max()

    if scale == 'mel':
        Sx_db = np.dot(construct_filterbank(), Sx_db)
        ax[1].set_yscale(value='symlog', base=2, linthresh=1000)
    elif scale == 'log':
        ax[1].set_yscale(value='symlog', base=2, linthresh=64, linscale=0.5)

    ax[1].yaxis.set_major_formatter(ScalarFormatter())
    x_labels = np.linspace(0, len(y)/SR, Sx_db.shape[1], dtype=np.float32)

    if scale == 'mel':
        y_labels = np.linspace(0, hz_to_mels(F_MAX_HZ), Sx_db.shape[0], dtype=np.float32)
        y_labels = mels_to_hz(y_labels)
    else:
        y_labels = np.linspace(0, F_MAX_HZ, Sx_db.shape[0], dtype=np.float32)

    img = axis.pcolormesh(x_labels, y_labels, Sx_db, cmap=CMAP)
    plt.colorbar(img, ax=axis, format='%+2.0f dB')


# %% Librosa spectrogram function for comparison reasons
def librosa_spectrogram(y, scale, axis):
    if scale == 'mel':
        s = melspectrogram(y=y, sr=SR, hop_length=HOP, win_length=WIN, n_fft=FFT, n_mels=N_MELS, window='hann', htk=True)
        s_db = librosa.power_to_db(np.abs(s), ref=np.max)
    else:
        s = librosa.stft(y=y, hop_length=HOP, win_length=WIN, n_fft=FFT, window='hann')
        s_db = librosa.amplitude_to_db(np.abs(s), ref=np.max)

    htk = scale == 'mel'
    img = librosa.display.specshow(s_db, sr=SR, hop_length=HOP, win_length=WIN, n_fft=FFT, x_axis='time', y_axis=scale, cmap=CMAP, ax=axis, htk=htk)
    plt.colorbar(img, format='%+2.0f dB', ax=axis)


# %% Read in and resample sound file
data, samplerate = sf.read('playback.wav', dtype='float32', always_2d=True)
y = (data[:,0] + data[:,1]) / 2
y = soxr.resample(y, samplerate, SR, 'HQ')

fig, ax = plt.subplots(nrows=2)

scale = 'mel'
librosa_spectrogram(y, scale=scale, axis=ax[0])
show_spectrogram(y, scale=scale, axis=ax[1])

plt.show()