import soundfile as sf
import scipy
import soxr
import librosa
import librosa.feature
import matplotlib.pyplot as plt
import numpy as np
from pyfilterbank import melbank
from matplotlib.mlab import window_hanning
from matplotlib.ticker import ScalarFormatter
from numpy.f2py.f90mod_rules import fgetdims2_sa
from scipy.signal.windows import *
from scipy.signal import ShortTimeFFT
from scipy.signal import spectrogram

# %% Read function using soundfile vs using Librosa
data, samplerate = sf.read('playback.wav', dtype='float32', always_2d=True)
y = (data[:,0] + data[:,1]) / 2

x, sr = librosa.load('playback.wav', sr=None, mono=True)

# %% Lower quality resampling using scipy vs Librosa
# y1 = scipy.signal.resample(y, int ((len(y) / samplerate) * 22050))
# print(y1)
#
# x1 = librosa.resample(x, orig_sr=sr, target_sr=22050, res_type='fft')
# print(x1)
# print(len(x1))

# %% Higher quality resampling using SOXR vs Librosa
y2 = soxr.resample(y, samplerate, 22050, 'HQ')

# x2 = librosa.resample(x, orig_sr=sr, target_sr=22050)

# %% Prepare plot
fig, ax = plt.subplots(nrows=2)

# %% Plotting using plt.step vs waveshow
times = np.linspace(0, len(y2) / 22050, len(y2), dtype='float32')
# ax[0].step(times, y2, where='post')
# ax[0].set_xlabel('Time (s)')
# ax[0].set_ylabel('Amplitude')

# librosa.display.waveshow(y2, sr=22050, ax=ax[1], where='post')

# %% Plotting a linear spectrogram using librosa vs scipy, both using librosa.stft
# # Librosa spectrogram
# s = librosa.stft(y2, n_fft=512, hop_length=128, win_length=512, window='hann')
# s_db = librosa.amplitude_to_db(np.abs(s), ref=np.max)
# img = librosa.display.specshow(s_db, sr=22050, n_fft=512, hop_length=128, win_length=512,
#                                ax=ax[0], cmap='viridis', x_axis='time', y_axis='linear')
# plt.colorbar(img, ax=ax[0])
#
# # Scipy spectrogram
# w = hann(512)
# SFT = ShortTimeFFT(win=w, hop=128, fs=22050, fft_mode='onesided', mfft=512)
# Sx = np.abs(SFT.stft(y2, p0=0, p1=int(np.ceil(len(y2)/128))))**2
# Sx_db = 10 * np.log10(np.fmax(Sx, 1e-5))
# Sx_db = Sx_db - Sx_db.max()
#
# _, _, f0, f1 = SFT.extent(len(y2))
# im1 = ax[1].imshow(Sx_db, origin='lower', aspect='auto', cmap='viridis',
#                    extent=(0, len(y2)/22050, f0, f1))
# plt.colorbar(im1, ax=ax[1])

# %% Plot the same but with a log magnitude spectrogram
# # Librosa spectrogram
# s = librosa.stft(y2, n_fft=512, hop_length=128, win_length=512, window='hann')
# s_db = librosa.amplitude_to_db(np.abs(s), ref=np.max)
# img = librosa.display.specshow(s_db, sr=22050, n_fft=512, hop_length=128, win_length=512,
#                                ax=ax[0], cmap='viridis', x_axis='time', y_axis='log')
# plt.colorbar(img, ax=ax[0])
#
# # Scipy spectrogram
# w = hann(512)
# SFT = ShortTimeFFT(win=w, hop=128, fs=22050, fft_mode='onesided', mfft=512)
# Sx = np.abs(SFT.stft(y2, p0=0, p1=int(np.ceil(len(y2)/128))))**2
# Sx_db = 10 * np.log10(np.fmax(Sx, 1e-5))
# Sx_db = Sx_db - Sx_db.max()
#
# # Log spectrogram inserts details here
# ax[1].set_yscale(value='symlog', base=2, linthresh=64, linscale=0.5)
# ax[1].yaxis.set_major_formatter(ScalarFormatter())
#
# _, _, f0, f1 = SFT.extent(len(y2))
# im1 = ax[1].imshow(Sx_db, origin='lower', aspect='auto', cmap='viridis',
#                    extent=(0, len(y2)/22050, f0, f1))
# plt.colorbar(im1, ax=ax[1])

# %% Plot a mel spectrogram
# # Librosa spectrogram
# s = librosa.feature.melspectrogram(y=y2, sr=22050, n_fft=512, hop_length=128, win_length=512, window='hann', n_mels=80)
# s_db = librosa.power_to_db(np.abs(s), ref=np.max)
# img = librosa.display.specshow(s_db, sr=22050, n_fft=512, hop_length=128, win_length=512,
#                                cmap='viridis', x_axis='time', y_axis='mel', ax=ax[0])
# plt.colorbar(img, ax=ax[0])
# ax[0].set_ylabel('Frequency (Hz)')
# ax[0].set_xlabel('Time (s)')

# Librosa filterbank
mel = librosa.filters.mel(sr=22050, n_fft=512, n_mels=80, htk=True)

# Scipy spectrogram
w = hann(512)
SFT = ShortTimeFFT(win=w, hop=128, fs=22050, fft_mode='onesided', mfft=512)
Sx = np.abs(SFT.stft(y2, p0=0, p1=int(np.ceil(len(y2)/128))))**2
Sx_db = 10 * np.log10(np.fmax(Sx, 1e-5))
Sx_db = Sx_db - Sx_db.max()

# Mel spectrogram inserts details here
n_mels = 80
# Convert lowest and highest frequency to mel
f_min, f_max, _, _ = SFT.extent(len(y2), axes_seq='ft')
f_min_mels = 2595 * np.log10(1 + f_min / 700)
f_max_mels = 2595 * np.log10(1 + f_max / 700)
# Create equally spaced points in Hz for the mel filterbanks
mel_banks = np.linspace(f_min_mels, f_max_mels, n_mels+2, dtype='float32')
mel_banks_hz = 700*(10**(mel_banks / 2595) - 1)
# Convert these to FFT bins
mel_banks_bins = np.floor((512+1)*mel_banks_hz / 22050).astype('int')
# Create our weights matrix
weights = np.zeros((n_mels, int(512//2 + 1)), dtype='float32')
for i in range(1, n_mels + 1):
    for j in range(int(512//2 + 1)):
        if mel_banks_bins[i-1] <= j <= mel_banks_bins[i]:
            den = mel_banks_bins[i] - mel_banks_bins[i-1]
            if den != 0:
                weights[i-1, j] = (j - mel_banks_bins[i-1]) / den
        elif mel_banks_bins[i] < j < mel_banks_bins[i+1]:
            den = mel_banks_bins[i+1] - mel_banks_bins[i]
            if den != 0:
                weights[i-1, j] = (mel_banks_bins[i+1] - j) / den

ax[0].set_yscale(value='symlog', base=2, linthresh=1000, linscale=1)
ax[0].yaxis.set_major_formatter(ScalarFormatter())

ax[1].set_yscale(value='symlog', base=2, linthresh=1000, linscale=1)
ax[1].yaxis.set_major_formatter(ScalarFormatter())

# Code borrowed from Librosa
weights = weights / (np.sum(weights, axis=1, keepdims=1) + 1e-10)

# # Plot the filterbanks
# ax[0].plot(mel.T)
# ax[1].plot(weights.T)

# Plot a spectrogram using mel filterbanks
melspec = np.matmul(mel, Sx_db)
_, _, f0, f1 = SFT.extent(len(y2))
im1 = ax[0].imshow(melspec, origin='lower', aspect='auto', cmap='viridis',
                   extent=(0, len(y2)/22050, f0, f1))
plt.colorbar(im1, ax=ax[0])

melspec = np.matmul(weights, Sx_db)
_, _, f0, f1 = SFT.extent(len(y2))
im1 = ax[1].imshow(melspec, origin='lower', aspect='auto', cmap='viridis',
                   extent=(0, len(y2)/22050, f0, f1))
plt.colorbar(im1, ax=ax[1])


# %% Display plot
plt.show()
