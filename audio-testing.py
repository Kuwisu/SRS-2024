import soundfile as sf
import scipy
import soxr
import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.mlab import window_hanning
from matplotlib.ticker import ScalarFormatter
from scipy.signal.windows import *
from scipy.signal import ShortTimeFFT
from scipy.signal import spectrogram

# %% Read function using soundfile vs using Librosa
data, samplerate = sf.read('playback.wav', dtype='float32', always_2d=True)
y = (data[:,0] + data[:,1]) / 2
print(y)

x, sr = librosa.load('playback.wav', sr=None, mono=True)
print(x)

# %% Lower quality resampling using scipy vs Librosa
# y1 = scipy.signal.resample(y, int ((len(y) / samplerate) * 22050))
# print(y1)
#
# x1 = librosa.resample(x, orig_sr=sr, target_sr=22050, res_type='fft')
# print(x1)
# print(len(x1))

# %% Higher quality resampling using SOXR vs Librosa
y2 = soxr.resample(y, samplerate, 22050, 'HQ')
print(y2)
print(len(y2))

# x2 = librosa.resample(x, orig_sr=sr, target_sr=22050)
# print(x2)
# print(len(x2))

# %% Prepare plot
fig, ax = plt.subplots(nrows=2)

# %% Plotting using plt.step vs waveshow
times = np.linspace(0, len(y2) / 22050, len(y2), dtype='float32')
# ax[0].step(times, y2, where='post')

# librosa.display.waveshow(y2, sr=22050, ax=ax[1], where='post')

# %% Plotting a linear spectrogram using librosa vs scipy, both using librosa.stft
# Librosa spectrogram
s = librosa.stft(y2, n_fft=512, hop_length=128, win_length=512, window='hann')
s_db = librosa.amplitude_to_db(np.abs(s), ref=np.max)
# img = librosa.display.specshow(s_db, sr=22050, n_fft=512, hop_length=128, win_length=512,
#                                ax=ax[0], cmap='viridis', x_axis='time', y_axis='linear')
# plt.colorbar(img, ax=ax[0])

# Scipy spectrogram
w = hann(512)
SFT = ShortTimeFFT(win=w, hop=128, fs=22050, fft_mode='onesided', mfft=512)
Sx = SFT.spectrogram(y2)
Sx_db = 10 * np.log10(np.fmax(Sx, 1e-5)) - 30
# im1 = ax[1].imshow(Sx_db, origin='lower', aspect='auto', cmap='viridis',
#                    extent=SFT.extent(len(y2)))
# plt.colorbar(im1, ax=ax[1])

# %% Plot the same but with a log magnitude spectrogram
# Librosa spectrogram
img = librosa.display.specshow(s_db, sr=22050, n_fft=512, hop_length=128, win_length=512,
                               ax=ax[0], cmap='viridis', x_axis='time', y_axis='log')
plt.colorbar(img, ax=ax[0])

# Scipy spectrogram
im1 = ax[1].imshow(Sx_db, origin='lower', aspect='auto', cmap='viridis',
                   extent=SFT.extent(len(y2)))
ticks = []
tick = 64
while tick < int(22050/2):
    ticks.append(tick)
    tick = tick * 2

plt.yscale('symlog')
plt.yticks(ticks)
plt.ylim(32, int(22050/2))
ax[1].yaxis.set_major_formatter(ScalarFormatter())
plt.colorbar(im1, ax=ax[1])

# %% Display plot
plt.show()
