import soundfile as sf
import scipy
import soxr
import librosa
import matplotlib.pyplot as plt
import numpy as np

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

# %% Plotting using matplotlib vs waveshow
fig, ax = plt.subplots(nrows=2)
times = np.linspace(0, len(y2) / 22050, len(y2), dtype='float32')

ax[0].step(times, y2, where='post')
librosa.display.waveshow(y2, sr=22050, ax=ax[1], where='post')
plt.show()

