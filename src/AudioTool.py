import numpy as np
import soundfile as sf
import soxr
from matplotlib.ticker import ScalarFormatter
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

from utils.conversion import mels_to_hz, hz_to_mels


class AudioTool:
    """
    This class manages graphing operations.
    Its methods take in waveform/spectrogram parameters and then
    applies them to an MPL figure that is passed in on initialisation.
    """

    def __init__(self, figure):
        # Draw a pair of axes for the waveform and spectrogram
        self.figure = figure
        self.figure.subplots(nrows=2, sharex=True)

        # Initialise values defined in other functions
        self.file_name = ''
        self.y = None
        self.sr = None
        self.native_sr = None
        self.colorbar = None

    def get_file_name(self):
        """ Retrieve the file name without the directory details. """
        return self.file_name.split('/')[-1]

    def get_native_rate(self):
        """ Retrieve the native sampling rate of the audio file. """
        return self.native_sr

    def get_sample_rate(self):
        """ Retrieve the sampling rate after resampling operations. """
        return self.sr

    def load_file(self, file_name):
        """
        Load the file with the given name using its native sampling rate into
        a numpy array representing the audio time series.
        If the file is stereo, average the channels and indicate this.

        :param file_name: the path to access an audio file
        :return whether the provided signal was mono or stereo
        """
        is_mono = True
        self.file_name = file_name
        self.y, self.native_sr = sf.read(self.file_name, dtype='float32')
        if self.y.ndim > 1:
            # This line of code was developed using generative AI.
            self.y = (self.y[:, 0] + self.y[:, 1]) / 2
            is_mono = False

        self.sr = self.native_sr
        return is_mono

    def produce_waveform(self):
        """ Graph the amplitude over time waveform of the loaded array. """
        self.figure.axes[0].cla()
        times = np.linspace(0, len(self.y) / self.sr, len(self.y), dtype='float32')
        self.figure.axes[0].step(times, self.y, where='post')
        self.figure.axes[0].set_title(f"Waveform of {self.get_file_name()} at {self.sr}Hz")
        self.figure.axes[0].set_ylabel('Amplitude')
        self.figure.axes[0].label_outer()

    def produce_spectrogram(self, n_fft=1024, hop_length=256, window=hann(1024),
                            scale='mel', n_mels=80, cmap='grey'):
        """
        Produce a spectrogram of the loaded array with a set of provided details.

        :param n_fft: the number of bins used to divide the window
        :param hop_length: the distance between each window
        :param window: an array containing a window function to apply in the STFT
        :param scale: whether the spectrogram is mel, linear, or logarithmic
        :param n_mels: the number of bands in the mel filter bank
        :param cmap: the colour map used to display the spectrogram
        """
        # Clear all pre-existing information
        if self.colorbar is not None:
            self.colorbar.remove()
        self.figure.axes[1].cla()

        # Create the spectrogram by taking the absolute square of the STFT
        sft = ShortTimeFFT(win=window, hop=hop_length, fs=self.sr, fft_mode='onesided', mfft=n_fft)
        sx = np.abs(sft.stft(self.y, p0=0, p1=int(np.ceil(len(self.y) / hop_length)))) ** 2

        # Convert the spectrogram into decibels
        sx_db = 10 * np.log10(np.fmax(sx, 1e-5))
        sx_db = sx_db - sx_db.max()

        if scale == 'mel':
            sx_db = np.dot(self.construct_filterbank(n_fft, n_mels), sx_db)
            self.figure.axes[1].set_yscale(value='symlog', base=2, linthresh=1000)
        elif scale == 'logarithmic':
            self.figure.axes[1].set_yscale(value='symlog', base=2, linthresh=64, linscale=0.5)

        # Display the colour bar in fixed ticks
        interval = 10
        lowest_tick = sx_db.min() + interval - (sx_db.min() % interval)
        num_ticks = int(np.abs(lowest_tick) / interval) + 1
        colorbar_ticks = np.linspace(lowest_tick, 0, num_ticks, dtype=sx_db.dtype)

        self.figure.axes[1].yaxis.set_major_formatter(ScalarFormatter())
        time_steps = np.linspace(0, len(self.y) / self.sr, sx_db.shape[1], dtype=np.float32)

        f_max = self.sr / 2
        if scale == 'mel':
            freq_steps = np.linspace(0, hz_to_mels(f_max), sx_db.shape[0], dtype=np.float32)
            freq_steps = mels_to_hz(freq_steps)
        else:
            freq_steps = np.linspace(0, f_max, sx_db.shape[0], dtype=np.float32)

        # Create a spectrogram image and colourbar
        img = self.figure.axes[1].pcolormesh(time_steps, freq_steps, sx_db, cmap=cmap)
        self.colorbar = self.figure.colorbar(img, ax=self.figure.axes[1], format='%+2.0f dB')
        self.colorbar.ax.get_yaxis().set_ticks(colorbar_ticks)

    def resample(self, target_sr=22050):
        """ Change the sampling rate of the numpy array. """
        # If attempting to resample to a larger sampling rate, reload the
        # original recording to prevent or reduce loss of data.
        if target_sr > self.sr:
            self.load_file(self.file_name)
        self.y = soxr.resample(self.y, self.sr, target_sr, 'HQ')
        self.sr = target_sr

    def construct_filterbank(self, n_fft, n_mels):
        """
        Create a transformation matrix to convert a spectrogram into a mel
        spectrogram.

        :param n_fft: the number of bins used to divide the window
        :param n_mels: the number of bands in the mel filter bank
        :return: the normalised transformation matrix
        """
        f_max = self.sr / 2
        # Retrieve the center frequencies of mel banks and convert them to Hz
        mel_banks = np.linspace(0, hz_to_mels(f_max), n_mels+2, dtype=np.float32)
        mel_banks_hz = mels_to_hz(mel_banks)

        # Retrieve the center frequencies of FFT bins
        fftfreqs = np.linspace(0, f_max, n_fft//2 + 1)

        # Construct a mel transformation matrix
        weights = np.zeros((n_mels, int(n_fft//2 + 1)), dtype=np.float32)
        for i in range(1, n_mels+1):
            lower = mel_banks_hz[i-1]
            center = mel_banks_hz[i]
            upper = mel_banks_hz[i+1]

            for j in range(len(fftfreqs)):
                if lower <= fftfreqs[j] <= center:
                    weights[i - 1, j] = (fftfreqs[j] - lower) / (center - lower)
                if center <= fftfreqs[j] <= upper:
                    weights[i - 1, j] = (upper - fftfreqs[j]) / (upper - center)

        # Normalise the matrix, adding a small value to prevent division errors
        return weights / (np.sum(weights, axis=1, keepdims=True)+1e-10)

