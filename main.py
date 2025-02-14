import matplotlib.pyplot as plt
import numpy as np
import pathlib
import soundfile as sf
import soxr
import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from matplotlib.ticker import ScalarFormatter
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import *

SPEC_SCALES = ["Mel", "Linear", "Logarithmic"]
# Map the text displayed in the window combo box to a scipy window function.
WINDOW_FUNCTIONS = {"Hanning": hann,
                    "Hamming": hamming,
                    "Blackman": blackman,
                    "Blackman-Harris": blackmanharris,
                    "Flat Top": flattop,
                    "Parzen": parzen,
                    "Triangular": triang,
                    "Rectangular": boxcar}
# Colour maps chosen are ones that blend from one colour to a different one.
COLOUR_MAPS = ["Gray", "Magma", "Inferno", "Plasma", "Viridis", "Cividis"]

def hz_to_mels(freq):
    """ Convert a frequency or array of frequencies to the Mel scale. """
    return 2595 * np.log10(1 + freq/700)

def mels_to_hz(freq):
    """ Convert a frequency or array of frequencies in the Mel scale to Hz. """
    return 700 * (10**(freq/2595) - 1)

def retrieve_int_field(line_edit, is_valid):
    """
    Text entries are checked to make sure they are valid integers.
    If not, the text box is cleared and coloured red to signal an error.

    :param line_edit: the text field to validate
    :param is_valid: whether the overall form is currently valid
    :return: True if integer or disabled, False otherwise; also return
    the text contained in the line edit
    """
    try:
        value = int(line_edit.text())
    except ValueError:
        value = 0
        if line_edit.isEnabled():
            line_edit.setText("")
            line_edit.setStyleSheet("border: 1px solid red;")
            return False, value
    return is_valid, value

class TabWidget(QTabWidget):
    """
    This class overrides the sizing method that QTabWidget uses so that
    it will automatically resize to fit the current widget rather than
    fitting the size of the largest widget.

    This code has been derived from user musicamante's implementation
    Link: https://stackoverflow.com/a/66053591
    """

    def __init__(self, *args, **kwargs):
        super(TabWidget, self).__init__(**kwargs)
        self.currentChanged.connect(self.updateGeometry)

    def minimumSizeHint(self):
        return self.sizeHint()

    def sizeHint(self):
        widget_size = self.currentWidget().sizeHint()
        tab_size = self.tabBar().sizeHint()
        size = QSize(
            max(widget_size.width(), tab_size.width()),
            widget_size.height() + tab_size.height()
        )
        return size

class AudioTool:
    """
    This class manages graphing operations.
    Its methods take in waveform/spectrogram parameters and then
    applies them to an MPL figure that is passed in on initialisation.
    """

    def __init__(self, figure):
        # Draw a pair of axes for the waveform and spectrogram
        self.figure = figure
        self.ax = self.figure.subplots(nrows=2, sharex=True)

        # Initialise values defined in other functions
        self.fileName = ''
        self.y = None
        self.orig_sr = None
        self.sr = None
        self.colorbar = None

    def getFileName(self):
        """ Retrieve the file name without the directory details. """
        return self.fileName.split('/')[-1]

    def getOriginalSampleRate(self):
        """ Retrieve the native sampling rate of the audio file. """
        return self.orig_sr

    def getCurrentSampleRate(self):
        """ Retrieve the sampling rate after resampling operations. """
        return self.sr

    def loadFile(self, file_name):
        """
        Load the file with the given name using its native sampling rate into
        a numpy array representing the audio time series.
        If the file is stereo, average the channels and indicate this.

        :param file_name: the path to access an audio file
        :return whether the provided signal was mono or stereo
        """
        is_mono = True
        self.fileName = file_name
        self.y, self.orig_sr = sf.read(self.fileName, dtype='float32')
        if self.y.ndim > 1:
            # This line of code was developed using generative AI.
            self.y = (self.y[:, 0] + self.y[:, 1]) / 2
            is_mono = False

        self.sr = self.orig_sr
        return is_mono

    def produceWaveform(self):
        """ Graph the amplitude over time waveform of the loaded array. """
        self.ax[0].cla()
        times = np.linspace(0, len(self.y) / self.sr, len(self.y), dtype='float32')
        self.ax[0].step(times, self.y, where='post')
        self.ax[0].set_title(f"Waveform of {self.getFileName()} at {self.sr}Hz")
        self.ax[0].set_ylabel('Amplitude')
        self.ax[0].label_outer()

    def produceSpectrogram(self, n_fft, hop_length, window, scale, n_mels, cmap):
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
        self.ax[1].cla()

        # Create the spectrogram by taking the absolute square of the STFT
        SFT = ShortTimeFFT(win=window, hop=hop_length, fs=self.sr, fft_mode='onesided', mfft=n_fft)
        Sx = np.abs(SFT.stft(self.y, p0=0, p1=int(np.ceil(len(self.y) / hop_length)))) ** 2

        # Convert the spectrogram into decibels
        Sx_db = 10 * np.log10(np.fmax(Sx, 1e-5))
        Sx_db = Sx_db - Sx_db.max()

        if scale == 'mel':
            Sx_db = np.dot(self.construct_filterbank(n_fft, n_mels), Sx_db)
            self.ax[1].set_yscale(value='symlog', base=2, linthresh=1000)
        elif scale == 'logarithmic':
            self.ax[1].set_yscale(value='symlog', base=2, linthresh=64, linscale=0.5)

        self.ax[1].yaxis.set_major_formatter(ScalarFormatter())
        x_labels = np.linspace(0, len(self.y) / self.sr, Sx_db.shape[1], dtype=np.float32)

        f_max = self.sr / 2
        if scale == 'mel':
            y_labels = np.linspace(0, hz_to_mels(f_max), Sx_db.shape[0], dtype=np.float32)
            y_labels = mels_to_hz(y_labels)
        else:
            y_labels = np.linspace(0, f_max, Sx_db.shape[0], dtype=np.float32)

        img = self.ax[1].pcolormesh(x_labels, y_labels, Sx_db, cmap=cmap)
        self.colorbar = self.figure.colorbar(img, ax=self.ax[1], format='%+2.0f dB')

    def resample(self, target_sr):
        """ Change the sampling rate of the numpy array. """
        # If attempting to resample to a larger sampling rate, reload the
        # original recording to prevent or reduce loss of data.
        if target_sr > self.sr:
            self.loadFile(self.fileName)
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
        # Retrieve the center frequencies of mel banks, then convert them to FFT bins
        f_max = self.sr / 2
        mel_banks = np.linspace(0, hz_to_mels(f_max), n_mels+2, dtype=np.float32)
        mel_banks = np.floor((n_fft+1) * mels_to_hz(mel_banks) / self.sr).astype(int)

        # Construct a mel transformation matrix
        weights = np.zeros((n_mels, int(n_fft//2 + 1)), dtype=np.float32)
        for i in range(1, n_mels+1):
            for j in range(1, int(n_fft//2 + 1)):
                if mel_banks[i - 1] <= j <= mel_banks[i]:
                    den = mel_banks[i] - mel_banks[i - 1]
                    if den != 0:
                        weights[i - 1, j] = (j - mel_banks[i - 1]) / den
                elif mel_banks[i] < j < mel_banks[i + 1]:
                    den = mel_banks[i + 1] - mel_banks[i]
                    if den != 0:
                        weights[i - 1, j] = (mel_banks[i + 1] - j) / den

        # Normalise the matrix, adding a small value to prevent division errors
        return weights / (np.sum(weights, axis=1, keepdims=True)+1e-10)

class UI(QMainWindow):
    """
    Defines the main window and all UI interactions.
    """

    def __init__(self):
        super(UI, self).__init__()

        # FILE
        uic.loadUi('ui-25-01.ui', self)
        central_widget = self.findChild(QWidget, 'centralwidget')

        # MENU BAR
        action_new = self.findChild(QAction, 'actionNew')
        action_open = self.findChild(QAction, 'actionOpen')
        action_save_png = self.findChild(QAction, 'actionSavePng')
        action_new.triggered.connect(self.reset)
        action_open.triggered.connect(self.import_file)
        action_save_png.triggered.connect(self.export_graph)

        # SOUND IMPORT TAB
        sound_import_frame = QFrame(central_widget)
        sound_import_frame.setAutoFillBackground(True)
        sound_import_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        open_file_button = QPushButton('Select Audio File')
        open_file_button.clicked.connect(self.import_file)

        self.file_label = QLabel('No file selected')
        self.file_label.setAlignment(Qt.AlignCenter)

        self.channel_label = QLabel('')
        self.channel_label.setAlignment(Qt.AlignCenter)

        alert_label = QLabel('This will clear all existing graphs.')
        alert_label.setAlignment(Qt.AlignCenter)
        alert_label.setStyleSheet("color: red;")

        sound_import_form = QFormLayout(sound_import_frame)
        sound_import_form.addRow(self.file_label)
        sound_import_form.addRow(self.channel_label)
        sound_import_form.addRow(open_file_button)
        sound_import_form.addRow(alert_label)

        # RESAMPLING TAB
        resample_frame = QFrame(central_widget)
        resample_frame.setAutoFillBackground(True)
        resample_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.sample_rate_field = QLineEdit("22050")
        self.sample_rate_field.setValidator(QIntValidator())
        self.sample_rate_field.textEdited.connect(
            lambda: self.sample_rate_field.setStyleSheet(QLineEdit().styleSheet()))

        self.resample_button = QPushButton('Resample')
        self.resample_button.clicked.connect(self.resample_wave)
        self.resample_button.setEnabled(False)

        self.native_rate_label = QLabel('No file selected')
        self.sample_rate_label = QLabel('')

        alert_label = QLabel('This will not affect an existing spectrogram.')
        alert_label.setAlignment(Qt.AlignCenter)
        alert_label.setStyleSheet("color: red;")

        alert_label2 = QLabel('Create a new one to see the effects of the resampling.')
        alert_label2.setAlignment(Qt.AlignCenter)
        alert_label2.setStyleSheet("color: red;")

        resample_form = QFormLayout(resample_frame)
        resample_form.addRow(self.native_rate_label)
        resample_form.addRow(self.sample_rate_label)
        resample_form.addRow(QLabel('New Sampling Rate (Hz)'), self.sample_rate_field)
        resample_form.addRow(self.resample_button)
        resample_form.addRow(alert_label)
        resample_form.addRow(alert_label2)

        # SPECTROGRAM SETTINGS FORM
        spectrogram_frame = QFrame(central_widget)
        spectrogram_frame.setAutoFillBackground(True)
        spectrogram_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.fft_field = QLineEdit("1024")
        self.fft_field.setValidator(QIntValidator())
        self.fft_field.textEdited.connect(
            lambda: self.fft_field.setStyleSheet(QLineEdit().styleSheet()))

        self.hop_field = QLineEdit("256")
        self.hop_field.setValidator(QIntValidator())
        self.hop_field.textEdited.connect(
            lambda: self.hop_field.setStyleSheet(QLineEdit().styleSheet()))

        self.window_length_field = QLineEdit("1024")
        self.window_length_field.setValidator(QIntValidator())
        self.window_length_field.textEdited.connect(
            lambda: self.window_length_field.setStyleSheet(QLineEdit().styleSheet()))

        self.mel_field = QLineEdit("80")
        self.mel_field.setValidator(QIntValidator())
        self.mel_field.textEdited.connect(
            lambda: self.mel_field.setStyleSheet(QLineEdit().styleSheet()))

        self.window_combobox = QComboBox()
        for window in WINDOW_FUNCTIONS.keys():
            self.window_combobox.addItem(window)

        self.scale_combobox = QComboBox()
        self.scale_combobox.activated[str].connect(
            lambda text: self.mel_field.setEnabled(text == "Mel"))
        for scale in SPEC_SCALES:
            self.scale_combobox.addItem(scale)

        self.colour_combobox = QComboBox()
        for colour in COLOUR_MAPS:
            self.colour_combobox.addItem(colour)
        self.reverse_checkbox = QCheckBox()

        self.spectrogram_button = QPushButton('Generate Spectrogram')
        self.spectrogram_button.clicked.connect(self.generate_spectrogram)
        self.spectrogram_button.setEnabled(False)

        spectrogram_form = QFormLayout(spectrogram_frame)
        spectrogram_form.addRow(QLabel("FFT Size (samples)"), self.fft_field)
        spectrogram_form.addRow(QLabel("Hop Size (samples)"), self.hop_field)
        spectrogram_form.addRow(QLabel("Window Length (samples)"), self.window_length_field)
        spectrogram_form.addRow(QLabel("Windowing Function"), self.window_combobox)
        spectrogram_form.addRow(QLabel("Scale"), self.scale_combobox)
        spectrogram_form.addRow(QLabel("Mel Bands"), self.mel_field)
        spectrogram_form.addRow(QLabel("Colour Map"), self.colour_combobox)
        spectrogram_form.addRow(QLabel("Reverse Colours?"), self.reverse_checkbox)
        spectrogram_form.addRow(self.spectrogram_button)

        # TAB MENU
        tab_widget = TabWidget()
        tab_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        tab_widget.addTab(sound_import_frame, "Import")
        tab_widget.addTab(resample_frame, "Resample")
        tab_widget.addTab(spectrogram_frame, "Create Spectrogram")
        tab_widget.addTab(QWidget(), "Hide")

        # PLOT CONTAINER
        # TODO: see about leaving this convoluted method
        plotting_frame = self.findChild(QFrame, 'specFrame')
        plotting_frame.setLayout(QVBoxLayout())
        self.canvas = FigureCanvasQTAgg(plt.Figure(layout='constrained'))
        plotting_frame.layout().addWidget(self.canvas)
        self.audio_tool = AudioTool(self.canvas.figure)

        plot_layout = QVBoxLayout()
        plot_layout.setSpacing(0)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        spacer.setFixedHeight(self.font().pointSize() + 2)

        plot_layout.addWidget(spacer)
        plot_layout.addWidget(plotting_frame)

        # GRID LAYOUT
        grid_layout = self.findChild(QGridLayout, 'gridLayout')
        grid_layout.addLayout(plot_layout, 0, 0, 2, 3)
        grid_layout.addWidget(tab_widget, 0, 0, 1, 1)

        # SHOW WINDOW
        self.adjustSize()
        self.show()

    def reset(self):
        """ Reset all values and figures to a clear state. """
        # Wipe the graph and reset the information stored in the graphing aid
        self.canvas.figure.clear()
        self.audio_tool = AudioTool(self.canvas.figure)
        self.canvas.draw()

        # Configure all parameter fields to the default upon opening
        self.file_label.setText('No file selected')
        self.channel_label.setText('')
        self.native_rate_label.setText('No file selected')
        self.sample_rate_label.setText('')
        self.sample_rate_field.setText("22050")
        self.fft_field.setText("1024")
        self.hop_field.setText("256")
        self.window_length_field.setText("1024")
        self.mel_field.setText("80")
        self.window_combobox.setCurrentIndex(0)
        self.scale_combobox.setCurrentIndex(0)
        self.mel_field.setEnabled(True)
        self.colour_combobox.setCurrentIndex(0)
        self.reverse_checkbox.setChecked(False)
        self.resample_button.setEnabled(False)
        self.spectrogram_button.setEnabled(False)

    def import_file(self):
        """
        Prompt the user to select an audio file to use from the file explorer.
        Then, load that file as an array and draw the waveform.
        """
        file_name = QFileDialog.getOpenFileName(self, 'Open audio file',
                                                pathlib.Path().resolve().as_posix(),
                                                'Audio (*.wav *.mp3 *.flac)')[0]
        if file_name:
            # Clear any existing plots and draw a new waveform graph.
            self.reset()
            is_mono = self.audio_tool.loadFile(file_name)
            self.audio_tool.produceWaveform()
            self.canvas.draw()

            # Enable the graphing buttons to be toggled.
            self.resample_button.setEnabled(True)
            self.spectrogram_button.setEnabled(True)

            # Configure info labels to display text based on the provided sound.
            channel = "Mono-channel signal" if is_mono else \
                "Multi-channel signal - channels have been averaged"
            self.file_label.setText(f"{self.audio_tool.getFileName()} selected")
            self.channel_label.setText(f"{channel}")
            self.native_rate_label.setText(f"Original Sampling Rate: {self.audio_tool.getOriginalSampleRate()}Hz")
            self.sample_rate_label.setText(f"Current Sampling Rate: {self.audio_tool.getCurrentSampleRate()}Hz")

    def export_graph(self):
        """
        Prompt the user to enter a file name on the file explorer, then saves
        the current image as a PNG to that file location.
        """
        name = QFileDialog.getSaveFileName(self, 'Save file',
                                           pathlib.Path().resolve().as_posix(),
                                           'PNG File (*.png)')
        self.canvas.figure.savefig(name[0], format='png')

    def resample_wave(self):
        """
        Change the sampling rate of an existing waveform diagram.
        """
        # Outline the text box and cancel the operation if the text entry is not int
        is_valid, new_sr = retrieve_int_field(self.sample_rate_field, True)
        if not is_valid:
            return

        self.audio_tool.resample(new_sr)
        self.audio_tool.produceWaveform()
        self.sample_rate_label.setText(f"Current Sampling Rate: {new_sr}Hz")
        self.canvas.draw()

    def generate_spectrogram(self):
        """
        Create a spectrogram using the provided audio file and the parameters
        that the user has entered in the text fields.
        """
        # Check every text field and outline it if it is invalid
        is_valid, fft = retrieve_int_field(self.fft_field, True)
        is_valid, hop = retrieve_int_field(self.hop_field, is_valid)
        is_valid, win = retrieve_int_field(self.window_length_field, is_valid)
        is_valid, mel = retrieve_int_field(self.mel_field, is_valid)

        if not is_valid:
            return

        # Extract terms accepted as arguments from the combo boxes
        scale = self.scale_combobox.currentText().lower()
        # The colour map is reversible by appending _r to it
        cmap = self.colour_combobox.currentText().lower()
        cmap = cmap + '_r' if self.reverse_checkbox.isChecked() else cmap
        # The window function array is retrieved using scipy
        window = WINDOW_FUNCTIONS.get(self.window_combobox.currentText())(win)

        # Create the spectrogram and apply it to the canvas
        self.audio_tool.produceSpectrogram(n_fft=fft, hop_length=hop, n_mels=mel, window=window,
                                           scale=scale, cmap=cmap)
        self.canvas.draw()


app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
