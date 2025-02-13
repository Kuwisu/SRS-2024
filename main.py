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

# Map the indices of the combo boxes to a tuple consisting of the string that
# is displayed to the user, and the name that will be used as an argument.
WINDOW_FUNCTIONS = {0: ("Hanning", hann),
                    1: ("Hamming", hamming),
                    2: ("Blackman", blackman),
                    3: ("Blackman-Harris", blackmanharris),
                    4: ("Flat Top", flattop),
                    5: ("Parzen", parzen),
                    6: ("Triangular", triang),
                    7: ("Rectangular", boxcar)}
SPEC_SCALES = {0: ("Mel", 'mel'),
               1: ("Linear", 'linear'),
               2: ("Logarithmic", 'log')}
# Colour maps chosen are ones that blend from one colour to a different one.
COLOUR_MAPS = ["Gray", "Magma", "Inferno", "Plasma", "Viridis", "Cividis"]

def hz_to_mels(freq):
    return 2595 * np.log10(1 + freq/700)

def mels_to_hz(freq):
    return 700 * (10**(freq/2595) - 1)

def validateLineEditInt(line_edit):
    """
    Text entries are checked to make sure they are valid integers.
    If not, the text box is cleared and coloured red to signal an error.

    :param line_edit: the text field to validate
    :return: True if integer or disabled, False otherwise
    """
    try:
        int(line_edit.text())
    except ValueError:
        if line_edit.isEnabled():
            line_edit.setText("")
            line_edit.setStyleSheet("border: 1px solid red;")
            return False
    return True

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
        elif scale == 'log':
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
        # original recording to prevent loss of data.
        if target_sr > self.sr:
            self.loadFile(self.fileName)
        self.y = soxr.resample(self.y, self.sr, target_sr, 'HQ')
        self.sr = target_sr

    def construct_filterbank(self, n_fft, n_mels):
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

        return weights / (np.sum(weights, axis=1, keepdims=True)+1e-10)

class UI(QMainWindow):
    """
    Defines the main window and all UI interactions.
    """

    def __init__(self):
        super(UI, self).__init__()

        # FILE
        uic.loadUi('ui-25-01.ui', self)
        self.centralWidget = self.findChild(QWidget, 'centralwidget')

        # MENU BAR
        self.actionNew = self.findChild(QAction, 'actionNew')
        self.actionOpen = self.findChild(QAction, 'actionOpen')
        self.actionSavePng = self.findChild(QAction, 'actionSavePng')
        self.actionNew.triggered.connect(self.reset)
        self.actionOpen.triggered.connect(self.importFile)
        self.actionSavePng.triggered.connect(self.exportPng)

        # SOUND IMPORT TAB
        self.soundImportFrame = QFrame(self.centralWidget)
        self.soundImportFrame.setAutoFillBackground(True)
        self.soundImportFrame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.audioChooseButton = QPushButton('Select Audio File')
        self.audioChooseButton.clicked.connect(self.importFile)

        self.audioLabel = QLabel('No file selected')
        self.audioLabel.setAlignment(Qt.AlignCenter)

        self.channelLabel = QLabel('')
        self.channelLabel.setAlignment(Qt.AlignCenter)

        alert_label = QLabel('This will clear all existing graphs.')
        alert_label.setAlignment(Qt.AlignCenter)
        alert_label.setStyleSheet("color: red;")

        self.soundImportForm = QFormLayout()
        self.soundImportFrame.setLayout(self.soundImportForm)
        self.soundImportForm.addRow(self.audioLabel)
        self.soundImportForm.addRow(self.channelLabel)
        self.soundImportForm.addRow(self.audioChooseButton)
        self.soundImportForm.addRow(alert_label)

        # RESAMPLING TAB
        self.waveFormFrame = QFrame(self.centralWidget)
        self.waveFormFrame.setAutoFillBackground(True)
        self.waveFormFrame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.rateLineEdit = QLineEdit("22050")
        self.rateLineEdit.setValidator(QIntValidator())
        self.rateLineEdit.textEdited.connect(
            lambda: self.rateLineEdit.setStyleSheet(QLineEdit().styleSheet()))

        self.resampleButton = QPushButton('Resample')
        self.resampleButton.clicked.connect(self.resampleWaveform)
        self.resampleButton.setEnabled(False)

        self.origRateLabel = QLabel('No file selected')
        self.currentRateLabel = QLabel('')

        alert_label = QLabel('This will not affect an existing spectrogram.')
        alert_label.setAlignment(Qt.AlignCenter)
        alert_label.setStyleSheet("color: red;")

        alert_label2 = QLabel('Create a new one to see the effects of the resampling.')
        alert_label2.setAlignment(Qt.AlignCenter)
        alert_label2.setStyleSheet("color: red;")

        self.waveForm = QFormLayout()
        self.waveFormFrame.setLayout(self.waveForm)
        self.waveForm.addRow(self.origRateLabel)
        self.waveForm.addRow(self.currentRateLabel)
        self.waveForm.addRow(QLabel('New Sampling Rate (Hz)'), self.rateLineEdit)
        self.waveForm.addRow(self.resampleButton)
        self.waveForm.addRow(alert_label)
        self.waveForm.addRow(alert_label2)

        # SPECTROGRAM SETTINGS FORM
        self.specFormFrame = QFrame(self.centralWidget)
        self.specFormFrame.setAutoFillBackground(True)
        self.specFormFrame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.fftLineEdit = QLineEdit("2048")
        self.fftLineEdit.textEdited.connect(
            lambda: self.fftLineEdit.setStyleSheet(QLineEdit().styleSheet()))

        self.hopLineEdit = QLineEdit("512")
        self.hopLineEdit.textEdited.connect(
            lambda: self.hopLineEdit.setStyleSheet(QLineEdit().styleSheet()))

        self.lengthLineEdit = QLineEdit("2048")
        self.lengthLineEdit.textEdited.connect(
            lambda: self.lengthLineEdit.setStyleSheet(QLineEdit().styleSheet()))

        self.melLineEdit = QLineEdit("80")
        self.melLineEdit.textEdited.connect(
            lambda: self.melLineEdit.setStyleSheet(QLineEdit().styleSheet()))

        # Add the fields to a list to iterate through them for error checking
        self.specLineEdits = [self.fftLineEdit, self.hopLineEdit, self.lengthLineEdit, self.melLineEdit]
        for lineEdit in self.specLineEdits:
            lineEdit.setValidator(QIntValidator())

        self.windowingComboBox = QComboBox()
        for index, window in WINDOW_FUNCTIONS.items():
            self.windowingComboBox.addItem(window[0])

        self.scaleComboBox = QComboBox()
        for index, scale in SPEC_SCALES.items():
            self.scaleComboBox.addItem(scale[0])
        self.scaleComboBox.activated[str].connect(
            lambda text: self.melLineEdit.setEnabled(text == "Mel"))

        self.colourComboBox = QComboBox()
        for colour in COLOUR_MAPS:
            self.colourComboBox.addItem(colour)
        self.reverseCheckBox = QCheckBox()

        self.specGenerateButton = QPushButton('Generate Spectrogram')
        self.specGenerateButton.clicked.connect(self.generateSpectrogram)
        self.specGenerateButton.setEnabled(False)

        self.specForm = QFormLayout()
        self.specFormFrame.setLayout(self.specForm)
        self.specForm.addRow(QLabel("FFT Size (samples)"), self.fftLineEdit)
        self.specForm.addRow(QLabel("Hop Size (samples)"), self.hopLineEdit)
        self.specForm.addRow(QLabel("Window Length (samples)"), self.lengthLineEdit)
        self.specForm.addRow(QLabel("Windowing Function"), self.windowingComboBox)
        self.specForm.addRow(QLabel("Scale"), self.scaleComboBox)
        self.specForm.addRow(QLabel("Mel Bands"), self.melLineEdit)
        self.specForm.addRow(QLabel("Colour Map"), self.colourComboBox)
        self.specForm.addRow(QLabel("Reverse Colours?"), self.reverseCheckBox)
        self.specForm.addRow(self.specGenerateButton)

        # TAB MENU
        self.tabWidget = TabWidget()
        self.tabWidget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.tabWidget.addTab(self.soundImportFrame, "Import")
        self.tabWidget.addTab(self.waveFormFrame, "Resample")
        self.tabWidget.addTab(self.specFormFrame, "Create Spectrogram")
        self.tabWidget.addTab(QWidget(), "Hide")

        # SPECTROGRAM CONTAINER
        self.specFrame = self.findChild(QFrame, 'specFrame')
        # TODO: clean up naming here - distinction between specFrameLayout and specFrame.layout()
        self.specFrame.setLayout(QVBoxLayout())
        self.canvas = FigureCanvasQTAgg(plt.Figure(layout='constrained'))
        self.specFrame.layout().addWidget(self.canvas)
        self.audioTool = AudioTool(self.canvas.figure)

        self.specFrameLayout = QVBoxLayout()
        self.specFrameLayout.setSpacing(0)

        spaceWidget = QWidget()
        spaceWidget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        spaceWidget.setFixedHeight(self.font().pointSize() + 2)

        self.specFrameLayout.addWidget(spaceWidget)
        self.specFrameLayout.addWidget(self.specFrame)

        # GRID LAYOUT
        self.gridLayout = self.findChild(QGridLayout, 'gridLayout')
        self.gridLayout.addLayout(self.specFrameLayout, 0, 0, 2, 3)
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)

        # SHOW WINDOW
        self.adjustSize()
        self.show()

    def reset(self):
        """ Reset all values and figures to a clear state. """
        # Wipe the graph and reset the information stored in the graphing aid
        self.canvas.figure.clear()
        self.audioTool = AudioTool(self.canvas.figure)
        self.canvas.draw()

        # Configure all parameter fields to the default upon opening
        self.audioLabel.setText('No file selected')
        self.channelLabel.setText('')
        self.origRateLabel.setText('No file selected')
        self.currentRateLabel.setText('')
        self.rateLineEdit.setText("22050")
        self.fftLineEdit.setText("2048")
        self.hopLineEdit.setText("512")
        self.lengthLineEdit.setText("2048")
        self.melLineEdit.setText("80")
        self.windowingComboBox.setCurrentIndex(0)
        self.scaleComboBox.setCurrentIndex(0)
        self.melLineEdit.setEnabled(True)
        self.colourComboBox.setCurrentIndex(0)
        self.reverseCheckBox.setChecked(False)
        self.resampleButton.setEnabled(False)
        self.specGenerateButton.setEnabled(False)

    def importFile(self):
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
            is_mono = self.audioTool.loadFile(file_name)
            self.audioTool.produceWaveform()
            self.canvas.draw()

            # Enable the graphing buttons to be toggled.
            self.resampleButton.setEnabled(True)
            self.specGenerateButton.setEnabled(True)

            # Configure info labels to display text based on the provided sound.
            self.audioLabel.setText(f"{self.audioTool.getFileName()} selected")
            self.channelLabel.setText(f"{"Mono-channel signal" if is_mono 
            else "Multi-channel signal - channels have been averaged"}")
            self.origRateLabel.setText(f"Original Sampling Rate: {self.audioTool.getOriginalSampleRate()}Hz")
            self.currentRateLabel.setText(f"Current Sampling Rate: {self.audioTool.getCurrentSampleRate()}Hz")

    def exportPng(self):
        """
        Prompt the user to enter a file name on the file explorer, then saves
        the current image as a PNG to that file location.
        """
        name = QFileDialog.getSaveFileName(self, 'Save file',
                                           pathlib.Path().resolve().as_posix(),
                                           'PNG File (*.png)')
        self.canvas.figure.savefig(name[0], format='png')

    def resampleWaveform(self):
        """
        Change the sampling rate of an existing waveform diagram.
        """
        # Outline it and cancel the operation if the text entry is not int
        if not validateLineEditInt(self.rateLineEdit):
            return

        self.audioTool.resample(int(self.rateLineEdit.text()))
        self.audioTool.produceWaveform()
        self.currentRateLabel.setText(f"Current Sampling Rate: {self.audioTool.getCurrentSampleRate()}Hz")
        self.canvas.draw()

    def generateSpectrogram(self):
        """
        Create a spectrogram using the provided audio file and the parameters
        that the user has entered in the text fields.
        """
        # Iterate through every field and outline it if it is invalid
        is_valid = True
        for lineEdit in self.specLineEdits:
            is_valid = validateLineEditInt(lineEdit) and is_valid

        if not is_valid:
            return

        # Extract terms accepted as arguments from the combo boxes
        scale = SPEC_SCALES.get(self.scaleComboBox.currentIndex())[1]
        # The colour map is reversible by appending _r to it
        cmap = self.colourComboBox.currentText().lower()
        cmap = cmap + '_r' if self.reverseCheckBox.isChecked() else cmap
        # The window function array is retrieved using scipy
        window_length = int(self.lengthLineEdit.text())
        window = WINDOW_FUNCTIONS.get(self.windowingComboBox.currentIndex())[1](window_length)

        # Create the spectrogram and apply it to the canvas
        self.audioTool.produceSpectrogram(n_fft=int(self.fftLineEdit.text()),
                                          hop_length=int(self.hopLineEdit.text()),
                                          n_mels=int(self.melLineEdit.text()),
                                          window=window, scale=scale, cmap=cmap)
        self.canvas.draw()


app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
