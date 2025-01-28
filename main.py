import traceback

import librosa
import librosa.feature
import matplotlib.pyplot as plt
import numpy
import pathlib
import sys

from PyQt5.QtCore import Qt, QSize
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic

# Map the indices of the combo boxes to a tuple consisting of the string that
# is displayed to the user, and the name that will be used by Librosa.
WINDOW_FUNCTIONS = {0: ("Hanning", 'hann'),
                    1: ("Hamming", 'hamming'),
                    2: ("Blackman", 'blackman'),
                    3: ("Blackman-Harris", 'blackmanharris'),
                    4: ("Flat Top", 'flattop'),
                    5: ("Parzen", 'parzen'),
                    6: ("Triangular", 'triang'),
                    7: ("Rectangular", 'boxcar')}
SPEC_SCALES = {0: ("Mel", 'mel'),
               1: ("Linear", 'linear'),
               2: ("Logarithmic", 'log')}
# Colour maps chosen are ones that blend from one colour to a different one.
COLOUR_MAPS = ["Gray", "Magma", "Inferno", "Plasma", "Viridis", "Cividis"]

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
        self.specExists = False

    def doesSpectrogramExist(self):
        """ Retrieve whether a spectrogram has been created yet. """
        return self.specExists

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

        :param file_name: the path to access an audio file
        """
        self.fileName = file_name
        self.y, self.orig_sr = librosa.load(self.fileName, sr=None, mono=True)
        self.sr = self.orig_sr

    def produceWaveform(self):
        """ Graph the amplitude over time waveform of the loaded array. """
        self.ax[0].cla()
        librosa.display.waveshow(self.y, sr=self.sr, ax=self.ax[0])
        self.ax[0].set_title(f"Waveform of {self.getFileName()} at {self.sr}Hz")
        self.ax[0].label_outer()

    def produceSpectrogram(self, n_fft, hop_length, win_length, window, scale, n_mels, cmap):
        """
        Produce a spectrogram of the loaded array with a set of provided details.

        :param n_fft: the number of bins used to divide the window
        :param hop_length: the distance between each window
        :param win_length: the number of samples contained in each window
        :param window: the function used to zero the edges of the window
        :param scale: whether the spectrogram is mel, linear, or logarithmic
        :param n_mels: the number of bands in the mel filter bank if
        the spectrogram is mel; 0 otherwise
        :param cmap: the colour map used to display the spectrogram
        """
        # Clear all pre-existing information
        if self.colorbar is not None:
            self.colorbar.remove()
        self.ax[1].cla()

        if scale == "mel":
            s = librosa.feature.melspectrogram(
                y=self.y, sr=self.sr, n_fft=n_fft, win_length=win_length,
                hop_length=hop_length, window=window, n_mels=n_mels)
            s_db = librosa.power_to_db(numpy.abs(s), ref=numpy.max)
        else:
            s = librosa.stft(self.y, n_fft=n_fft, hop_length=hop_length,
                             win_length=win_length, window=window)
            s_db = librosa.power_to_db(numpy.abs(s), ref=numpy.max)

        self.img = librosa.display.specshow(s_db, sr=self.sr,
                                            n_fft=n_fft, win_length=win_length,
                                            hop_length=hop_length, cmap=cmap,
                                            ax=self.ax[1], x_axis='time', y_axis=scale)
        self.colorbar = self.figure.colorbar(self.img, ax=self.ax[1], format="%+2.f dB")

        self.ax[1].set_title(f'{scale} spectrogram with {win_length} window length')
        self.ax[1].label_outer()
        self.specExists = True

    def resample(self, target_sr):
        """ Change the sampling rate of the numpy array. """
        # If attempting to resample to a larger sampling rate, reload the
        # original recording to prevent loss of data.
        if target_sr > self.sr:
            self.loadFile(self.fileName)
        self.y = librosa.resample(self.y, orig_sr=self.sr, target_sr=target_sr)
        self.sr = target_sr

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
        self.soundImportForm = QFormLayout()
        self.soundImportFrame.setLayout(self.soundImportForm)

        self.audioChooseButton = QPushButton('Select Audio File')
        self.audioChooseButton.clicked.connect(self.importFile)
        self.audioLabel = QLabel('No file selected')
        self.audioLabel.setAlignment(Qt.AlignCenter)
        alert_label = QLabel('This will clear all existing graphs.')
        alert_label.setAlignment(Qt.AlignCenter)
        alert_label.setStyleSheet("color: red;")

        self.soundImportForm.addRow(self.audioLabel)
        self.soundImportForm.addRow(self.audioChooseButton)
        self.soundImportForm.addRow(alert_label)

        # WAVEFORM SETTINGS FORM
        self.waveFormFrame = QFrame(self.centralWidget)
        self.waveFormFrame.setAutoFillBackground(True)
        self.waveFormFrame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.waveForm = QFormLayout()
        self.waveFormFrame.setLayout(self.waveForm)

        self.rateLineEdit = QLineEdit("22050")
        self.rateLineEdit.setValidator(QDoubleValidator())
        self.rateLineEdit.textEdited.connect(
            lambda: self.rateLineEdit.setStyleSheet(QLineEdit().styleSheet()))

        self.resampleButton = QPushButton('Resample')
        self.resampleButton.clicked.connect(self.resampleWaveform)
        self.resampleButton.setEnabled(False)

        self.origRateLabel = QLabel('No file selected')
        self.currentRateLabel = QLabel('No file selected')
        alert_label = QLabel('This will not affect existing spectrograms.')
        alert_label.setAlignment(Qt.AlignCenter)
        alert_label.setStyleSheet("color: red;")
        alert_label2 = QLabel('Create a new one to see the effects of the resampling.')
        alert_label2.setAlignment(Qt.AlignCenter)
        alert_label2.setStyleSheet("color: red;")

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
        self.specForm = QFormLayout()
        self.specFormFrame.setLayout(self.specForm)

        # Prepare 4 text fields and add them to a list for quick error checking
        self.fftLineEdit = QLineEdit("2048")
        self.hopLineEdit = QLineEdit("512")
        self.lengthLineEdit = QLineEdit("2048")
        self.melLineEdit = QLineEdit("80")
        self.specLineEdits = [self.fftLineEdit, self.hopLineEdit, self.lengthLineEdit, self.melLineEdit]

        for lineEdit in self.specLineEdits:
            lineEdit.setValidator(QIntValidator())

        self.fftLineEdit.textEdited.connect(
            lambda: self.fftLineEdit.setStyleSheet(QLineEdit().styleSheet()))
        self.hopLineEdit.textEdited.connect(
            lambda: self.hopLineEdit.setStyleSheet(QLineEdit().styleSheet()))
        self.lengthLineEdit.textEdited.connect(
            lambda: self.lengthLineEdit.setStyleSheet(QLineEdit().styleSheet()))
        self.melLineEdit.textEdited.connect(
            lambda: self.melLineEdit.setStyleSheet(QLineEdit().styleSheet()))

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

        # GRID LAYOUT
        self.gridLayout = self.findChild(QGridLayout, 'gridLayout')
        self.specFrame = self.findChild(QFrame, 'specFrame')
        self.specFrameLayout = QVBoxLayout()
        self.gridLayout.addLayout(self.specFrameLayout, 0, 0, 2, 3)

        self.specFrame.setLayout(QVBoxLayout())
        self.canvas = FigureCanvasQTAgg(plt.Figure(layout='constrained'))
        self.specFrame.layout().addWidget(self.canvas)
        self.audioTool = AudioTool(self.canvas.figure)

        spaceWidget = QWidget()
        spaceWidget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.specFrameLayout.setSpacing(0)

        spaceWidget.setFixedHeight(self.font().pointSize() + 2)
        self.specFrameLayout.addWidget(spaceWidget)
        self.specFrameLayout.addWidget(self.specFrame)

        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        self.gridLayout.addWidget(QWidget(), 1, 0, 1, 1)
        self.gridLayout.addWidget(QWidget(), 0, 1, 1, 2)

        # SHOW WINDOW
        self.adjustSize()
        self.show()

    def reset(self):
        self.canvas.figure.clear()
        self.audioTool = AudioTool(self.canvas.figure)
        self.canvas.draw()

        # Configure all parameter fields to the default upon opening
        self.audioLabel.setText('No file selected')
        self.origRateLabel.setText('No file selected')
        self.currentRateLabel.setText('No file selected')
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
        Triggers upon clicking the 'Open' menu bar option or 'Import Audio' button.
        Prompts the user to select a audio file to use from the file explorer.
        Then, load that file into Librosa and draw the waveform.
        """
        file_name = QFileDialog.getOpenFileName(self, 'Open audio file',
                                                pathlib.Path().resolve().as_posix(),
                                                'Audio (*.wav *.mp3 *.flac)')[0]
        if file_name:
            self.reset()
            self.audioTool.loadFile(file_name)
            self.audioTool.produceWaveform()
            self.canvas.draw()

            self.resampleButton.setEnabled(True)
            self.specGenerateButton.setEnabled(True)

            self.audioLabel.setText(f"{self.audioTool.getFileName()} selected")
            self.origRateLabel.setText(f"Original Sampling Rate: {self.audioTool.getOriginalSampleRate()}Hz")
            self.currentRateLabel.setText(f"Current Sampling Rate: {self.audioTool.getCurrentSampleRate()}Hz")

    def exportPng(self):
        """
        Triggers upon clicking the 'Save as PNG' option on the menu bar.
        Prompts the user to enter a file name on the file explorer, then saves
        the current plot image to that file location.
        """
        name = QFileDialog.getSaveFileName(self, 'Save file',
                                           pathlib.Path().resolve().as_posix(),
                                           'PNG File (*.png)')
        self.canvas.figure.savefig(name[0], format='png')

    def resampleWaveform(self):
        try:
            self.audioTool.resample(int(self.rateLineEdit.text()))
        except ValueError:
            self.rateLineEdit.setStyleSheet("border: 1px solid red;")
            return

        self.audioTool.produceWaveform()
        self.currentRateLabel.setText(f"Current Sampling Rate: {self.audioTool.getCurrentSampleRate()}Hz")
        self.canvas.draw()

    def generateSpectrogram(self):
        """
        Triggers upon clicking the 'Generate Spectrogram' button.
        Combines the selected audio file with the parameter options
        that the user has selected.
        """
        is_valid = True
        for lineEdit in self.specLineEdits:
            is_valid = self.checkLineEdit(lineEdit) and is_valid

        if not is_valid:
            return

        # Extract the correct term from the combo boxes
        scale = SPEC_SCALES.get(self.scaleComboBox.currentIndex())[1]
        window = WINDOW_FUNCTIONS.get(self.windowingComboBox.currentIndex())[1]
        n_mels = int(self.melLineEdit.text()) if scale == 'mel' else 0
        cmap = self.colourComboBox.currentText().lower()
        cmap = cmap + '_r' if self.reverseCheckBox.isChecked() else cmap

        # Create the spectrogram
        self.audioTool.produceSpectrogram(n_fft=int(self.fftLineEdit.text()),
                                          hop_length=int(self.hopLineEdit.text()),
                                          win_length=int(self.lengthLineEdit.text()),
                                          window=window, scale=scale, n_mels=n_mels,
                                          cmap=cmap)
        self.canvas.draw()

    def checkLineEdit(self, line_edit):
        try:
            int(line_edit.text())
        except ValueError:
            if line_edit.isEnabled():
                line_edit.setStyleSheet("border: 1px solid red;")
                return False
        return True

app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
