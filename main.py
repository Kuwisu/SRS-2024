import traceback

import librosa
import librosa.feature
import matplotlib.pyplot as plt
import numpy
import pathlib
import sys

from PyQt5.QtCore import Qt
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

class AudioTool:
    def __init__(self, figure):
        self.figure = figure
        self.fileName = ''
        self.y = None
        self.sr = None
        self.colorbar = None
        self.ax = self.figure.subplots(nrows=2, sharex=True)

    def loadFile(self, file_name):
        self.fileName = file_name
        self.y, self.sr = librosa.load(self.fileName, mono=True)

    def produceWaveform(self):
        self.ax[0].cla()
        librosa.display.waveshow(self.y, sr=self.sr, ax=self.ax[0])
        self.ax[0].set_title(f"Waveform of {self.fileName.split('/')[-1]} at {self.sr}Hz")
        self.ax[0].label_outer()

    def produceSpectrogram(self, n_fft, hop_length, win_length, window, scale, n_mels):
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

        img = librosa.display.specshow(s_db, sr=self.sr, n_fft=n_fft,
                                       win_length=win_length, hop_length=hop_length,
                                       ax=self.ax[1], x_axis='time', y_axis=scale)
        if self.colorbar is None:
            self.colorbar = self.figure.colorbar(img, ax=self.ax[1], format="%+2.f dB")

        self.ax[1].set_title(f'{scale} spectrogram with {win_length} window length')
        self.ax[1].label_outer()

    def resample(self, target_sr):
        # If attempting to resample to a larger sample rate, try to revert to
        # the original recording.
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

        # Load the UI file and load the center widget
        uic.loadUi('ui-25-01.ui', self)
        self.centralWidget = self.findChild(QWidget, 'centralWidget')

        # Load all the menu bar items
        self.actionNew = self.findChild(QAction, 'actionNew')
        self.actionOpen = self.findChild(QAction, 'actionOpen')
        self.actionSavePng = self.findChild(QAction, 'actionSavePng')
        self.actionNew.triggered.connect(self.reset)
        self.actionOpen.triggered.connect(self.importFile)
        self.actionSavePng.triggered.connect(self.exportPng)

        # Load the grid and change the frame to occupy 2*3 cells rather than 1*1
        self.gridLayout = self.findChild(QGridLayout, 'gridLayout')
        self.specFrame = self.findChild(QFrame, 'specFrame')
        self.gridLayout.addWidget(self.specFrame, 0, 0, 2, 3)

        # Prepare the display canvas and tool that interacts with it
        self.specFrame.setLayout(QVBoxLayout())
        self.canvas = FigureCanvasQTAgg(plt.Figure(layout='constrained'))
        self.specFrame.layout().addWidget(self.canvas)
        self.audioTool = AudioTool(self.canvas.figure)

        # Create the menu for spectrogram parameters
        self.specFormFrame = QFrame(self.centralWidget)
        self.specFormFrame.setAutoFillBackground(True)
        self.specFormFrame.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self.specForm = QFormLayout()
        self.specFormFrame.setLayout(self.specForm)
        self.gridLayout.addWidget(self.specFormFrame, 0, 1, 1, 1, alignment=Qt.AlignJustify)
        self.gridLayout.addWidget(QWidget().setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding), 1, 0, 1, 1)

        # # Create the menu for waveform parameters
        # self.waveFormFrame = QFrame(self.centralWidget)
        # self.waveFormFrame.setAutoFillBackground(True)
        # self.waveFormFrame.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        # self.waveForm = QFormLayout()
        # self.waveFormFrame.setLayout(self.waveForm)
        # self.gridLayout.addWidget(self.waveFormFrame, 0, 0, 1, 1, alignment=Qt.AlignTop)

        # Create the text entries for parameters
        self.fftLineEdit = QLineEdit()
        self.hopLineEdit = QLineEdit()
        self.lengthLineEdit = QLineEdit()
        self.melLineEdit = QLineEdit()

        # Fill out the window function combobox
        self.windowingComboBox = QComboBox()
        for index, window in WINDOW_FUNCTIONS.items():
            self.windowingComboBox.addItem(window[0])
        self.windowingComboBox.setCurrentIndex(0)

        # Fill out the scale combobox and bind the mel bands field to only
        # be writeable when it is a Mel spectrogram.
        self.scaleComboBox = QComboBox()
        for index, scale in SPEC_SCALES.items():
            self.scaleComboBox.addItem(scale[0])
        self.scaleComboBox.setCurrentIndex(0)
        self.scaleComboBox.activated[str].connect(
            lambda text: self.melLineEdit.setEnabled(text == "Mel"))

        # Configure the text fields to only allow numbers
        self.fftLineEdit.setValidator(QIntValidator())
        self.hopLineEdit.setValidator(QIntValidator())
        self.lengthLineEdit.setValidator(QIntValidator())
        self.melLineEdit.setValidator(QIntValidator())

        # Create the button that will create a spectrogram when clicked
        self.specGenerateButton = QPushButton('Generate Spectrogram')
        self.specGenerateButton.clicked.connect(self.generateSpectrogram)

        # Create the rows to go into the spectrogram form
        self.specForm.addRow(QLabel("FFT Size (samples)"), self.fftLineEdit)
        self.specForm.addRow(QLabel("Hop Size (samples)"), self.hopLineEdit)
        self.specForm.addRow(QLabel("Window Length (samples)"), self.lengthLineEdit)
        self.specForm.addRow(QLabel("Windowing Function"), self.windowingComboBox)
        self.specForm.addRow(QLabel("Scale"), self.scaleComboBox)
        self.specForm.addRow(QLabel("Mel Bands"), self.melLineEdit)
        self.specForm.addRow(self.specGenerateButton)

        # Load the parameter entry button and allow it to toggle the form
        self.specButton = self.findChild(QPushButton, 'specButton')
        self.specButton.clicked.connect(
            lambda: self.specFormFrame.setVisible(not self.specFormFrame.isVisible()))

        # Show the application window
        self.show()

    def reset(self):
        self.canvas.figure.clear()
        self.canvas.draw()
        self.audioTool = AudioTool(self.canvas.figure)

        # Configure all parameter fields to the default upon opening
        self.fftLineEdit.setText('')
        self.hopLineEdit.setText('')
        self.lengthLineEdit.setText('')
        self.melLineEdit.setText('')
        self.windowingComboBox.setCurrentIndex(0)
        self.scaleComboBox.setCurrentIndex(0)
        self.melLineEdit.setEnabled(True)

    def importFile(self):
        """
        Triggers upon clicking the 'Open' option on the menu bar.
        Prompts the user to select a audio file to use from the file explorer.
        Then, load that file into Librosa and draw the waveform.
        """
        file_name = QFileDialog.getOpenFileName(self, 'Open audio file',
                                                pathlib.Path().resolve().as_posix(),
                                                'Audio (*.wav *.mp3 *.flac)')[0]
        if file_name:
            self.audioTool.loadFile(file_name)
            self.audioTool.produceWaveform()
            self.canvas.draw()

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

    def generateSpectrogram(self):
        """
        Triggers upon clicking the 'Generate Spectrogram' button.
        Combines the selected audio file with the parameter options
        that the user has selected.
        """
        # Extract the correct term from the combo boxes
        scale = SPEC_SCALES.get(self.scaleComboBox.currentIndex())[1]
        window = WINDOW_FUNCTIONS.get(self.windowingComboBox.currentIndex())[1]
        n_mels = int(self.melLineEdit.text()) if scale == 'mel' else 0

        # Create the spectrogram
        self.audioTool.produceSpectrogram(n_fft=int(self.fftLineEdit.text()),
                                          hop_length=int(self.hopLineEdit.text()),
                                          win_length=int(self.lengthLineEdit.text()),
                                          window=window, scale=scale, n_mels=n_mels)
        self.canvas.draw()

app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
