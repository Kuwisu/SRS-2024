import traceback

import librosa
import librosa.feature
import matplotlib.pyplot as plt
import numpy
import pathlib
import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic

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

        img = librosa.display.specshow(s_db, sr=self.sr, ax=self.ax[1], x_axis='time', y_axis=scale)
        if self.colorbar is None:
            self.colorbar = self.figure.colorbar(img, ax=self.ax[1], format="%+2.f dB")

        # times = librosa.frames_to_time(numpy.arange(s_db.shape[1]), sr=self.sr, hop_length=hop_length)
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

        # Load the UI file
        uic.loadUi('spec-shower-ui.ui', self)

        # Import and prepare the spectrogram display frame
        self.specFrame = self.findChild(QFrame, 'specFrame')
        self.specFrame.setLayout(QVBoxLayout())
        self.canvas = FigureCanvasQTAgg(plt.Figure(layout='constrained'))
        self.specFrame.layout().addWidget(self.canvas)

        # Prepare the tool that will interact with the display canvas
        self.audioTool = AudioTool(self.canvas.figure)

        # Import relevant menu bar items
        self.actionNew = self.findChild(QAction, 'actionNew')
        self.actionOpen = self.findChild(QAction, 'actionOpen')
        self.actionSavePng = self.findChild(QAction, 'actionSavePng')
        self.actionNew.triggered.connect(self.reset)
        self.actionOpen.triggered.connect(self.importFile)
        self.actionSavePng.triggered.connect(self.exportPng)

        # Import the parameter dropdown and allow it to toggle the frame on/off
        self.paramButton = self.findChild(QPushButton, 'paramButton')
        self.paramEditFrame = self.findChild(QFrame, 'paramEditFrame')
        self.paramButton.clicked.connect(
            lambda: self.paramEditFrame.setVisible(
                not self.paramEditFrame.isVisible()))

        # Import the parameter fields
        self.rateLineEdit = self.findChild(QLineEdit, 'rateLineEdit')
        self.fftLineEdit = self.findChild(QLineEdit, 'fftLineEdit')
        self.hopLineEdit = self.findChild(QLineEdit, 'hopLineEdit')
        self.lengthLineEdit = self.findChild(QLineEdit, 'lengthLineEdit')
        self.windowingComboBox = self.findChild(QComboBox, 'windowingComboBox')
        # The mel bands field is only enabled when creating a mel spectrogram
        self.scaleComboBox = self.findChild(QComboBox, 'scaleComboBox')
        self.dimensionLineEdit = self.findChild(QLineEdit, 'dimensionLineEdit')
        self.scaleComboBox.activated[str].connect(
            lambda text: self.dimensionLineEdit.setEnabled(text == "Mel"))

        # Configure the text fields to only allow numbers
        self.rateLineEdit.setValidator(QDoubleValidator())
        self.fftLineEdit.setValidator(QIntValidator())
        self.hopLineEdit.setValidator(QIntValidator())
        self.lengthLineEdit.setValidator(QIntValidator())
        self.dimensionLineEdit.setValidator(QIntValidator())

        # Import the button that will create a spectrogram when clicked
        self.createButton = self.findChild(QPushButton, 'createButton')
        self.failLabel = self.findChild(QLabel, 'failLabel')
        self.createButton.clicked.connect(self.generate)

        # Show the application window
        self.show()

    def reset(self):
        # Clear any saved sound file and generated plot
        self.canvas.figure.clear()
        self.canvas.draw()
        self.audioTool = AudioTool(self.canvas.figure)

        # Configure all parameter fields to the default upon opening
        self.rateLineEdit.setText('')
        self.fftLineEdit.setText('')
        self.hopLineEdit.setText('')
        self.lengthLineEdit.setText('')
        self.dimensionLineEdit.setText('')
        self.windowingComboBox.setCurrentIndex(0)
        self.scaleComboBox.setCurrentIndex(0)
        self.dimensionLineEdit.setEnabled(True)

    def importFile(self):
        """
        Triggers upon clicking the 'Open' option on the menu bar.
        Prompts the user to select a audio file to use from the file explorer.
        """
        file_name = QFileDialog.getOpenFileName(self, 'Open audio file',
                                                pathlib.Path().resolve().as_posix())[0]
        self.audioTool.loadFile(file_name)
        self.audioTool.produceWaveform()
        self.canvas.draw()

    def exportPng(self):
        """
        Triggers upon clicking the 'Save as PNG' option on the menu bar.
        Prompts the user to enter a file name on the file explorer, then saves
        the current plot image to that file location.
        """
        name = QFileDialog.getSaveFileName(self, 'Save file')
        self.canvas.figure.savefig(name[0])

    def generate(self):
        """
        Triggers upon clicking the 'Generate Spectrogram' button.
        Combines the selected audio file with the parameter options
        that the user has selected.
        """
        scale = self.scaleComboBox.currentText().lower()
        scale = "log" if scale == "logarithmic" else scale
        n_mels = int(self.dimensionLineEdit.text()) if scale == "mel" else 0

        self.audioTool.resample(int(self.rateLineEdit.text()))
        self.audioTool.produceWaveform()
        self.audioTool.produceSpectrogram(n_fft=int(self.fftLineEdit.text()),
                                          hop_length=int(self.hopLineEdit.text()),
                                          win_length=int(self.lengthLineEdit.text()),
                                          window=self.windowingComboBox.currentText().lower(),
                                          scale=scale, n_mels=n_mels)
        self.canvas.draw()

app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
