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

class UI(QMainWindow):
    """
    Defines the main window and all UI interactions.
    """
    def __init__(self):
        super(UI, self).__init__()

        # Load the UI file
        uic.loadUi('spec-shower-ui.ui', self)

        # Import and prepare the frame
        self.specFrame = self.findChild(QFrame, 'specFrame')
        self.specFrame.setLayout(QVBoxLayout())
        self.canvas = FigureCanvasQTAgg(plt.Figure())
        self.specFrame.layout().addWidget(self.canvas)
        self.ax = None

        # Import relevant menu bar items
        self.fileName = ''
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
        self.fileName = ''
        self.canvas.figure.clear()
        self.canvas.draw()
        self.ax = None

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
        self.fileName = QFileDialog.getOpenFileName(self, 'Open audio file',
                                                     pathlib.Path().resolve().as_posix())[0]

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
        self.canvas.figure.clear()
        y, sr = librosa.load(self.fileName, sr=float (self.rateLineEdit.text()), mono=True)
        self.ax = self.canvas.figure.subplots()

        # Mel spectrograms are created differently in Librosa
        if self.scaleComboBox.currentText().lower() == "mel":
            s = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=int(self.dimensionLineEdit.text()),
                                               n_fft=int(self.fftLineEdit.text()),
                                               hop_length=int(self.hopLineEdit.text()),
                                               win_length=int(self.lengthLineEdit.text()),
                                               window=self.windowingComboBox.currentText().lower())
            s_db = librosa.power_to_db(numpy.abs(s), ref=numpy.max)
        else:
            s = librosa.stft(y,
                             n_fft=int (self.fftLineEdit.text()),
                             hop_length=int (self.hopLineEdit.text()),
                             win_length=int (self.lengthLineEdit.text()),
                             window=self.windowingComboBox.currentText().lower())
            s_db = librosa.amplitude_to_db(numpy.abs(s), ref=numpy.max)

        y_axis = self.scaleComboBox.currentText().lower()
        y_axis = "log" if y_axis == "logarithmic" else y_axis
        img = librosa.display.specshow(s_db, sr=sr, ax=self.ax, x_axis='time', y_axis=y_axis)

        self.canvas.figure.colorbar(img, ax=self.ax)

        self.canvas.draw()

app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
