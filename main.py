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
# Colour maps chosen are ones that blend from one colour to another.
COLOUR_MAPS = ["Magma", "Inferno", "Plasma", "Viridis", "Cividis", "Gray"]

class TabWidget(QTabWidget):
    """
    Override the sizing method that the tab widget uses.

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
    def __init__(self, figure):
        self.figure = figure
        self.fileName = ''
        self.y = None
        self.orig_sr = None
        self.sr = None
        self.colorbar = None
        self.ax = self.figure.subplots(nrows=2, sharex=True)

    def getFileName(self):
        return self.fileName.split('/')[-1]

    def getOriginalSampleRate(self):
        return self.orig_sr

    def getCurrentSampleRate(self):
        return self.sr

    def loadFile(self, file_name):
        self.fileName = file_name
        self.y, self.orig_sr = librosa.load(self.fileName, mono=True)
        self.sr = self.orig_sr

    def produceWaveform(self):
        self.ax[0].cla()
        librosa.display.waveshow(self.y, sr=self.sr, ax=self.ax[0])
        self.ax[0].set_title(f"Waveform of {self.getFileName()} at {self.sr}Hz")
        self.ax[0].label_outer()

    def produceSpectrogram(self, n_fft, hop_length, win_length, window, scale, n_mels, cmap):
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

        self.img = librosa.display.specshow(s_db, sr=self.sr, n_fft=n_fft,
                                            win_length=win_length, hop_length=hop_length,
                                            ax=self.ax[1], x_axis='time', y_axis=scale,
                                            cmap=cmap)
        if self.colorbar is None:
            self.colorbar = self.figure.colorbar(self.img, ax=self.ax[1], format="%+2.f dB")

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

        self.soundImportForm.addRow(self.audioChooseButton)
        self.soundImportForm.addRow(self.audioLabel)

        # WAVEFORM SETTINGS FORM
        self.waveFormFrame = QFrame(self.centralWidget)
        self.waveFormFrame.setAutoFillBackground(True)
        self.waveFormFrame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.waveForm = QFormLayout()
        self.waveFormFrame.setLayout(self.waveForm)

        self.rateLineEdit = QLineEdit()
        self.rateLineEdit.setValidator(QDoubleValidator())

        self.resampleButton = QPushButton('Resample')
        self.resampleButton.clicked.connect(self.resampleWaveform)

        self.origRateLabel = QLabel('')
        self.currentRateLabel = QLabel('')
        alert_label = QLabel('This will affect existing spectrograms')
        alert_label.setAlignment(Qt.AlignCenter)

        self.waveForm.addRow(self.origRateLabel)
        self.waveForm.addRow(self.currentRateLabel)
        self.waveForm.addRow(QLabel('New Sampling Rate (Hz)'), self.rateLineEdit)
        self.waveForm.addRow(self.resampleButton)
        self.waveForm.addRow(alert_label)

        # SPECTROGRAM SETTINGS FORM
        self.specFormFrame = QFrame(self.centralWidget)
        self.specFormFrame.setAutoFillBackground(True)
        self.specFormFrame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.specForm = QFormLayout()
        self.specFormFrame.setLayout(self.specForm)

        self.fftLineEdit = QLineEdit()
        self.hopLineEdit = QLineEdit()
        self.lengthLineEdit = QLineEdit()
        self.melLineEdit = QLineEdit()
        self.fftLineEdit.setValidator(QIntValidator())
        self.hopLineEdit.setValidator(QIntValidator())
        self.lengthLineEdit.setValidator(QIntValidator())
        self.melLineEdit.setValidator(QIntValidator())

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
            # self.reset()
            self.audioTool.loadFile(file_name)
            self.audioTool.produceWaveform()
            self.canvas.draw()

            self.audioLabel.setText(f"{self.audioTool.getFileName()} selected")
            self.origRateLabel.setText(f"Original Sampling Rate: {self.audioTool.getOriginalSampleRate()}Hz")
            self.currentRateLabel.setText(f"Current Sampling Rate: {self.audioTool.getCurrentSampleRate()}Hz")

            self.tabWidget.setCurrentIndex(1)
            self.rateLineEdit.setText(f"{self.audioTool.getCurrentSampleRate()}")

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
        self.audioTool.resample(int(self.rateLineEdit.text()))
        self.audioTool.produceWaveform()
        self.currentRateLabel.setText(f"Current Sampling Rate: {self.audioTool.getCurrentSampleRate()}Hz")
        self.generateSpectrogram()
        self.canvas.draw()

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
        cmap = self.colourComboBox.currentText().lower()
        cmap = cmap + '_r' if self.reverseCheckBox.isChecked() else cmap

        # Create the spectrogram
        self.audioTool.produceSpectrogram(n_fft=int(self.fftLineEdit.text()),
                                          hop_length=int(self.hopLineEdit.text()),
                                          win_length=int(self.lengthLineEdit.text()),
                                          window=window, scale=scale, n_mels=n_mels,
                                          cmap=cmap)
        self.canvas.draw()

    def recolorSpectrogram(self, colormap):
        self.audioTool.recolor(colormap.lower())

app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()
