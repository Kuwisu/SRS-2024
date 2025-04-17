from inspect import signature
import pathlib
import sys

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from scipy.signal.windows import *

from AudioTool import AudioTool
from TabWidget import TabWidget
from utils.ui import *


# Spectrogram y-axis scaling options
SPEC_SCALES = ["Mel", "Linear", "Logarithmic"]

# The text displayed in the window combo box mapped to a scipy window function.
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


class UI(QMainWindow):
    """
    Defines the main window and all UI interactions.
    """

    def __init__(self):
        super().__init__()

        # BASIC INITIALISATION & LOADING
        uic.loadUi('base-ui.ui', self)
        central_widget = self.findChild(QWidget, 'centralwidget')
        self.audio_tool = None

        # MENU BAR
        action_new = self.findChild(QAction, 'actionNew')
        action_open = self.findChild(QAction, 'actionOpen')
        action_save_png = self.findChild(QAction, 'actionSavePng')
        action_save_txt = self.findChild(QAction, 'actionSaveText')

        action_new.triggered.connect(self.reset)
        action_open.triggered.connect(self.import_file)
        action_save_png.triggered.connect(self.export_graph)
        action_save_txt.triggered.connect(self.export_parameters)

        # SOUND IMPORT TAB
        sound_import_frame = QFrame(central_widget)
        sound_import_frame.setAutoFillBackground(True)
        sound_import_frame.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        open_file_button = QPushButton('Select Audio File')
        open_file_button.clicked.connect(self.import_file)

        self.file_label = QLabel()
        self.file_label.setAlignment(Qt.AlignCenter)

        self.channel_label = QLabel()
        self.channel_label.setAlignment(Qt.AlignCenter)

        alert_label = QLabel('This will clear all existing graphs and fields.')
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

        self.sample_rate_field = QLineEdit()

        self.resample_button = QPushButton('Resample')
        self.resample_button.clicked.connect(self.resample_wave)
        self.resample_button.setEnabled(False)

        self.native_rate_label = QLabel()
        self.sample_rate_label = QLabel()

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

        self.fft_field = QLineEdit()
        self.hop_field = QLineEdit()
        self.window_length_field = QLineEdit()
        self.mel_field = QLineEdit()

        self.window_combobox = QComboBox()
        for window in WINDOW_FUNCTIONS:
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

        # CONFIGURE TEXT FIELD SETTINGS
        pos_int_validator = QIntValidator()
        pos_int_validator.setBottom(1)

        fields = [self.sample_rate_field, self.fft_field, self.hop_field,
                  self.window_length_field, self.mel_field]
        for field in fields:
            field.setValidator(pos_int_validator)
            field.textEdited.connect(lambda _, f=field: f.setStyleSheet(QLineEdit().styleSheet()))

        # TAB MENU
        tab_widget = TabWidget()
        tab_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        tab_widget.addTab(sound_import_frame, "Import")
        tab_widget.addTab(resample_frame, "Resample")
        tab_widget.addTab(spectrogram_frame, "Create Spectrogram")
        tab_widget.addTab(QWidget(), "Hide")

        # PLOT CONTAINER
        plotting_frame = self.findChild(QFrame, 'specFrame')
        plotting_frame.setLayout(QVBoxLayout())
        self.canvas = FigureCanvasQTAgg(plt.Figure(layout='constrained'))
        plotting_frame.layout().addWidget(self.canvas)

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
        self.reset()
        self.adjustSize()
        self.show()

    def reset(self):
        """ Reset all values and figures to a clear state. """
        # Wipe the graph and reset the information stored in the graphing aid
        self.canvas.figure.clear()
        self.audio_tool = AudioTool(self.canvas.figure)
        self.canvas.draw()

        # Configure all parameter fields to the default upon opening.
        self.file_label.setText('No file selected')
        self.channel_label.setText('')
        self.native_rate_label.setText('No file selected')
        self.sample_rate_label.setText('')

        # The default parameters are derived from the spectrogram creation function's defaults.
        self.sample_rate_field.setText(
            f"{signature(self.audio_tool.resample).parameters['target_sr'].default}"
        )
        self.fft_field.setText(
            f"{signature(self.audio_tool.produce_spectrogram).parameters['n_fft'].default}"
        )
        self.hop_field.setText(
            f"{signature(self.audio_tool.produce_spectrogram).parameters['hop_length'].default}"
        )
        self.window_length_field.setText(
            f"{len(signature(self.audio_tool.produce_spectrogram).parameters['window'].default)}"
        )
        self.mel_field.setText(
            f"{signature(self.audio_tool.produce_spectrogram).parameters['n_mels'].default}"
        )

        self.window_combobox.setCurrentIndex(0)
        self.scale_combobox.setCurrentIndex(0)
        self.colour_combobox.setCurrentIndex(0)
        self.mel_field.setEnabled(True)
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
            is_mono = self.audio_tool.load_file(file_name)
            self.audio_tool.produce_waveform()
            self.canvas.draw()

            # Enable the graphing buttons to be toggled.
            self.resample_button.setEnabled(True)
            self.spectrogram_button.setEnabled(True)

            # Configure info labels to display text based on the provided sound.
            channel = "Mono-channel signal" if is_mono else \
                "Multi-channel signal - channels have been averaged"
            self.file_label.setText(f"{self.audio_tool.get_file_name()} selected")
            self.channel_label.setText(f"{channel}")
            self.native_rate_label.setText(f"Original Sampling Rate: "
                                           f"{self.audio_tool.get_native_rate()}Hz")
            self.sample_rate_label.setText(f"Current Sampling Rate: "
                                           f"{self.audio_tool.get_sample_rate()}Hz")

    def export_graph(self):
        """
        Prompt the user to enter a file name on the file explorer, then saves
        the current image as a PNG to that file location.
        """
        file_name = QFileDialog.getSaveFileName(self, 'Save file',
                                           pathlib.Path().resolve().as_posix(),
                                           'PNG File (*.png)')[0]
        if file_name:
            self.canvas.figure.savefig(file_name, format='png')

    def export_parameters(self):
        """
        Prompt the user to enter a file name on the file explorer, then saves all added
        parameters in a TXT paragraph to that file location. If any key qualities are
        missing, display a message box alerting the user to them.
        """
        missing_info = QMessageBox()
        missing_info.setIcon(QMessageBox.Warning)
        missing_info.setWindowTitle("Error saving parameters")

        if self.audio_tool.get_file_name() == '':
            missing_info.setText('No file has been imported.')
            missing_info.exec_()
            return

        # Safely extract info from every text field
        is_valid, fft = retrieve_int_field(self.fft_field)
        is_valid, hop = retrieve_int_field(self.hop_field, is_valid)
        is_valid, win = retrieve_int_field(self.window_length_field, is_valid)
        is_valid, mels = retrieve_int_field(self.mel_field, is_valid)

        if not is_valid:
            missing_info.setText("Some text fields are not populated correctly.")
            missing_info.setInformativeText("These have been highlighted.")
            missing_info.exec_()
            return

        sample_rate = self.audio_tool.get_sample_rate()
        scale = self.scale_combobox.currentText().lower()
        window = self.window_combobox.currentText()

        article = get_article(mels) if scale == 'mel' else "a"
        dimension = f"{mels}-band " if scale == 'mel' else ""

        contents = (f"{article} {dimension}{scale} spectrogram at a {sample_rate}Hz sampling rate "
                    f"computed using an STFT with {fft} bins, a hop size of {hop}, "
                    f"and a {window} window of length {win}.")

        file_name = QFileDialog.getSaveFileName(self, 'Save file',
                                           pathlib.Path().resolve().as_posix(),
                                           'Text File (*.txt)')[0]
        if file_name:
            with open(file_name, 'w') as f:
                f.write(contents)

    def resample_wave(self):
        """
        Change the sampling rate of an existing waveform diagram.
        """
        # Outline the text box and cancel the operation if the text entry is not int
        is_valid, new_sr = retrieve_int_field(self.sample_rate_field)
        if not is_valid:
            return

        self.audio_tool.resample(new_sr)
        self.audio_tool.produce_waveform()
        self.sample_rate_label.setText(f"Current Sampling Rate: {new_sr}Hz")
        self.canvas.draw()

    def generate_spectrogram(self):
        """
        Create a spectrogram using the provided audio file and the parameters
        that the user has entered in the text fields.
        """
        # Check every text field and outline it if it is invalid
        is_valid, fft = retrieve_int_field(self.fft_field)
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
        self.audio_tool.produce_spectrogram(n_fft=fft, hop_length=hop, n_mels=mel, window=window,
                                            scale=scale, cmap=cmap)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    UIWindow = UI()
    app.exec_()
