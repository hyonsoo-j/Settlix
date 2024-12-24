import os
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtCore import pyqtSignal
from source.utils import enable_widgets
from source.config import Config
import traceback

ui_path = os.path.join(os.path.dirname(__file__), Config.UI_PATH, 'data_window.ui')
WindowTemplate, TemplateBaseClass = pg.Qt.loadUiType(ui_path)

class DataWindow(QMainWindow):
    data_generated = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__()
        self.ui = WindowTemplate()
        self.ui.setupUi(self)
        self.setWindowTitle("Detailed Data Settings")
        
        self.ui.apply_pushbutton.clicked.connect(self.save_data)
        self.ui.cancel_pushbutton.clicked.connect(self.close_window)

        self.parent = parent

        # Set the fixed size to prevent window resizing
        self.setFixedSize(self.size())

    def closeEvent(self, event):
        if self.parent:
            enable_widgets(self.parent.ui)
        super().closeEvent(event)

    def close_window(self):
        try:
            self.soil_data = {'softground': None, 'clay': None, 'silt': None, 'sand': None}
            self.data_generated.emit(self.soil_data)
            self.close()
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error: {e}")
            print("Traceback (most recent call last):")
            print(tb)

    def save_data(self):
        try:
            softground = self.ui.softground_lineedit.text()
            clay = self.ui.clay_lineedit.text()
            silt = self.ui.silt_lineedit.text()
            sand = self.ui.sand_lineedit.text()

            self.soil_data = {
                'softground': self.validate_input(softground),
                'clay': self.validate_input(clay),
                'silt': self.validate_input(silt),
                'sand': self.validate_input(sand)
            }

            if any(value is None for value in self.soil_data.values()):
                QMessageBox.warning(self, "Error", "Please fill in all fields with valid numbers.")
                return

            self.data_generated.emit(self.soil_data)
            self.close()
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error: {e}")
            print("Traceback (most recent call last):")
            print(tb)

    def validate_input(self, input_value):
        try:
            return float(input_value) if input_value else None
        except ValueError:
            return None
