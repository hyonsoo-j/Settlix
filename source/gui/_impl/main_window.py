import os
import sys
import pandas as pd
import pyqtgraph as pg
from pathlib import Path
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from .data_window import DataWindow
from .height_window import GraphWindow
from source.utils import save_to_png, save_to_csv, enable_widgets, disable_widgets, load_csv_file, link_x_axes, plot_graphs
from source.models import PredictionThread
from source.config import Config

pg.mkQApp()
pg.setConfigOption('background', 'w')

base_path = Path(__file__).parent.parent.parent.parent
ui_path = os.path.join(os.path.dirname(__file__), Config.UI_PATH, 'main_window.ui')
WindowTemplate, TemplateBaseClass = pg.Qt.loadUiType(ui_path)

classifications = {
    'Soft Ground Depth': 0,
    'Clay Ratio': 1,
}

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = WindowTemplate()
        self.ui.setupUi(self)
        self.setWindowTitle(f"{Config.APP_NAME} v.{Config.APP_VERSION}")
        self.initialize_ui() 
        self.initialize_variables()
        self.initialize_status()
        self.initialize_value()

        self.original_data = pd.DataFrame()
        # Date, fill_height, settlement, predicted_settlement

        disable_widgets(self.ui)
        self.ui.openfile_pushbutton.setEnabled(True)

        # List to hold proxies to prevent garbage collection
        self.ui.proxies = []

    def initialize_ui(self):
        # Connect buttons to their respective methods
        self.ui.openfile_pushbutton.clicked.connect(self.open_file)
        self.ui.datasetting_pushbutton.clicked.connect(self.set_soil_data)
        self.ui.height_pushbutton.clicked.connect(self.set_height)    
        self.ui.config_pushbutton.clicked.connect(self.set_config)    

        self.ui.prediction_pushbutton.clicked.connect(self.predict)

        self.ui.img_pushbutton.clicked.connect(self.save_figure)       
        self.ui.csv_pushbutton.clicked.connect(self.save_csv)  
        self.ui.help_pushbutton.clicked.connect(self.help)

        self.ui.actionOpen.triggered.connect(self.open_file)
        self.ui.actionExit.triggered.connect(self.close)

        self.ui.progressbar.setMaximum(100)
        self.ui.progressbar.setValue(0)

        link_x_axes(self.ui.height_graph.getViewBox(), self.ui.settlement_graph.getViewBox())

    def initialize_variables(self):
        self.mode = None
        self.present_date = None
        self.predict_date = None
        self.model_name = None
        self.classification = None
        self.model_update = False
        self.constrain_to_last = False

        self.data_config = {
            'softground': None,
            'clay': None,
            'silt': None,
            'sand': None
        }
        
        self.date = pd.DataFrame()
        self.before_height_data = pd.DataFrame()
        self.after_height_data = pd.DataFrame()
        self.before_settlement_data = pd.DataFrame()
        self.after_settlement_data = pd.DataFrame()
        self.predicted_settlement_data = pd.DataFrame()

    def initialize_status(self):
        self.ui.height_status_label.setText("Soft Ground Depth : (m)")
        self.ui.clay_status_label.setText("Clay : (m)")
        self.ui.silt_status_label.setText("Silt : (m)")
        self.ui.sand_status_label.setText("Sand : (m)")
        self.ui.present_date_label.setText("Observation Days :")
        self.ui.predict_date_label.setText("Prediction Days :")
        self.ui.model_status_label.setText("Prediction Model :")
        self.ui.class_status_label.setText("Class Method :")

    def initialize_value(self):
        self.ui.present_value_label.setText("Current Settlement : (cm)")
        self.ui.pred_value_label.setText("Final Pred Settlement : (cm)")
        self.ui.real_value_label.setText("Final Act Settlement : (cm)")
        self.ui.value_gap_label.setText("Final Settlement Gap : (cm)")
    
    def set_values(self):
        self.ui.present_value_label.setText(f"Current Settlement : {self.before_settlement_data['settlement'].iloc[-1]:.2f} (cm)")
        self.ui.pred_value_label.setText(f"Final Pred Settlement : {self.predicted_settlement_data['settlement'].iloc[-1]:.2f} (cm)")
        self.ui.real_value_label.setText(f"Final Act Settlement : {self.after_settlement_data['settlement'].iloc[-1]:.2f} (cm)")
        self.ui.value_gap_label.setText(f"Final Settlement Gap : {self.after_settlement_data['settlement'].iloc[-1] - self.predicted_settlement_data['settlement'].iloc[-1]:.2f} (cm)")

    def set_data(self):
        self.data ={
            'Date': self.date,
            'before_height': self.before_height_data,
            'after_height': self.after_height_data,
            'before_settlement': self.before_settlement_data,
            'after_settlement': self.after_settlement_data,
            'predicted_settlement': self.predicted_settlement_data
        }

    def draw_graph(self):
        self.set_data()
        plot_graphs(self.ui, self.data)

    def open_file(self):
        try:
            disable_widgets(self.ui)
            load_data, fileName = load_csv_file(self)

            if load_data is not None:
                self.initialize_status()
                self.initialize_value()
                self.initialize_variables()
                self.original_data = load_data.copy()

                self.date = load_data[['Date']].copy()
                self.before_height_data = load_data[['fill_height']].copy()
                self.before_settlement_data = load_data[['settlement']].copy()
                
                self.ui.filename_label.setText(fileName)

                self.present_date = len(self.original_data)

                self.draw_graph()
            
                self.ui.progressbar.setValue(0)
                self.ui.present_date_label.setText(f"Prediction Days : {self.present_date}")
                self.ui.present_lineedit.setText(str(self.present_date))
                
                self.ui.log_textbrowser.append(f"The file [{fileName}] has been successfully loaded.")
                enable_widgets(self.ui)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while opening the file: {e}")
            enable_widgets(self.ui)

    def init_soil_data(self, soil_data):
        enable_widgets(self.ui)

        if None in soil_data.values():
            return

        self.soil_data = soil_data
        self.data_config['softground'] = float(self.soil_data['softground'])
        self.data_config['clay'] = float(self.soil_data['clay'])
        self.data_config['silt'] = float(self.soil_data['silt'])
        self.data_config['sand'] = float(self.soil_data['sand'])

        self.ui.height_status_label.setText(f"Soft Ground Depth : {float(self.data_config['softground']):.2f} (m)")
        self.ui.clay_status_label.setText(f"Clay : {float(self.data_config['clay']):.2f} (m)")
        self.ui.silt_status_label.setText(f"Silt : {float(self.data_config['silt']):.2f} (m)")
        self.ui.sand_status_label.setText(f"Sand : {float(self.data_config['sand']):.2f} (m)")
        self.ui.log_textbrowser.append(f'\nThe soil data has been successfully configured.')

    def set_soil_data(self):
        self.datasetting_window = DataWindow(self)
        self.datasetting_window.data_generated.connect(self.init_soil_data)
        self.datasetting_window.show()
        disable_widgets(self.ui)

    def update_after_height_data(self, new_after_df):
        self.after_height_data = new_after_df
        self.draw_graph()

    def set_height(self):
        try:
            if self.original_data.empty:
                QMessageBox.critical(self, "Error", "No data is available.")
                return
            if self.present_date is None:
                QMessageBox.critical(self, "Error", "Observation days have not been set.")
                return
            if self.predict_date is None:
                QMessageBox.critical(self, "Error", "Prediction days have not been set.")
                return

                    
            self.graph_window = GraphWindow(self.before_height_data, self.after_height_data, self.present_date, self.predict_date, self)
            self.graph_window.data_generated.connect(self.update_after_height_data)
            self.graph_window.show()
            disable_widgets(self.ui)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while setting the height: {e}")

    def set_config(self):
        try:
            self.mode = self.ui.mode_combobox.currentText()
            self.model_name = self.ui.model_combobox.currentText()
            self.classification = self.ui.class_combobox.currentText()
            self.model_update = self.ui.update_checkbox.isChecked()
            self.constrain_to_last = self.ui.constrain_checkbox.isChecked()
            
            # present date
            if self.mode == 'Prediction':
                # set date
                self.present_date = len(self.original_data)
                self.ui.present_lineedit.setText(str(self.present_date))

                self.predict_date = int(self.ui.predict_lineedit.text())
                if self.predict_date <= 0:
                    raise ValueError("Prediction days must be a positive integer.")
                if self.predict_date < self.present_date:
                    raise ValueError("Prediction days cannot be less than the current days.")

                # set dataset
                last_fill_height = self.original_data['fill_height'].iloc[-1]
                self.before_height_data = self.original_data[['fill_height']].copy()
                self.after_height_data = pd.DataFrame(
                    {'fill_height': [last_fill_height] * (self.predict_date - self.present_date)},
                    index=pd.RangeIndex(start=self.present_date, stop=self.predict_date)
                )
                self.before_settlement_data = self.original_data[['settlement']].copy()
                self.after_settlement_data = pd.DataFrame()

                temp_date = self.original_data[['Date']].copy()
                last_date = temp_date['Date'].iloc[-1]
                additional_dates = pd.date_range(start=pd.to_datetime(last_date) + pd.Timedelta(days=1), 
                                                periods=self.predict_date - self.present_date)

                additional_dates_df = pd.DataFrame({'Date': additional_dates})

                temp_date = pd.concat([temp_date, additional_dates_df], ignore_index=True)

                self.date = temp_date.copy()

            if self.mode == 'Test':
                # set date
                self.predict_date = len(self.original_data)
                self.ui.predict_lineedit.setText(str(self.predict_date))

                self.present_date = int(self.ui.present_lineedit.text())
                if self.present_date < 0:
                    raise ValueError("Observation days must be a positive integer.")
                if self.present_date >= self.predict_date:
                    raise ValueError(f"Observation days must be less than the prediction days.\nCurrent data length: {len(self.predict_date)}")

                # set dataset
                self.date = self.original_data[['Date']].copy()
                self.before_height_data = self.original_data[['fill_height']].iloc[:self.present_date].copy()
                self.after_height_data = self.original_data[['fill_height']].iloc[self.present_date:].copy()
                self.before_settlement_data = self.original_data[['settlement']].iloc[:self.present_date].copy()
                self.after_settlement_data = self.original_data[['settlement']].iloc[self.present_date:].copy()
                self.predicted_settlement_data = pd.DataFrame()

            self.draw_graph()

            # log and status
            self.ui.present_date_label.setText(f"Observation Days : {self.present_date}")
            self.ui.predict_date_label.setText(f"Prediction Days : {self.predict_date}")
            self.ui.model_status_label.setText(f"Prediction Model : {self.model_name}")
            self.ui.class_status_label.setText(f"Class Method : {self.classification}")
            
            self.ui.log_textbrowser.append(f'\nMode set to {self.mode}.')
            self.ui.log_textbrowser.append(f'\nObservation days set to {self.present_date} days.')
            self.ui.log_textbrowser.append(f'\nPrediction days set to {self.predict_date} days.')
            self.ui.log_textbrowser.append(f'\n{self.model_name} model has been successfully configured.')

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while configuring the file: {type(e).__name__} - {e}")

    def predict(self):
        try:
            if self.original_data.empty:
                QMessageBox.critical(self, "Error", "No data is available.")
                return
            if self.soil_data is None or None in self.soil_data.values():
                QMessageBox.critical(self, "Error", "Soil data is missing.")
                return

            if (self.mode is None or 
                self.present_date is None or 
                self.predict_date is None or 
                self.model_name is None or 
                self.classification is None):
                QMessageBox.critical(self, "Error", "File setup is not complete.")
                return

            self.ui.progressbar.setValue(0)
            self.ui.log_textbrowser.append("\nStarting the prediction...")

            self.prediction_thread = PredictionThread(
                self.data,
                self.data_config,
                self.present_date,
                self.predict_date,
                self.model_name,
                classifications[self.classification],
                self.model_update,
                self.constrain_to_last,
                parent=self
            )

            self.prediction_thread.finished.connect(self.on_prediction_finished)
            self.prediction_thread.error.connect(self.on_prediction_error)

            self.prediction_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while configuring the prediction: {e}")

    def on_prediction_finished(self, result):
        try:
            self.predicted_settlement_data = result
            self.set_data()
            self.set_values()
            self.draw_graph()
            self.ui.log_textbrowser.append("\nPrediction has been completed.")
            self.ui.progressbar.setValue(100)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while processing the results: {e}")

    def on_prediction_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred during prediction: {error_message}")

    def save_figure(self):
        try:
            if self.original_data.empty:
                QMessageBox.critical(self, "Error", "There is no data to save.")
                return

            file_name = os.path.basename(self.ui.filename_label.text()).split('.')[0]
            selected_model = self.ui.model_combobox.currentText()
            predict_date = self.ui.predict_lineedit.text()

            output_file_name = save_to_png(self.data, base_path, file_name, selected_model, self.present_date, predict_date)
            self.ui.log_textbrowser.append(f"\nGraph has been saved as [{output_file_name}].")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while saving the graph: {e}")

    def save_csv(self):
        try:
            if self.original_data.empty:
                QMessageBox.critical(self, "Error", "There is no data to save.")
                return

            file_name = os.path.basename(self.ui.filename_label.text()).split('.')[0]
            selected_model = self.ui.model_combobox.currentText()
            predict_date = self.ui.predict_lineedit.text()

            output_file_name = save_to_csv(self.data, base_path, file_name, selected_model, predict_date)
            self.ui.log_textbrowser.append(f"\nCSV file has been saved as [{output_file_name}].")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred while saving the CSV file: {e}")

    def help(self):
        pass

    def close(self):
        reply = QMessageBox.question(self, 'Exit', 'Do you want to close the program?', 
                                    QMessageBox.No | QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            sys.exit()