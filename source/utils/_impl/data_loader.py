import pandas as pd
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from source.config import Config

def preprocess_data(data):
    
    data['Date'] = pd.to_datetime(data['Date'])
    
    full_date_range = pd.date_range(start=data['Date'].min(), end=data['Date'].max(), freq='D')
    data = data.set_index('Date').reindex(full_date_range).rename_axis('Date').reset_index()

    data.loc[:, 'fill_height'] = data['fill_height'].ffill()
    
    data['settlement'].interpolate(method='linear', inplace=True)

    fill_height_non_zero_index = data[data['fill_height'] != 0].index[0]
    data = data.loc[fill_height_non_zero_index-1:]
    
    data.at[data.index[0], 'settlement'] = 0.0
    
    data = data.dropna(how='all')
    
    return data

def load_csv_file(ui):
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(ui, "Select Prediction File", "", "CSV Files (*.csv);;All Files (*)", options=options)
    
    if fileName:
        try:
            data = pd.read_csv(fileName)
            if not all(column in data.columns for column in Config.DATA_COLUMNS):
                QMessageBox.critical(ui, "Data Error", f"The data file must contain the following columns: {Config.DATA_COLUMNS}.")
                return None, fileName
            
            data = preprocess_data(data)
            return data, fileName
        
        except Exception as e:
            QMessageBox.critical(ui, "Error", f"An error occurred while loading the file: {e}")
            return None, fileName
    
    return None, fileName
