import os
from . import _RNN, _LSTM, _GRU
from . import hyperparameters
from .data_utils import reconstruct_dataframe
from .model_utils import update_model

from pathlib import Path
import neuralprophet
import random
import json
import torch
import numpy as np
import pandas as pd
from neuralprophet import load as load_np
from joblib import load as load_scaler
from PyQt5.QtCore import QThread, pyqtSignal

from source.config import Config

class PredictionThread(QThread):
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, dict_data, data_config, observed_days, predicted_days, model_name, classification, update, constrain_to_last, parent=None):
        super(PredictionThread, self).__init__(parent)
        self.dict_data = dict_data
        self.data_config = data_config
        self.observed_days = observed_days
        self.predicted_days = predicted_days
        self.model_name = model_name
        self.classification = classification
        self.update = update
        self.constrain_to_last = constrain_to_last

    def run(self):
        try:
            # predict_time_series 실행
            result = predict_time_series(
                self.dict_data,
                self.data_config,
                self.observed_days,
                self.predicted_days,
                self.model_name,
                self.classification,
                self.update,
                self.constrain_to_last
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
            
def set_seed():
    seed = Config.RANDOM_SEED
    np.random.seed(seed)
    random.seed(seed)
    neuralprophet.set_random_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

def get_class_name(data, data_config, classification):
    if classification == 0:
        if data_config['softground'] <4:
            class_name = "m0"
        elif data_config['softground'] <10:
            class_name = "m1"
        elif data_config['softground'] <14:
            class_name = "m2"
        else:
            class_name = "m3"
            
    elif classification == 1:
        if data_config['clay'] * data['fill_height'].iloc[-1] <16:
            class_name = "m0"
        elif data_config['clay'] * data['fill_height'].iloc[-1] <32:
            class_name = "m1"
        else:
            class_name = "m2"

    else:
        raise ValueError("Invalid classification name")
    print(f'class_name: {class_name}')
    return class_name

def get_model(data, data_config, model_name, classification):
    class_name = get_class_name(data, data_config, classification)
    base_model_path = Path(Config.BASE_PATH) / "resource" / "models" / "clay_class"
        
    model_path = base_model_path / model_name / f"{model_name}_{class_name}.pt"
    properties_path = base_model_path / model_name / f"{model_name}_{class_name}_properties.json"
    scaler_path = base_model_path / model_name / f"{model_name}_{class_name}_scaler.pkl"
    np_model_path = base_model_path / "NP" / f"NP_{class_name}.np"
    
    with open(properties_path, 'r') as json_file:
        model_properties = json.load(json_file)

    if model_name == "RNN":
        model_class = _RNN
    elif model_name == "LSTM":
        model_class = _LSTM
    elif model_name == "GRU":
        model_class = _GRU
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    Hyperparams = hyperparameters.Hyperparameters(model_properties)
    
    model = model_class.Model(Hyperparams)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    NP_model = load_np(np_model_path)

    scaler = load_scaler(scaler_path)
    
    return model, NP_model, scaler, Hyperparams

def combine_np_prediction(data, NP_model, observed_days):
    data_np = data[['Date', 'settlement']].copy()
    data_np.rename(columns={'Date': 'ds', 'settlement': 'y'}, inplace=True)
    data_np = data_np.iloc[:observed_days]

    future = NP_model.make_future_dataframe(data_np, 
                                         periods=len(data)-observed_days,
                                         n_historic_predictions=True)
    forecast = NP_model.predict(future)

    data['trend'] = forecast['trend'].values
    
    data = data[['settlement', 'fill_height', 'trend']]

    return data

def predict_n_steps(model, input_seq, fill_height, trend, n_steps, observed_days, window_size, constrain_to_last):
    model.eval()
    predictions = []
    device = Config.DEVICE
    input_seq = input_seq.to(device) 
    fill_height = fill_height.to(device)
    
    idx = observed_days

    if trend is not None:
        trend = trend.to(device)

        for _ in range(n_steps):
            prediction = model(input_seq) 
            if constrain_to_last and prediction.item() > input_seq[0, -1, 0].item():
                prediction = input_seq[0, -1, 0].unsqueeze(0).unsqueeze(1)

            predictions.append(prediction.item())
            fill_height_idx = fill_height[idx].unsqueeze(0).unsqueeze(0)
            trend_idx = trend[idx].unsqueeze(0).unsqueeze(0)

            prediction = torch.cat((prediction, fill_height_idx, trend_idx), dim=1)
            input_seq = torch.cat((input_seq[:, 1:], prediction.unsqueeze(0)), dim=1)
            input_seq = input_seq[:, -window_size:, :]  # Keep only the last 'window_size' timesteps

            idx += 1
            
        return predictions
    
    else:
        idx = observed_days

        for _ in range(n_steps):
            prediction = model(input_seq) 
            if constrain_to_last and prediction.item() > input_seq[0, -1, 0].item():
                prediction = input_seq[0, -1, 0].unsqueeze(0).unsqueeze(1)

            predictions.append(prediction.item())
            fill_height_idx = fill_height[idx].unsqueeze(0).unsqueeze(1)

            prediction = torch.cat((prediction, fill_height_idx), dim=1)
            input_seq = torch.cat((input_seq[:, 1:], prediction.unsqueeze(0)), dim=1)
            input_seq = input_seq[:, -window_size:, :]

            idx += 1

        return predictions


def predict_time_series(dict_data, data_config, observed_days, predicted_days, model_name, classification, update, constrain_to_last):
    COLUMNS = ['settlement', 'fill_height']
    TARGET_COLUMN = 'settlement'
    INPUT_SIZE = 2

    set_seed()

    data = reconstruct_dataframe(dict_data)
    
    if model_name == "NP-LSTM" or model_name == "NP-GRU":
        Trend = True
        model_name = model_name.split('-')[1]

    model, NP_model, scaler, hyperparams = get_model(data, data_config, model_name, classification)
    model = model.to(Config.DEVICE)
    
    if Trend:
        data = combine_np_prediction(data, NP_model, observed_days)
        COLUMNS.append('trend')
        INPUT_SIZE += 1
    print('-------------------data-------------------')
    print(data)


    data_scaled = torch.FloatTensor(scaler.transform(data[COLUMNS]))

    print('-------------------update-------------------')
    if update:
        train_data = pd.DataFrame(
            data_scaled[:observed_days, :].numpy(), columns=COLUMNS
        )
        print('start update')
        train_data['idx'] = 0
        model = update_model(model, train_data, hyperparams, TARGET_COLUMN)
        print("-------------Model updated-----------------")
    
    n_step = predicted_days - observed_days
    data_seq = data_scaled[observed_days-hyperparams.WINDOW_SIZE:observed_days,:].unsqueeze(0)
    print(data_seq.shape)
    fill_height = data_scaled[:,1]

    if Trend:
        trend = data_scaled[:,2]
        print('fill_height', len(fill_height))
        print('trend', len(trend))
        predictions = predict_n_steps(model, data_seq, fill_height, trend, n_step, observed_days, hyperparams.WINDOW_SIZE, constrain_to_last)
        predictions_dim = [[i, 0, 0] for i in predictions]
    else:
        trend = None
        predictions = predict_n_steps(model, data_seq, fill_height, trend, n_step, observed_days, hyperparams.WINDOW_SIZE, constrain_to_last)
        predictions_dim = [[i, 0] for i in predictions]

    predictions_origin = scaler.inverse_transform(predictions_dim)
    
    df_predictions = pd.DataFrame(predictions_origin[:,0], columns=['settlement'])

    df_predictions.index = range(observed_days, observed_days + len(predictions_origin))

    return df_predictions