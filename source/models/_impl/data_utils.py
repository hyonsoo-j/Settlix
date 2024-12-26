import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from source.config import Config

def reconstruct_dataframe(dict_data):

    date = dict_data['Date'].reset_index(drop=True)

    settlement = pd.concat(
        [dict_data['before_settlement'], dict_data['after_settlement']]
    ).sort_index().reset_index(drop=True)

    fill_height = pd.concat(
        [dict_data['before_height'], dict_data['after_height']]
    ).sort_index().reset_index(drop=True)

    predicted_settlement = pd.concat(
        [dict_data['before_settlement'], dict_data['predicted_settlement']]
    ).sort_index().reset_index(drop=True)

    reconstructed_df = pd.DataFrame({
        'Date': date['Date'],
        'settlement': settlement['settlement'],
        'fill_height': fill_height['fill_height'],
        'predicted_settlement': predicted_settlement['settlement']
    })

    print('Reconstructed DataFrame:')
    print(reconstructed_df)
    print('-----------------------------')
    return reconstructed_df


def generate_sequences(data, window_size, TARGET_COLUMN):
    input_sequences = []
    target_sequences = []
    for _, group_data in data.groupby('idx'):
        if len(group_data) >= window_size + 1:
            for i in range(len(group_data) - window_size):
                input_seq = group_data.iloc[i:i + window_size].drop(columns=['idx']).values
                target_seq = group_data.iloc[i + window_size][TARGET_COLUMN]
                input_sequences.append(input_seq)
                target_sequences.append(target_seq)
    return input_sequences, target_sequences

class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

def split_data_and_create_dataloaders(inputs, targets, Hyperparams):

    train_inputs, valid_inputs, train_targets, valid_targets = train_test_split(
        inputs, targets, train_size = Config.TRAIN_RATIO, random_state = Config.RANDOM_SEED)

    train_dataset = TimeSeriesDataset(train_inputs, train_targets)
    valid_dataset = TimeSeriesDataset(valid_inputs, valid_targets)
    
    train_dataloader = DataLoader(train_dataset, batch_size = Hyperparams.BATCH_SIZE, shuffle = False)
    valid_dataloader = DataLoader(valid_dataset, batch_size = Hyperparams.BATCH_SIZE, shuffle = False)
    
    return train_dataloader, valid_dataloader

