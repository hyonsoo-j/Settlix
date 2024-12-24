import sys
import copy
import torch
from torch import nn
from source.config import Config
from .data_utils import generate_sequences, split_data_and_create_dataloaders


def train_and_validate(model, train_dataloader, valid_dataloader, optimizer, criterion, max_epochs, device, patience):
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            if targets.dim() == 1:
                targets = targets.unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(valid_dataloader)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f" | Early stopping at epoch {epoch + 1}")
                break

        sys.stdout.write(f"\rEpoch [{epoch + 1}/{max_epochs}] "
                         f"- Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}")
        sys.stdout.flush()

    print(f"\nBest validation loss: {best_loss:.6f} at epoch {best_epoch + 1}")
    return best_loss, best_model_wts, best_epoch

def update_model(model, train_data, Hyperparams, TARGET_COLUMN):
    inputs, targets = generate_sequences(train_data, Hyperparams.WINDOW_SIZE, TARGET_COLUMN)
    train_dataloader, valid_dataloader = split_data_and_create_dataloaders(
        inputs, targets, Hyperparams)

    model = model.to(Config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Hyperparams.LEARNING_RATE)

    _, best_model_wts, _ = train_and_validate(
    model, train_dataloader, valid_dataloader, optimizer, criterion, Config.MAX_NUM_EPOCHS, Config.DEVICE, Config.PATIENCE)

    model.load_state_dict(best_model_wts)

    return model

