import torch.nn as nn
import torch.nn.functional as F
from typing import List

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, lstm_units: List[int], dropout_rate: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.lstm_units = lstm_units
        
        self.lstm_layers = nn.ModuleList()
        
        num_layers = len(lstm_units)
        for i in range(num_layers):
            in_size = input_size if i == 0 else lstm_units[i-1]
            out_size = lstm_units[i]
            dropout_rate = 0 if i == 0 else dropout_rate
            self.lstm_layers.append(nn.LSTM(in_size, out_size, batch_first=True, dropout=dropout_rate))

        self.dropout = nn.Dropout(dropout_rate)
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_units[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        for lstm_layer in self.lstm_layers:
            x, _ = lstm_layer(x)
            x = self.dropout(x)
            
        x = x[:, -1, :]
        
        output = self.classifier(x)
        return output.squeeze()
    
class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True