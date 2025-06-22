import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
from pathlib import Path

class ModelTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
        
    def create_data_loaders(self, X_train, y_train, X_val, y_val):
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.config['model']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['model']['batch_size'], shuffle=False)

        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch_X, batch_y in tqdm(train_loader, desc="Training"):
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions.extend(outputs.detach().cpu().numpy())
            targets.extend(batch_y.detach().cpu().numpy())
            
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(targets, np.round(predictions) > 0.5)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc="Validation"):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                predictions.extend(outputs.detach().cpu().numpy())
                targets.extend(batch_y.detach().cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        
        pred_binary = np.array(predictions) > 0.5
        accuracy = accuracy_score(targets, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, pred_binary, average='binary')
        auc = roc_auc_score(targets, predictions)
        
        return avg_loss, accuracy, precision, recall, f1, auc
    
    def train(self, X_train, y_train, X_val, y_val):
        train_loader, val_loader = self.create_data_loaders(X_train, y_train, X_val, y_val)
        print("Validation targets:", np.unique(y_val, return_counts=True))
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['model']['epochs']):
            train_loss, train_accuracy = self.train_epoch(train_loader)

            val_loss, val_accuracy, val_precision, val_recall, val_f1, val_auc = self.validate(val_loader)
            
            print(f"Epoch {epoch+1}/{self.config['model']['epochs']}")
            print(f"Train Loss: {train_loss:.2f}, Train Accuracy: {train_accuracy:.2f}")
            print(f"Val Loss: {val_loss:.2f}, Val Accuracy: {val_accuracy:.2f}, "
                  f"Precision: {val_precision:.2f}, Recall: {val_recall:.2f}, "
                  f"F1 Score: {val_f1:.2f}, AUC: {val_auc:.2f}")
            print("-" * 50)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['model']['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
    def save_model(self):
        model_path = Path(self.config['data']['processed_data_path']) / 'models'
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path / 'lstm_model.pth')
        
    def evaluate(self, X_test, y_test):
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test_tensor).cpu().numpy()
            
        pred_binary = predictions > 0.5
        accuracy = accuracy_score(y_test, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred_binary, average='binary')
        auc = roc_auc_score(y_test, predictions)
        
        print(f"Test Set Results:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-Score: {f1:.2f}")
        print(f"AUC: {auc:.2f}")
        
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}