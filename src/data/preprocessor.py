import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict
import joblib
from pathlib import Path

class TimeSeriesPreprocessor:
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.sequence_length = config['data']['sequence_length']
        self.prediction_horizon = config['data']['prediction_horizon']
        self.sensor_columns = config['data']['sensor_columns'].copy()
        
    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequences = []
        targets = []
        machine_ids = []
        printed = 0
        
        for machine_id in data['machine_id'].unique():
            machine_data = data[data['machine_id'] == machine_id].sort_values('timestamp')

            if not printed:
                print(machine_data.head())
                printed = 1
            
            if len(machine_data) < self.sequence_length + self.prediction_horizon:
                continue
                
            sensor_values = machine_data[self.sensor_columns].values
            failure_values = machine_data['failure'].values
            
            for i in range(len(sensor_values) - self.sequence_length - self.prediction_horizon + 1):
                seq = sensor_values[i:i + self.sequence_length]
                
                target_window = failure_values[i + self.sequence_length:
                                             i + self.sequence_length + self.prediction_horizon]
                target = np.max(target_window) if len(target_window) > 0 else 0
                
                sequences.append(seq)
                targets.append(target)
                machine_ids.append(machine_id)
        
        return np.array(sequences), np.array(targets), np.array(machine_ids)
    
    def temporal_split(self, sequences: np.ndarray, targets: np.ndarray, 
                      machine_ids: np.ndarray, data: pd.DataFrame) -> Tuple:
        unique_timestamps = sorted(data['timestamp'].unique())
        n_timestamps = len(unique_timestamps)
        
        train_end_idx = int(n_timestamps * 0.7)
        val_end_idx = int(n_timestamps * 0.85)
        
        train_end_time = unique_timestamps[train_end_idx]
        val_end_time = unique_timestamps[val_end_idx]
        
        sequence_timestamps = []
        for i, machine_id in enumerate(machine_ids):
            machine_data = data[data['machine_id'] == machine_id].sort_values('timestamp')
            machine_sequences_count = len(sequences[machine_ids == machine_id])
            if machine_sequences_count > 0:
                start_idx = i % len(machine_data)
                if start_idx < len(machine_data):
                    sequence_timestamps.append(machine_data.iloc[start_idx]['timestamp'])
                else:
                    sequence_timestamps.append(machine_data.iloc[-1]['timestamp'])
        
        if len(sequence_timestamps) != len(sequences):
            print("Warning: Using index-based temporal split")
            train_idx = int(len(sequences) * 0.7)
            val_idx = int(len(sequences) * 0.85)
            
            X_train = sequences[:train_idx]
            y_train = targets[:train_idx]
            X_val = sequences[train_idx:val_idx]
            y_val = targets[train_idx:val_idx]
            X_test = sequences[val_idx:]
            y_test = targets[val_idx:]
        else:
            sequence_timestamps = pd.to_datetime(sequence_timestamps)
            
            train_mask = sequence_timestamps <= train_end_time
            val_mask = (sequence_timestamps > train_end_time) & (sequence_timestamps <= val_end_time)
            test_mask = sequence_timestamps > val_end_time
            
            X_train = sequences[train_mask]
            y_train = targets[train_mask]
            X_val = sequences[val_mask]
            y_val = targets[val_mask]
            X_test = sequences[test_mask]
            y_test = targets[test_mask]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple:
        print("Starting preprocessing...")
        
        data = data.sort_values(['machine_id', 'timestamp'])
        
        data = data.ffill().bfill()
        
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        
        print("Adding rolling statistics...")
        for col in self.sensor_columns:
            data[f'{col}_rolling_mean_6h'] = data.groupby('machine_id')[col].rolling(
                window=6, min_periods=1).mean().reset_index(0, drop=True)
            data[f'{col}_rolling_std_6h'] = data.groupby('machine_id')[col].rolling(
                window=6, min_periods=1).std().fillna(0).reset_index(0, drop=True)
            data[f'{col}_rolling_mean_24h'] = data.groupby('machine_id')[col].rolling(
                window=24, min_periods=1).mean().reset_index(0, drop=True)
        
        engineered_cols = self.sensor_columns.copy()
        for col in self.sensor_columns:
            engineered_cols.extend([
                f'{col}_rolling_mean_6h', 
                f'{col}_rolling_std_6h', 
                f'{col}_rolling_mean_24h'
            ])
        
        engineered_cols.extend(['hour', 'day_of_week', 'month'])
        
        self.sensor_columns = [col for col in engineered_cols if col in data.columns]
        
        print(f"Using {len(self.sensor_columns)} features: {self.sensor_columns}")
        
        data = data.bfill().fillna(0)
        
        print("Normalizing features...")
        feature_data = data[self.sensor_columns].copy()
        data[self.sensor_columns] = self.scaler.fit_transform(feature_data)
        
        print("Creating sequences...")
        X, y, machine_ids = self.create_sequences(data)
        
        print(f"Created {len(X)} sequences")
        print(f"Failure rate in sequences: {y.mean():.4f}")
        
        unique, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")
        
        if len(unique) == 1:
            raise ValueError(f"Only one class present in targets: {unique[0]}. "
                           "Check your failure generation logic.")
        
        print("Performing temporal split...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_split(
            X, y, machine_ids, data
        )
        
        for split_name, split_y in [("train", y_train), ("val", y_val), ("test", y_test)]:
            unique_split = np.unique(split_y)
            print(f"{split_name} split: {len(split_y)} samples, "
                  f"failure rate: {split_y.mean():.4f}, "
                  f"classes: {unique_split}")
            
            if len(unique_split) == 1:
                print(f"WARNING: {split_name} split has only one class!")
        
        processed_path = Path(self.config['data']['processed_data_path'])
        processed_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, processed_path / "scaler.pkl")
        
        joblib.dump(self.sensor_columns, processed_path / "feature_names.pkl")
        
        print("Preprocessing completed!")
        
        return X_train, X_val, X_test, y_train, y_val, y_test