import torch
import joblib
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys

sys.path.append('src')

from src.models.lstm_model import LSTMModel
from src.data.preprocessor import TimeSeriesPreprocessor

class PredictiveMaintenanceInference:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.load_model_and_scaler()
        
    def load_model_and_scaler(self):
        scaler_path = Path(self.config['data']['processed_data_path']) / 'scaler.pkl'
        self.scaler = joblib.load(scaler_path)
        
        input_size = len(self.config['data']['sensor_columns']) * 3 + 2
        self.model = LSTMModel(
            input_size=input_size,
            lstm_units=self.config['model']['lstm_units'],
            dropout_rate=self.config['model']['dropout_rate']
        )
        
        model_path = Path(self.config['data']['processed_data_path']) / 'models' / 'lstm_model.pth'
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
    def preprocess_real_time_data(self, data: pd.DataFrame) -> np.ndarray:
        required_columns = self.config['data']['sensor_columns'] + ['machine_id', 'timestamp']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
            
        data = data.sort_values(['machine_id', 'timestamp'])
        
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        
        sensor_cols = self.config['data']['sensor_columns']
        for col in sensor_cols:
            data[f'{col}_rolling_mean'] = data.groupby('machine_id')[col].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True)
            data[f'{col}_rolling_std'] = data.groupby('machine_id')[col].rolling(window=6, min_periods=1).std().fillna(0).values
            
        feature_cols = []
        for col in sensor_cols:
            feature_cols.extend([col, f'{col}_rolling_mean', f'{col}_rolling_std'])
        feature_cols.extend(['hour', 'day_of_week'])
        
        scaled_features = self.scaler.transform(data[feature_cols])
        
        return scaled_features
    
    def predict_failure_risk(self, data: pd.DataFrame) -> dict:
        features = self.preprocess_real_time_data(data)
        
        sequence_length = self.config['data']['sequence_length']
        
        predictions = {}
        
        for machine_id in data['machine_id'].unique():
            machine_mask = data['machine_id'] == machine_id
            machine_features = features[machine_mask]
            
            if len(machine_features) >= sequence_length:
                sequence = machine_features[-sequence_length:]
            else:
                sequence = np.zeros((sequence_length, machine_features.shape[1]))
                sequence[-len(machine_features):] = machine_features
            
            sequence = np.expand_dims(sequence, 0)
            
            # Predict
            with torch.no_grad():
                risk_score = self.model(torch.FloatTensor(sequence)).item()
            
            predictions[machine_id] = {
                'failure_risk': risk_score,
                'risk_level': self.get_risk_level(risk_score),
                'recommendation': self.get_recommendation(risk_score)
            }
        
        return predictions
    
    def get_risk_level(self, risk_score: float) -> str:
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_recommendation(self, risk_score: float) -> str:
        if risk_score >= 0.8:
            return "Immediate maintenance required - Stop operation"
        elif risk_score >= 0.6:
            return "Schedule maintenance within 24 hours"
        elif risk_score >= 0.4:
            return "Monitor closely - Schedule preventive maintenance"
        else:
            return "Normal operation - Continue monitoring"

def main():
    predictor = PredictiveMaintenanceInference()
    
    sample_data = pd.DataFrame({
        'machine_id': [1, 1, 2, 2, 3, 3],
        'timestamp': pd.date_range('2024-01-01', periods=6, freq='H'),
        'temperature': [75, 82, 78, 95, 71, 73],
        'vibration': [0.5, 0.8, 0.4, 1.2, 0.3, 0.4],
        'pressure': [100, 95, 102, 85, 105, 103],
        'humidity': [45, 48, 43, 52, 41, 42],
        'speed': [1800, 1750, 1820, 1600, 1850, 1840]
    })
    
    predictions = predictor.predict_failure_risk(sample_data)
    
    print("🔮 Failure Risk Predictions")
    print("=" * 50)
    
    for machine_id, pred in predictions.items():
        print(f"Machine {machine_id}:")
        print(f"  Risk Score: {pred['failure_risk']:.3f}")
        print(f"  Risk Level: {pred['risk_level']}")
        print(f"  Recommendation: {pred['recommendation']}")
        print()

if __name__ == "__main__":
    main()
        