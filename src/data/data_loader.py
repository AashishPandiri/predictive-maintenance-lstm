import pandas as pd
import numpy as np
from pathlib import Path
import yaml

class IoTDataLoader:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        
    def generate_synthetic_data(self, n_samples: int = 100000) -> pd.DataFrame:
        np.random.seed(42)
        
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='5min')
        
        machine_ids = np.random.choice(range(1, self.data_config['n_machines'] + 1), 
                                     size=n_samples)
        
        data = []
        machine_states = {}
        
        for i in range(n_samples):
            machine_id = machine_ids[i]
            
            if machine_id not in machine_states:
                machine_states[machine_id] = {
                    'degradation_level': 0.0,
                    'last_maintenance': 0,
                    'operating_hours': 0
                }
            
            state = machine_states[machine_id]
            
            state['operating_hours'] += 1
            
            degradation_rate = np.random.uniform(0.0001, 0.0005)  # Very slow degradation
            state['degradation_level'] = min(1.0, state['degradation_level'] + degradation_rate)
            
            if np.random.random() < 0.0001:
                state['degradation_level'] *= 0.1
                state['last_maintenance'] = 0
            
            state['last_maintenance'] += 1
            
            base_temp = 75 + np.random.normal(0, 3)
            base_vibration = 0.3 + np.random.normal(0, 0.05)
            base_pressure = 100 + np.random.normal(0, 5)
            base_humidity = 45 + np.random.normal(0, 3)
            base_speed = 1800 + np.random.normal(0, 50)
            
            degradation = state['degradation_level']
            
            temp = base_temp + degradation * 20 + np.random.normal(0, 2)
            
            vibration = base_vibration + degradation * 0.8 + np.random.normal(0, 0.1)
            
            pressure = base_pressure - degradation * 25 + np.random.normal(0, 3)
            
            humidity = base_humidity + (temp - 75) * 0.3 + np.random.normal(0, 2)
            
            speed_variation = degradation * 300
            speed = base_speed + np.random.normal(0, 30 + speed_variation)
            
            base_failure_prob = 0.001
            
            failure_prob = base_failure_prob * (1 + np.exp(degradation * 8))
            
            if temp > 95:
                failure_prob *= 2
            if vibration > 1.0:
                failure_prob *= 2.5
            if pressure < 75:
                failure_prob *= 1.8
            if abs(speed - 1800) > 400:
                failure_prob *= 1.5
            
            maintenance_factor = 1 + (state['last_maintenance'] / 8760) * 0.5
            failure_prob *= maintenance_factor
            
            failure_prob = min(failure_prob, 0.15)
            
            failure = np.random.binomial(1, failure_prob)
            
            if failure:
                state['degradation_level'] = np.random.uniform(0.0, 0.2)  
                state['last_maintenance'] = 0
            
            data.append([
                dates[i], machine_id, temp, vibration, 
                pressure, humidity, speed, failure
            ])
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'machine_id', 'temperature', 'vibration',
            'pressure', 'humidity', 'speed', 'failure'
        ])
        
        df = df.sort_values(['machine_id', 'timestamp'])
        failure_rate = df['failure'].mean()
        print(f"Generated data with failure rate: {failure_rate:.4f} ({failure_rate*100:.2f}%)")
        
        if failure_rate < 0.005:
            print("Adjusting failure rate to ensure model training...")
            n_additional_failures = int(len(df) * 0.01) - df['failure'].sum()
            if n_additional_failures > 0:
                high_risk_mask = (
                    (df['temperature'] > df['temperature'].quantile(0.8)) |
                    (df['vibration'] > df['vibration'].quantile(0.8)) |
                    (df['pressure'] < df['pressure'].quantile(0.2))
                )
                high_risk_indices = df[high_risk_mask & (df['failure'] == 0)].index
                if len(high_risk_indices) > n_additional_failures:
                    failure_indices = np.random.choice(
                        high_risk_indices, 
                        size=n_additional_failures, 
                        replace=False
                    )
                    df.loc[failure_indices, 'failure'] = 1
                    
        final_failure_rate = df['failure'].mean()
        print(f"Final failure rate: {final_failure_rate:.4f} ({final_failure_rate*100:.2f}%)")
        
        return df
    
    def load_data(self) -> pd.DataFrame:
        """Load data from file or generate synthetic data"""
        raw_path = Path(self.data_config['raw_data_path'])
        
        if (raw_path / 'sensor_data.csv').exists():
            print("Loading existing sensor data...")
            return pd.read_csv(raw_path / 'sensor_data.csv', parse_dates=['timestamp'])
        else:
            print("Generating synthetic data...")
            data = self.generate_synthetic_data()
            raw_path.mkdir(parents=True, exist_ok=True)
            data.to_csv(raw_path / 'sensor_data.csv', index=False)
            return data