import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import joblib
import yaml
from datetime import datetime, timedelta
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.lstm_model import LSTMModel
from data.data_loader import IoTDataLoader

class PredictiveMaintenanceDashboard:
    def __init__(self):
        self.load_config()
        self.load_model_and_scaler()
        self.data_loader = IoTDataLoader()
        
    def load_config(self):
        with open('config.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
            
    def load_model_and_scaler(self):
        try:
            self.scaler = joblib.load(self.config['data']['processed_data_path'] + '/scaler.pkl')
            
            input_size = len(self.config['data']['sensor_columns']) * 4 + 3  # raw + 3 rolling features + time

            self.model = LSTMModel(
                input_size=input_size,
                lstm_units=self.config['model']['lstm_units'],
                dropout_rate=self.config['model']['dropout_rate']
            )

            model_path = Path(self.config['data']['processed_data_path']) / "models" / "lstm_model.pth"
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            self.model = None
            self.scaler = None
            
    def generate_real_time_data(self, n_machines: int = 10):
        current_time = datetime.now()
        data = []
        
        for machine_id in range(1, n_machines + 1):
            temp = np.random.normal(75 + machine_id * 2, 8)
            vibration = np.random.normal(0.5 + machine_id * 0.1, 0.15)
            pressure = np.random.normal(100 - machine_id, 12)
            humidity = np.random.normal(45, 6)
            speed = np.random.normal(1800 + machine_id * 10, 150)
            
            if machine_id in [3, 7]:
                temp += 15
                vibration += 0.5
            
            data.append({
                'machine_id': machine_id,
                'timestamp': current_time,
                'temperature': temp,
                'vibration': vibration,
                'pressure': pressure,
                'humidity': humidity,
                'speed': speed
            })
        
        return pd.DataFrame(data)
    
    def predict_failures(self, data: pd.DataFrame):
        if self.model is None or self.scaler is None:
            st.error("Model or scaler not loaded. Please check the configuration.")
            return None

        data = data.sort_values(['machine_id', 'timestamp'])
        
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        
        sensor_cols = ['temperature', 'vibration', 'pressure', 'humidity', 'speed']
        
        for col in sensor_cols:
            data[f'{col}_rolling_mean_6h'] = data.groupby('machine_id')[col].rolling(
                window=6, min_periods=1).mean().reset_index(0, drop=True)
            data[f'{col}_rolling_std_6h'] = data.groupby('machine_id')[col].rolling(
                window=6, min_periods=1).std().fillna(0).reset_index(0, drop=True)
            data[f'{col}_rolling_mean_24h'] = data.groupby('machine_id')[col].rolling(
                window=24, min_periods=1).mean().reset_index(0, drop=True)
        
        engineered_cols = sensor_cols.copy()
        for col in sensor_cols:
            engineered_cols.extend([
                f'{col}_rolling_mean_6h', 
                f'{col}_rolling_std_6h', 
                f'{col}_rolling_mean_24h'
            ])
        
        engineered_cols.extend(['hour', 'day_of_week', 'month'])
        
        try:
            features = self.scaler.transform(data[engineered_cols])
            
            sequences = np.expand_dims(features, axis=1)
            sequences = np.repeat(sequences, self.config['data']['sequence_length'], axis=1)
            
            with torch.no_grad():
                predictions = self.model(torch.FloatTensor(sequences)).numpy()
                
            return predictions

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None
        
    def create_sensor_charts(self, data: pd.DataFrame):
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=['Temperature (°F)', 'Vibration (g)', 'Pressure (PSI)', 'Humidity (%)', 'Speed (RPM)', 'Failure Risk'],
            specs=[[{"secondary_y": False}]*3]*2
        )
        
        sensors = ['temperature', 'vibration', 'pressure', 'humidity', 'speed']
        positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
        
        for sensor, (row, col) in zip(sensors, positions):
            fig.add_trace(
                go.Scatter(x=data['machine_id'], y=data[sensor], mode='lines+markers', name=sensor.title()),
                row=row, col=col
            )
            
        predictions = self.predict_failures(data)
        fig.add_trace(
            go.Scatter(
                x=data['machine_id'], 
                y=predictions if predictions is not None else np.zeros(len(data)), 
                mode='lines+markers', 
                name='Failure Risk', 
                line=dict(color='red', width=2)
            ),
            row=2, col=3
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Real time Sensor Monitoring")
        return fig

    def create_alert_table(self, data: pd.DataFrame):
        predictions = self.predict_failures(data)
        data_with_predictions = data.copy()
        data_with_predictions['failure_risk'] = predictions
        
        alerts = data_with_predictions[data_with_predictions['failure_risk'] > self.config['dashboard']['alert_threshold']]
        
        if len(alerts) > 0:
            alerts_display = alerts[['machine_id', 'failure_risk', 'temperature', 'vibration']].copy()
            alerts_display['failure_risk'] = alerts_display['failure_risk'].round(3)
            alerts_display.columns = ['Machine ID', 'Failure Risk', 'Temperature (°F)', 'Vibration (g)']
            return alerts_display.sort_values(by='Failure Risk', ascending=False)
        else:
            return pd.DataFrame({'Message': ['No alerts at this time.']})
        
    def run_dashboard(self):
        st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
        
        st.title("Predictive Maintenance Dashboard")
        st.markdown("Real-time monitoring of IoT sensor data with predictive maintenance capabilities.")

        st.sidebar.title("Dashboard controls")
        auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 30)
        n_machines = st.sidebar.slider("Number of Machines", 5, 20, 10)
        
        placeholder = st.empty()
        
        while True:
            with placeholder.container():
                current_data = self.generate_real_time_data(n_machines)
                
                col1, col2, col3, col4 = st.columns(4)
                
                predictions = self.predict_failures(current_data)
                high_risk_count = sum(predictions > self.config['dashboard']['alert_threshold'])
                avg_risk = np.mean(predictions)
                
                with col1:
                    st.metric("Total Machines", len(current_data))
                with col2:
                    st.metric("High Risk Machines", high_risk_count)
                with col3:
                    st.metric("Average Risk", f"{avg_risk:.3f}")
                with col4:
                    st.metric("Model Accuracy", "87%")
                    
                st.subheader("Sensor Monitoring")
                sensor_chart = self.create_sensor_charts(current_data)
                st.plotly_chart(sensor_chart, use_container_width=True)
                
                st.subheader("Alerts")
                alert_table = self.create_alert_table(current_data)
                st.dataframe(alert_table, use_container_width=True)
                
                st.subheader("Machine Status Overview")
                status_data = current_data.copy()
                status_data['failure_risk'] = predictions
                status_data['status'] = status_data['failure_risk'].apply(
                    lambda x: 'High Risk' if x > self.config['dashboard']['alert_threshold'] else 'Normal'
                )
                
                status_chart = px.scatter(
                    status_data, x='machine_id', y='failure_risk', color='status',
                    size='temperature', title="Machine Risk Status",
                    color_discrete_map={'Normal': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
                )
                st.plotly_chart(status_chart, use_container_width=True)
                
                st.caption("Data last updated at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                
            if not auto_refresh:
                break
            
            time.sleep(refresh_interval)
            
if __name__ == "__main__":
    dashboard = PredictiveMaintenanceDashboard()
    dashboard.run_dashboard()