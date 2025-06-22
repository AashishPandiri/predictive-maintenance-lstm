import yaml
import sys

sys.path.append('src')

from src.data.data_loader import IoTDataLoader
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.lstm_model import LSTMModel
from src.models.trainer import ModelTrainer

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    print("Predictive Maintenance Model Training")
    print("=" * 50)
    
    print("Loading data...")
    data_loader = IoTDataLoader('config.yaml')
    data = data_loader.load_data()
    
    print(f"Loaded {len(data)} records from {data['machine_id'].nunique()} machines.")
    print(f"Data range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"Failure rate: {data['failure'].mean() * 100:.3f}%")
    
    print("Preprocessing data...")
    preprocessor = TimeSeriesPreprocessor(config)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess_data(data)

    print(f"Training sequences: {X_train.shape}")
    print(f"Validation sequences: {X_val.shape}")
    print(f"Test sequences: {X_val.shape}")
    print(f"Features per timestep: {X_train.shape[2]}")
    
    print("Creating model...")
    model = LSTMModel(
        input_size=X_train.shape[2],
        lstm_units=config['model']['lstm_units'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    print(f"Model architecture: {config['model']['lstm_units']} LSTM units, {config['model']['dropout_rate']} dropout rate")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("Training model...")
    trainer = ModelTrainer(model, config)
    trainer.train(X_train, y_train, X_val, y_val)
    
    print("Evaluating model on test set...")
    results = trainer.evaluate(X_test, y_test)
    
    print("Training completed successfully!")
    print(f"Final test accuracy: {results['accuracy']:.3f}")
    print(f"Model and preprocessor saved to {config['data']['processed_data_path']}")
    
if __name__ == "__main__":
    main()