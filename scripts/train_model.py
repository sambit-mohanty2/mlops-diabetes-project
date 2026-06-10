"""Training script for diabetic risk prediction model."""

import logging
import argparse
from pathlib import Path
import yaml

import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader, DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.models.trainer import ModelBuilder, ModelTrainer
from src.evaluation.metrics import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(config: dict):
    """Prepare and preprocess data."""
    logger.info("Starting data preparation...")
    
    # Load data
    loader = DataLoader(
        data_path=config['data']['raw_path'],
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    data = loader.load()
    
    # Get basic stats
    stats = loader.get_basic_stats()
    logger.info(f"Data shape: {stats['shape']}")
    logger.info(f"Missing values: {stats['missing_values']}")
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    data = preprocessor.handle_missing_values(data, strategy='mean')
    data = preprocessor.remove_outliers(data, method='iqr', threshold=1.5)
    
    # Prepare features and target
    feature_cols = config['features']['numerical_features']
    X = data[feature_cols]
    y = data[config['data']['stratify_on']]
    
    # Split data
    X_train, X_test, y_train, y_test = loader.split_data(X, y)
    
    # Normalize data
    X_train, X_test = preprocessor.normalize_data(X_train, X_test, method='standard')
    
    # Save processed data
    processed_path = Path(config['data']['processed_path'])
    processed_path.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(processed_path / "X_train.csv", index=False)
    X_test.to_csv(processed_path / "X_test.csv", index=False)
    y_train.to_csv(processed_path / "y_train.csv", index=False)
    y_test.to_csv(processed_path / "y_test.csv", index=False)
    
    logger.info("Data preparation completed")
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, config: dict):
    """Train the model."""
    logger.info("Starting model training...")
    
    # Create model builder
    builder = ModelBuilder(
        model_type=config['model']['algorithm'],
        random_state=config['model']['random_state']
    )
    
    # Build and train model
    builder.build_model(config['model']['hyperparameters'])
    builder.train(X_train, y_train)
    
    # Save model
    models_path = Path("models")
    models_path.mkdir(parents=True, exist_ok=True)
    
    import joblib
    model_file = models_path / f"{config['model']['algorithm']}_model.pkl"
    joblib.dump(builder.model, model_file)
    logger.info(f"Model saved to {model_file}")
    
    return builder.model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    logger.info("Starting model evaluation...")
    
    evaluator = ModelEvaluator(model, X_test, y_test)
    metrics = evaluator.calculate_metrics()
    
    logger.info(f"Model Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Generate visualizations
    outputs_path = Path("outputs")
    outputs_path.mkdir(parents=True, exist_ok=True)
    
    evaluator.plot_confusion_matrix(str(outputs_path / "confusion_matrix.png"))
    evaluator.plot_roc_curve(str(outputs_path / "roc_curve.png"))
    
    logger.info("Model evaluation completed")
    return metrics


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train diabetic risk prediction model")
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "full"],
        default="full",
        help="Execution mode"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    try:
        if args.mode in ["prepare", "full"]:
            X_train, X_test, y_train, y_test = prepare_data(config)
        
        if args.mode in ["train", "full"]:
            if args.mode == "train":
                # Load prepared data
                X_train = pd.read_csv("data/processed/X_train.csv")
                X_test = pd.read_csv("data/processed/X_test.csv")
                y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
                y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
            
            model = train_model(X_train, y_train, config)
            metrics = evaluate_model(model, X_test, y_test)
        
        logger.info("Pipeline completed successfully!")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
