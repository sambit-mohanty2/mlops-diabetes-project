"""Model training and management."""

import logging
import json
from pathlib import Path
from typing import Dict, Tuple, Any

import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

logger = logging.getLogger(__name__)


class ModelBuilder:
    """Build and train models."""
    
    def __init__(self, model_type: str = 'catboost', random_state: int = 42):
        """
        Initialize ModelBuilder.
        
        Args:
            model_type: Type of model ('logistic', 'rf', 'xgboost', 'lightgbm', 'catboost')
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.hyperparameters = {}
        
    def build_model(self, hyperparameters: Dict[str, Any] = None):
        """
        Build model with specified hyperparameters.
        
        Args:
            hyperparameters: Model hyperparameters
        """
        if hyperparameters is None:
            hyperparameters = {}
        
        self.hyperparameters = hyperparameters
        
        if self.model_type == 'logistic':
            self.model = LogisticRegression(random_state=self.random_state, **hyperparameters)
        
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                **hyperparameters
            )
        
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False,
                **hyperparameters
            )
        
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                random_state=self.random_state,
                **hyperparameters
            )
        
        elif self.model_type == 'catboost':
            self.model = CatBoostClassifier(
                random_state=self.random_state,
                verbose=0,
                **hyperparameters
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Built {self.model_type} model with hyperparameters: {hyperparameters}")
        return self.model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, validation_data: Tuple = None) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training target
            validation_data: Optional (X_val, y_val) for early stopping
            
        Returns:
            Training history/info
        """
        if self.model is None:
            self.build_model()
        
        if validation_data is not None and self.model_type in ['xgboost', 'lightgbm', 'catboost']:
            X_val, y_val = validation_data
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        logger.info(f"Trained {self.model_type} model")
        return {'status': 'success', 'model_type': self.model_type}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        if self.model is None:
            self.build_model()
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')
        
        cv_info = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'fold_scores': scores.tolist()
        }
        
        logger.info(f"Cross-validation score: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return cv_info


class ModelTrainer:
    """Manage model training and comparison."""
    
    def __init__(self, mlflow_tracking_uri: str = None):
        """
        Initialize ModelTrainer.
        
        Args:
            mlflow_tracking_uri: MLflow tracking URI
        """
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.models = {}
        self.results = {}
        
    def train_multiple_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        models_config: Dict = None
    ) -> Dict:
        """
        Train multiple models.
        
        Args:
            X_train: Training features
            y_train: Training target
            models_config: Configuration for models to train
            
        Returns:
            Dictionary of trained models
        """
        if models_config is None:
            models_config = {
                'logistic': {},
                'rf': {},
                'xgboost': {},
                'lightgbm': {},
                'catboost': {'iterations': 500}
            }
        
        for model_name, hyperparams in models_config.items():
            try:
                logger.info(f"Training {model_name}...")
                builder = ModelBuilder(model_type=model_name)
                builder.build_model(hyperparams)
                builder.train(X_train, y_train)
                self.models[model_name] = builder.model
                logger.info(f"Successfully trained {model_name}")
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        return self.models
    
    def save_model(self, model_name: str, save_path: str):
        """
        Save model to disk.
        
        Args:
            model_name: Name of the model
            save_path: Path to save model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        import joblib
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models[model_name], save_path)
        logger.info(f"Saved {model_name} to {save_path}")
    
    def load_model(self, model_path: str) -> Any:
        """Load model from disk."""
        import joblib
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
