"""Model evaluation and metrics."""

import logging
from typing import Dict, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model performance."""
    
    def __init__(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Initialize ModelEvaluator.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
        
    def generate_predictions(self):
        """Generate predictions."""
        self.y_pred = self.model.predict(self.X_test)
        
        if hasattr(self.model, 'predict_proba'):
            self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        else:
            self.y_pred_proba = self.model.predict(self.X_test)
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        if self.y_pred is None:
            self.generate_predictions()
        
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred, zero_division=0),
            'recall': recall_score(self.y_test, self.y_pred, zero_division=0),
            'f1': f1_score(self.y_test, self.y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba),
        }
        
        logger.info(f"Metrics: {self.metrics}")
        return self.metrics
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        if self.y_pred is None:
            self.generate_predictions()
        
        report = classification_report(self.y_test, self.y_pred)
        logger.info(f"\nClassification Report:\n{report}")
        return report
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if self.y_pred is None:
            self.generate_predictions()
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        return cm
    
    def plot_confusion_matrix(self, save_path: str = None):
        """Plot confusion matrix."""
        cm = self.get_confusion_matrix()
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, save_path: str = None):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {save_path}")
        
        plt.close()
    
    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report."""
        self.calculate_metrics()
        
        report = {
            'metrics': self.metrics,
            'classification_report': self.get_classification_report(),
            'confusion_matrix': self.get_confusion_matrix().tolist(),
        }
        
        return report
