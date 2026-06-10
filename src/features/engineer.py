"""Feature engineering utilities."""

import logging
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif, SelectKBest, RFE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handle feature engineering and transformation."""
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.feature_names = None
        self.selector = None
        
    def create_polynomial_features(self, X: pd.DataFrame, degree: int = 2, include_bias: bool = False) -> pd.DataFrame:
        """
        Create polynomial features.
        
        Args:
            X: Input features
            degree: Polynomial degree
            include_bias: Include bias term
            
        Returns:
            DataFrame with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X)
        
        # Get feature names
        feature_names = poly.get_feature_names_out(X.columns)
        return pd.DataFrame(X_poly, columns=feature_names)
    
    def create_interaction_features(self, X: pd.DataFrame, interactions: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features.
        
        Args:
            X: Input features
            interactions: List of feature pairs to interact
            
        Returns:
            DataFrame with added interaction features
        """
        X_copy = X.copy()
        
        for feat1, feat2 in interactions:
            if feat1 in X.columns and feat2 in X.columns:
                X_copy[f"{feat1}_x_{feat2}"] = X[feat1] * X[feat2]
                logger.info(f"Created interaction feature: {feat1}_x_{feat2}")
        
        return X_copy
    
    def select_features_mutual_information(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
        """
        Select top k features using mutual information.
        
        Args:
            X: Input features
            y: Target variable
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        logger.info(f"Selected {len(selected_features)} features using mutual information: {selected_features}")
        
        self.selector = selector
        return pd.DataFrame(X_selected, columns=selected_features)
    
    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series, k: int = 10) -> pd.DataFrame:
        """
        Select top k features using Recursive Feature Elimination.
        
        Args:
            X: Input features
            y: Target variable
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        logger.info(f"Selected {len(selected_features)} features using RFE: {selected_features}")
        
        self.selector = selector
        return pd.DataFrame(X_selected, columns=selected_features)
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            DataFrame with feature importance scores
        """
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 5 important features:\n{importance_df.head()}")
        return importance_df
