"""Data loading and processing utilities."""

import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare data for model training."""
    
    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to data file
            test_size: Test set size ratio
            random_state: Random state for reproducibility
        """
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.random_state = random_state
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load(self) -> pd.DataFrame:
        """Load data from file."""
        try:
            if self.data_path.suffix == '.csv':
                self.data = pd.read_csv(self.data_path)
            elif self.data_path.suffix in ['.xlsx', '.xls']:
                self.data = pd.read_excel(self.data_path)
            elif self.data_path.suffix == '.parquet':
                self.data = pd.read_parquet(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
            logger.info(f"Loaded data from {self.data_path} with shape {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_basic_stats(self) -> dict:
        """Get basic statistics about the data."""
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicates': self.data.duplicated().sum(),
        }
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        logger.info(f"Train set size: {self.X_train.shape[0]}, Test set size: {self.X_test.shape[0]}")
        return self.X_train, self.X_test, self.y_train, self.y_test


class DataPreprocessor:
    """Handle data preprocessing and cleaning."""
    
    def __init__(self):
        """Initialize DataPreprocessor."""
        self.scaler = None
        self.imputer = None
        
    def handle_missing_values(self, data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Handle missing values.
        
        Args:
            data: Input dataframe
            strategy: Imputation strategy ('mean', 'median', 'mode', 'drop')
            
        Returns:
            Dataframe with missing values handled
        """
        if strategy == 'drop':
            return data.dropna()
        elif strategy in ['mean', 'median', 'mode']:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy=strategy)
            return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def remove_outliers(self, data: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers using IQR or Z-score method.
        
        Args:
            data: Input dataframe
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            Dataframe with outliers removed
        """
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if method == 'iqr':
            Q1 = data[numeric_columns].quantile(0.25)
            Q3 = data[numeric_columns].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = ~((data[numeric_columns] < lower_bound) | (data[numeric_columns] > upper_bound)).any(axis=1)
            return data[mask]
        
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(data[numeric_columns]))
            mask = ~(z_scores > threshold).any(axis=1)
            return data[mask]
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def normalize_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame, method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize data.
        
        Args:
            X_train: Training data
            X_test: Test data
            method: 'standard' or 'minmax'
            
        Returns:
            Normalized train and test data
        """
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        self.scaler = scaler
        
        return X_train_scaled, X_test_scaled
