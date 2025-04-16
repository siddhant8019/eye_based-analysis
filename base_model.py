import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple
from preprocessor import EyeTrackingPreprocessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseEyeTrackingModel:
    """Base class for all eye tracking impairment detection models."""
    
    def __init__(self, model_name: str):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the specific model implementation
        """
        self.model_name = model_name
        self.preprocessor = EyeTrackingPreprocessor()
        self.model = None
        self.scaler = None
        self.is_trained = False
        logger.info(f"Initialized {model_name} model")
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Common preprocessing steps.
        
        Args:
            df: Input DataFrame with eye tracking data
            
        Returns:
            Preprocessed DataFrame
        """
        if not self.preprocessor.validate_data(df):
            raise ValueError("Invalid input data format")
        return self.preprocessor.extract_basic_features(df)
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        """
        Validate input data.
        
        Args:
            df: Input DataFrame with eye tracking data
            
        Returns:
            True if data is valid, False otherwise
        """
        return self.preprocessor.validate_data(df)
    
    def save_results(self, results: Dict, path: str) -> None:
        """
        Save results to CSV.
        
        Args:
            results: Dictionary of results to save
            path: Path to save the results
        """
        try:
            pd.DataFrame([results]).to_csv(path, index=False)
            logger.info(f"Results saved to {path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise ValueError(f"Error saving results: {e}")
            
    def prepare_training_data(self, data: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            data: List of dictionaries containing features and labels
            
        Returns:
            Tuple of (X_scaled, y) for model training
        """
        try:
            df = pd.DataFrame(data)
            X = df.drop('label', axis=1)
            y = df['label']
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y.values
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise ValueError(f"Error preparing training data: {e}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using the trained model.
        
        Args:
            df: Input DataFrame with eye tracking data
            
        Returns:
            Dictionary with prediction results
        """
        raise NotImplementedError("Subclasses must implement predict method")
    
    def extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from preprocessed data.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Dictionary of extracted features
        """
        raise NotImplementedError("Subclasses must implement extract_features method") 