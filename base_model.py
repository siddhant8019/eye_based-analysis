import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple
from preprocessor import EyeTrackingPreprocessor

class BaseEyeTrackingModel:
    def __init__(self):
        self.preprocessor = EyeTrackingPreprocessor()
        self.model = None
        self.scaler = None
        self.is_trained = False
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Common preprocessing steps."""
        if not self.preprocessor.validate_data(df):
            raise ValueError("Invalid input data format")
        return self.preprocessor.extract_basic_features(df)
    
    def validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input data."""
        return self.preprocessor.validate_data(df)
    
    def save_results(self, results: Dict, path: str) -> None:
        """Save results to CSV."""
        try:
            pd.DataFrame([results]).to_csv(path, index=False)
        except Exception as e:
            raise ValueError(f"Error saving results: {e}")
            
    def prepare_training_data(self, data: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training."""
        try:
            df = pd.DataFrame(data)
            X = df.drop('label', axis=1)
            y = df['label']
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y.values
        except Exception as e:
            raise ValueError(f"Error preparing training data: {e}") 