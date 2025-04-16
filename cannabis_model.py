import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Tuple
from base_model import BaseEyeTrackingModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CannabisModel(BaseEyeTrackingModel):
    """Model for detecting Cannabis impairment based on eye tracking data."""
    
    # Feature importance weights based on research
    FEATURE_WEIGHTS = {
        "convergence_features": 0.40,  # Most important according to research
        "nystagmus_features": 0.30,    # Second most important
        "pupil_features": 0.30,        # Supporting features
    }

    # Thresholds based on research findings
    THRESHOLDS = {
        "convergence": {
            "normal": 2.0,      # degrees
            "impaired": 5.0,    # degrees
        },
        "nystagmus": {
            "velocity": 2.5,    # degrees/second
            "frequency": 4.0,   # Hz
        },
        "pupil": {
            "normal_size": 3.5, # mm
            "impaired_size": 5.0, # mm
        },
    }

    def __init__(self):
        """Initialize the model."""
        super().__init__("CannabisModel")
        self.imputer = SimpleImputer(strategy='mean')
        logger.info("Initialized CannabisModel")

    def calculate_convergence_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate convergence metrics from eye tracking data.
        
        Args:
            df: Preprocessed DataFrame with eye tracking data
            
        Returns:
            Dictionary of convergence metrics
        """
        try:
            # Calculate convergence angle (difference between left and right eye angles)
            df['convergence_angle'] = df['right_eye_angle_x'] - df['left_eye_angle_x']
            
            # Calculate convergence velocity
            df['convergence_velocity'] = df['convergence_angle'].diff() / df['dt']
            
            # Apply smoothing
            window = min(5, len(df) - 1)
            if window > 3:
                df['convergence_velocity'] = savgol_filter(df['convergence_velocity'].fillna(0), window, 2)
            
            # Calculate metrics
            metrics = {
                'mean_convergence': float(df['convergence_angle'].mean()),
                'std_convergence': float(df['convergence_angle'].std()),
                'max_convergence': float(df['convergence_angle'].max()),
                'min_convergence': float(df['convergence_angle'].min()),
                'convergence_range': float(df['convergence_angle'].max() - df['convergence_angle'].min()),
                'mean_convergence_velocity': float(df['convergence_velocity'].mean()),
                'std_convergence_velocity': float(df['convergence_velocity'].std()),
            }
            
            # Calculate coefficient of variation
            if metrics['mean_convergence'] != 0:
                metrics['cv_convergence'] = metrics['std_convergence'] / metrics['mean_convergence']
            else:
                metrics['cv_convergence'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in calculate_convergence_metrics: {e}")
            raise ValueError(f"Error in calculate_convergence_metrics: {e}")

    def detect_nystagmus(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect nystagmus in eye tracking data.
        
        Args:
            df: Preprocessed DataFrame with eye tracking data
            
        Returns:
            Dictionary of nystagmus metrics
        """
        try:
            # Use the preprocessor's nystagmus detection
            _, nystagmus_metrics = self.preprocessor.detect_nystagmus(
                df, 
                velocity_threshold=self.THRESHOLDS['nystagmus']['velocity'],
                min_duration=10
            )
            
            return nystagmus_metrics
            
        except Exception as e:
            logger.error(f"Error in detect_nystagmus: {e}")
            raise ValueError(f"Error in detect_nystagmus: {e}")

    def extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features from the preprocessed data.
        
        Args:
            df: Preprocessed DataFrame with eye tracking data
            
        Returns:
            Dictionary of extracted features
        """
        try:
            features = {}
            
            # Calculate convergence metrics
            convergence_metrics = self.calculate_convergence_metrics(df)
            features.update(convergence_metrics)
            
            # Detect nystagmus
            nystagmus_metrics = self.detect_nystagmus(df)
            features.update(nystagmus_metrics)
            
            # Calculate pupil metrics (if available)
            if 'left_pupil_size' in df.columns and 'right_pupil_size' in df.columns:
                features['mean_pupil_size'] = float(np.mean([
                    df['left_pupil_size'].fillna(0).mean(),
                    df['right_pupil_size'].fillna(0).mean()
                ]))
                features['std_pupil_size'] = float(np.mean([
                    df['left_pupil_size'].fillna(0).std(),
                    df['right_pupil_size'].fillna(0).std()
                ]))
            else:
                # Default values if pupil data is not available
                features['mean_pupil_size'] = 3.5
                features['std_pupil_size'] = 0.5
            
            return features
            
        except Exception as e:
            logger.error(f"Error in extract_features: {e}")
            raise ValueError(f"Error in extract_features: {e}")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using prepared data.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        try:
            # Handle NaN values
            X_imputed = self.imputer.fit_transform(X)
            
            # Initialize and train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.model.fit(X_imputed, y)
            self.is_trained = True
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in train: {e}")
            raise ValueError(f"Error in train: {e}")

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using trained model.
        
        Args:
            df: Input DataFrame with eye tracking data
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Preprocess data and extract features
            df = this.preprocess(df)
            features = this.extract_features(df)
            
            # Convert features to array
            X = np.array([list(features.values())])
            
            # Handle NaN values
            X_imputed = this.imputer.transform(X)
            
            # Scale features
            X_scaled = this.scaler.transform(X_imputed)
            
            # Make prediction
            prediction = this.model.predict(X_scaled)[0]
            probability = this.model.predict_proba(X_scaled)[0]
            
            # Determine impairment factors
            factors = this.determine_impairment_factors(features)
            
            return {
                'prediction': bool(prediction),
                'probability': float(probability[1]),
                'features': features,
                'factors': factors
            }
            
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            raise ValueError(f"Error in predict: {e}")
    
    def determine_impairment_factors(self, features: Dict[str, float]) -> Dict[str, bool]:
        """
        Determine which factors indicate impairment.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            Dictionary of impairment factors
        """
        try:
            factors = {}
            
            # Check convergence factors
            factors['high_convergence'] = features['mean_convergence'] > self.THRESHOLDS['convergence']['impaired']
            factors['high_convergence_variability'] = features['std_convergence'] > self.THRESHOLDS['convergence']['impaired']
            factors['large_convergence_range'] = features['convergence_range'] > self.THRESHOLDS['convergence']['impaired'] * 2
            
            # Check nystagmus factors
            factors['nystagmus_present'] = (
                features.get('left_nystagmus_count', 0) > 0.1 or 
                features.get('right_nystagmus_count', 0) > 0.1
            )
            factors['high_nystagmus_velocity'] = (
                features.get('left_nystagmus_peak_velocity', 0) > self.THRESHOLDS['nystagmus']['velocity'] or
                features.get('right_nystagmus_peak_velocity', 0) > self.THRESHOLDS['nystagmus']['velocity']
            )
            
            # Check pupil factors
            factors['abnormal_pupil_size'] = features['mean_pupil_size'] > self.THRESHOLDS['pupil']['impaired_size']
            factors['high_pupil_variability'] = features['std_pupil_size'] > 1.0
            
            return factors
            
        except Exception as e:
            logger.error(f"Error in determine_impairment_factors: {e}")
            raise ValueError(f"Error in determine_impairment_factors: {e}") 