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

class NystagmusModel(BaseEyeTrackingModel):
    """Model for detecting Nystagmus impairment based on eye tracking data."""
    
    # Feature importance weights based on research
    FEATURE_WEIGHTS = {
        "nystagmus_features": 0.50,  # Most important according to research
        "velocity_features": 0.30,   # Second most important
        "stability_features": 0.20,  # Supporting features
    }

    # Thresholds based on research findings
    THRESHOLDS = {
        "nystagmus": {
            "velocity": 2.0,     # degrees/second
            "frequency": 3.0,    # Hz
            "amplitude": 1.0,    # degrees
            "min_duration": 10,  # frames
        },
        "velocity": {
            "normal": 30.0,     # degrees/second
            "impaired": 50.0,   # degrees/second
        },
        "stability": {
            "normal": 0.5,      # degrees
            "impaired": 1.0,    # degrees
        },
    }

    def __init__(self):
        """Initialize the model."""
        super().__init__("NystagmusModel")
        self.imputer = SimpleImputer(strategy='mean')
        logger.info("Initialized NystagmusModel")

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
                min_duration=self.THRESHOLDS['nystagmus']['min_duration']
            )
            
            return nystagmus_metrics
            
        except Exception as e:
            logger.error(f"Error in detect_nystagmus: {e}")
            raise ValueError(f"Error in detect_nystagmus: {e}")

    def calculate_velocity_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate velocity metrics from eye tracking data.
        
        Args:
            df: Preprocessed DataFrame with eye tracking data
            
        Returns:
            Dictionary of velocity metrics
        """
        try:
            metrics = {}
            
            for eye in ['left', 'right']:
                # Get velocity data
                velocity = df[f'{eye}_velocity'].fillna(0)
                
                # Apply smoothing
                window = min(5, len(velocity) - 1)
                if window > 3:
                    velocity = savgol_filter(velocity, window, 2)
                
                # Calculate metrics
                metrics[f'{eye}_mean_velocity'] = float(np.mean(velocity))
                metrics[f'{eye}_std_velocity'] = float(np.std(velocity))
                metrics[f'{eye}_max_velocity'] = float(np.max(velocity))
                metrics[f'{eye}_min_velocity'] = float(np.min(velocity))
                metrics[f'{eye}_velocity_range'] = float(np.max(velocity) - np.min(velocity))
                
                # Calculate acceleration
                acceleration = np.diff(velocity) / df['dt'].iloc[1:]
                metrics[f'{eye}_mean_acceleration'] = float(np.mean(acceleration))
                metrics[f'{eye}_std_acceleration'] = float(np.std(acceleration))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in calculate_velocity_metrics: {e}")
            raise ValueError(f"Error in calculate_velocity_metrics: {e}")

    def calculate_stability_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate stability metrics from eye tracking data.
        
        Args:
            df: Preprocessed DataFrame with eye tracking data
            
        Returns:
            Dictionary of stability metrics
        """
        try:
            metrics = {}
            
            for eye in ['left', 'right']:
                # Get position data
                position_x = df[f'{eye}_eye_angle_x'].fillna(0)
                position_y = df[f'{eye}_eye_angle_y'].fillna(0)
                
                # Calculate stability metrics
                metrics[f'{eye}_position_std_x'] = float(np.std(position_x))
                metrics[f'{eye}_position_std_y'] = float(np.std(position_y))
                metrics[f'{eye}_position_range_x'] = float(np.max(position_x) - np.min(position_x))
                metrics[f'{eye}_position_range_y'] = float(np.max(position_y) - np.min(position_y))
                
                # Calculate drift
                drift_x = np.polyfit(np.arange(len(position_x)), position_x, 1)[0]
                drift_y = np.polyfit(np.arange(len(position_y)), position_y, 1)[0]
                metrics[f'{eye}_drift_x'] = float(drift_x)
                metrics[f'{eye}_drift_y'] = float(drift_y)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in calculate_stability_metrics: {e}")
            raise ValueError(f"Error in calculate_stability_metrics: {e}")

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
            
            # Detect nystagmus
            nystagmus_metrics = self.detect_nystagmus(df)
            features.update(nystagmus_metrics)
            
            # Calculate velocity metrics
            velocity_metrics = self.calculate_velocity_metrics(df)
            features.update(velocity_metrics)
            
            # Calculate stability metrics
            stability_metrics = self.calculate_stability_metrics(df)
            features.update(stability_metrics)
            
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
            this.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            this.model.fit(X_imputed, y)
            this.is_trained = True
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
            if not this.is_trained:
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
            
            # Check nystagmus factors
            factors['nystagmus_present'] = (
                features.get('left_nystagmus_count', 0) > 0.1 or 
                features.get('right_nystagmus_count', 0) > 0.1
            )
            factors['high_nystagmus_velocity'] = (
                features.get('left_nystagmus_peak_velocity', 0) > this.THRESHOLDS['nystagmus']['velocity'] or
                features.get('right_nystagmus_peak_velocity', 0) > this.THRESHOLDS['nystagmus']['velocity']
            )
            factors['high_nystagmus_frequency'] = (
                features.get('left_nystagmus_frequency', 0) > this.THRESHOLDS['nystagmus']['frequency'] or
                features.get('right_nystagmus_frequency', 0) > this.THRESHOLDS['nystagmus']['frequency']
            )
            
            # Check velocity factors
            for eye in ['left', 'right']:
                factors[f'{eye}_high_velocity'] = features[f'{eye}_mean_velocity'] > this.THRESHOLDS['velocity']['impaired']
                factors[f'{eye}_high_velocity_variability'] = features[f'{eye}_std_velocity'] > this.THRESHOLDS['velocity']['impaired']
            
            # Check stability factors
            for eye in ['left', 'right']:
                factors[f'{eye}_high_position_variability'] = (
                    features[f'{eye}_position_std_x'] > this.THRESHOLDS['stability']['impaired'] or
                    features[f'{eye}_position_std_y'] > this.THRESHOLDS['stability']['impaired']
                )
                factors[f'{eye}_significant_drift'] = (
                    abs(features[f'{eye}_drift_x']) > this.THRESHOLDS['stability']['impaired'] or
                    abs(features[f'{eye}_drift_y']) > this.THRESHOLDS['stability']['impaired']
                )
            
            return factors
            
        except Exception as e:
            logger.error(f"Error in determine_impairment_factors: {e}")
            raise ValueError(f"Error in determine_impairment_factors: {e}") 