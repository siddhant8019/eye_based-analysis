import numpy as np
import pandas as pd
from scipy.ndimage import binary_opening
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Tuple
from base_model import BaseEyeTrackingModel

class SmoothPursuitModel(BaseEyeTrackingModel):
    """Model for detecting Lack of Smooth Pursuit (LoSP) based on eye tracking data."""
    
    # Feature importance weights based on research
    FEATURE_WEIGHTS = {
        "fixation_features": 0.25,  # Most important according to research
        "saccade_features": 0.35,  # Second most important
        "eye_closure": 0.40,  # Supporting features
    }

    # Velocity thresholds based on research findings (in degrees/second)
    VELOCITY_THRESHOLDS = {
        "fixation": {
            "normal": 15.0,  # deg/s during stable fixation
            "impaired": 30.0,  # deg/s indicating potential impairment
        },
        "saccade": {
            "min": 30.0,  # deg/s minimum for saccade detection
            "normal_peak": 300.0,  # deg/s normal peak velocity
            "impaired_peak": 200.0,  # deg/s indicating potential impairment
        },
    }

    def __init__(self):
        super().__init__()
        self.imputer = SimpleImputer(strategy='mean')

    def detect_fixations_and_saccades(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Detect fixations and saccades in the eye tracking data."""
        try:
            metrics = {}
            
            for eye in ['left', 'right']:
                velocity = df[f'{eye}_velocity'].fillna(0)  # Fill NaN with 0
                
                # Detect saccades using velocity threshold
                saccade_threshold = self.VELOCITY_THRESHOLDS['saccade']['min']
                is_saccade = velocity > saccade_threshold
                
                # Apply minimum duration constraint (3 frames at 30 Hz = 100ms)
                min_saccade_duration = 3
                is_saccade = binary_opening(is_saccade, structure=np.ones(min_saccade_duration))
                
                # Calculate saccade metrics
                saccade_count = np.sum(is_saccade) / len(velocity)  # Normalize by sequence length
                saccade_peak_velocity = np.max(velocity[is_saccade]) if np.any(is_saccade) else 0
                
                # Detect fixations (periods of low velocity)
                fixation_threshold = self.VELOCITY_THRESHOLDS['fixation']['normal']
                is_fixation = velocity < fixation_threshold
                
                # Apply minimum duration constraint for fixations (5 frames at 30 Hz ≈ 167ms)
                min_fixation_duration = 5
                is_fixation = binary_opening(is_fixation, structure=np.ones(min_fixation_duration))
                
                # Calculate fixation stability
                fixation_velocity = velocity[is_fixation]
                if len(fixation_velocity) > 0 and np.mean(fixation_velocity) > 0:
                    fixation_stability = 1.0 - (np.std(fixation_velocity) / np.mean(fixation_velocity))
                    fixation_stability = max(0.0, min(1.0, fixation_stability))  # Clip to [0, 1]
                else:
                    fixation_stability = 0.0
                
                # Store metrics
                metrics.update({
                    f'{eye}_saccade_count': float(saccade_count),
                    f'{eye}_saccade_peak_velocity': float(saccade_peak_velocity),
                    f'{eye}_fixation_stability': float(fixation_stability)
                })
                
                # Add detection results to DataFrame
                df[f'{eye}_is_saccade'] = is_saccade
                df[f'{eye}_is_fixation'] = is_fixation
            
            # Print debug information
            print("\nSaccade and Fixation Detection Results:")
            print(f"Number of frames: {len(df)}")
            print("Metrics:", metrics)
            
            return df, metrics
            
        except Exception as e:
            raise ValueError(f"Error in detect_fixations_and_saccades: {e}")

    def extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract features from the preprocessed data."""
        try:
            features = {}
            
            # Calculate average velocities
            features['eye_velocity'] = float(np.mean([
                df['left_velocity'].fillna(0).mean(),
                df['right_velocity'].fillna(0).mean()
            ]))
            
            # Calculate velocity stability
            features['velocity_stability'] = float(np.mean([
                df['left_velocity'].fillna(0).std(),
                df['right_velocity'].fillna(0).std()
            ]))
            
            # Add fixation and saccade metrics
            _, event_metrics = self.detect_fixations_and_saccades(df)
            features.update(event_metrics)
            
            return features
            
        except Exception as e:
            raise ValueError(f"Error in extract_features: {e}")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model using prepared data."""
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
            
        except Exception as e:
            raise ValueError(f"Error in train: {e}")

    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using trained model."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Preprocess data and extract features
            df = self.preprocess(df)
            features = self.extract_features(df)
            
            # Convert features to array
            X = np.array([list(features.values())])
            
            # Handle NaN values
            X_imputed = self.imputer.transform(X)
            
            # Scale features
            X_scaled = self.scaler.transform(X_imputed)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
            
            return {
                'prediction': bool(prediction),
                'probability': float(probability[1]),
                'features': features
            }
            
        except Exception as e:
            raise ValueError(f"Error in predict: {e}") 