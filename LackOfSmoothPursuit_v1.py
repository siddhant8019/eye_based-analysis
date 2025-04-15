import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import binary_opening
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from typing import Dict, Any, List, Tuple
import warnings
import logging

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define feature columns
FEATURE_COLUMNS = [
    "eye_velocity",
    "velocity_stability",
    "left_fixation_stability",
    "left_saccade_count",
    "left_saccade_peak_velocity",
    "right_fixation_stability",
    "right_saccade_count",
    "right_saccade_peak_velocity",
]

class LackOfSmoothPursuit:
    """Model for detecting Lack of Smooth Pursuit (LoSP) based on eye tracking data.
    
    This model analyzes eye movements to detect impairment indicators which are key 
    indicators of lack of smooth pursuit in standardized field sobriety tests.
    """

    # Feature importance weights based on research
    FEATURE_WEIGHTS = {
        "fixation_features": 0.25,  # Most important according to research
        "saccade_features": 0.35,  # Second most important
        "eye_closure": 0.40,  # Supporting features
    }

    # Velocity thresholds based on research findings
    VELOCITY_THRESHOLDS = {
        "fixation": {
            "normal": 0.1,  # deg/s during stable fixation
            "impaired": 0.3,  # deg/s indicating potential impairment
        },
        "saccade": {
            "min": 30.0,  # deg/s minimum for saccade detection
            "normal_peak": 300.0,  # deg/s normal peak velocity
            "impaired_peak": 200.0,  # deg/s indicating potential impairment
        },
    }

    # Time windows for analysis (in seconds)
    TIME_WINDOWS = {
        "optimal": 60,  # Optimal window size from research
        "minimum": 20,  # Minimum reliable window
        "fixation": 0.25,  # Typical fixation duration
        "saccade": 0.05,  # Typical saccade duration
    }

    def __init__(self):
        """Initialize the model."""
        self.scaler = None
        self.model = None
        self.is_trained = False
        self.params = self._get_default_params()

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default model parameters."""
        return {
            "velocity_threshold": 1.2,
            "smoothness_threshold": 0.6,
            "max_time_diff": 0.05,
            "min_frames": 30,  # Minimum number of valid frames required
        }

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input DataFrame containing eye tracking data.
        
        Args:
            df (pd.DataFrame): Raw DataFrame with eye tracking data
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame with calculated features
        """
        try:
            # Ensure required columns exist
            required_columns = [
                'time', 'left_eye_angle_x', 'left_eye_angle_y',
                'right_eye_angle_x', 'right_eye_angle_y'
            ]
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in input data")

            # Calculate time differences
            df['dt'] = df['time'].diff() / 1e7  # Convert to seconds
            
            # Remove rows with invalid time differences
            df = df[df['dt'] > 0].copy()
            
            # Calculate velocities for both eyes
            for eye in ['left', 'right']:
                for coord in ['x', 'y']:
                    col = f'{eye}_eye_angle_{coord}'
                    vel_col = f'{col}_velocity'
                    df[vel_col] = df[col].diff() / df['dt']
                    
                    # Apply Savitzky-Golay filter for smoothing
                    window = min(5, len(df) - 1)
                    if window > 3:
                        df[vel_col] = savgol_filter(df[vel_col].fillna(0), window, 2)
            
            # Calculate magnitude velocities
            for eye in ['left', 'right']:
                df[f'{eye}_velocity'] = np.sqrt(
                    df[f'{eye}_eye_angle_x_velocity']**2 + 
                    df[f'{eye}_eye_angle_y_velocity']**2
                ) * (180 / np.pi)  # Convert to degrees/second
            
            return df
            
        except Exception as e:
            logger.error(f"Error in preprocess_data: {e}")
            raise

    def detect_fixations_and_saccades(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Detect fixations and saccades in the eye tracking data.
        
        Args:
            df (pd.DataFrame): Preprocessed DataFrame with velocity calculations
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, float]]: DataFrame with detected events and metrics
        """
        try:
            metrics = {}
            
            for eye in ['left', 'right']:
                velocity = df[f'{eye}_velocity']
                
                # Detect saccades using adaptive thresholding
                baseline_velocity = np.median(velocity)
                velocity_std = np.std(velocity)
                saccade_threshold = max(
                    self.VELOCITY_THRESHOLDS['saccade']['min'],
                    baseline_velocity + 2 * velocity_std
                )
                
                # Detect saccades
                is_saccade = velocity > saccade_threshold
                
                # Apply minimum duration constraint
                min_saccade_duration = 3  # minimum frames
                is_saccade = binary_opening(is_saccade, structure=np.ones(min_saccade_duration))
                
                # Calculate saccade metrics
                saccade_count = np.sum(is_saccade)
                saccade_peak_velocity = np.max(velocity[is_saccade]) if np.any(is_saccade) else 0
                
                # Detect fixations (periods of low velocity)
                fixation_threshold = self.VELOCITY_THRESHOLDS['fixation']['normal']
                is_fixation = velocity < fixation_threshold
                
                # Calculate fixation stability
                fixation_velocity = velocity[is_fixation]
                fixation_stability = 1.0 - (np.std(fixation_velocity) / np.mean(fixation_velocity)) if len(fixation_velocity) > 0 else 0
                
                # Store metrics
                metrics.update({
                    f'{eye}_saccade_count': float(saccade_count),
                    f'{eye}_saccade_peak_velocity': float(saccade_peak_velocity),
                    f'{eye}_fixation_stability': float(fixation_stability)
                })
                
                # Add detection results to DataFrame
                df[f'{eye}_is_saccade'] = is_saccade
                df[f'{eye}_is_fixation'] = is_fixation
            
            return df, metrics
            
        except Exception as e:
            logger.error(f"Error in detect_fixations_and_saccades: {e}")
            raise

    def extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract features from the preprocessed data.
        
        Args:
            df (pd.DataFrame): Preprocessed DataFrame with detected events
            
        Returns:
            Dict[str, float]: Dictionary of extracted features
        """
        try:
            features = {}
            
            # Calculate average velocities
            features['eye_velocity'] = float(np.mean([
                df['left_velocity'].mean(),
                df['right_velocity'].mean()
            ]))
            
            # Calculate velocity stability
            features['velocity_stability'] = float(np.mean([
                df['left_velocity'].std(),
                df['right_velocity'].std()
            ]))
            
            # Add fixation and saccade metrics
            _, event_metrics = self.detect_fixations_and_saccades(df)
            features.update(event_metrics)
            
            return features
            
        except Exception as e:
            logger.error(f"Error in extract_features: {e}")
            raise

    def prepare_training_data(self, data: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training.
        
        Args:
            data (List[Dict[str, float]]): List of feature dictionaries
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features array and labels array
        """
        try:
            # Convert list of dictionaries to DataFrame
            df = pd.DataFrame(data)
            
            # Separate features and labels
            X = df.drop('label', axis=1)
            y = df['label']
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y.values
            
        except Exception as e:
            logger.error(f"Error in prepare_training_data: {e}")
            raise

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model using prepared data.
        
        Args:
            X (np.ndarray): Scaled feature array
            y (np.ndarray): Label array
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Initialize and train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Log training metrics
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            logger.info(f"Training score: {train_score:.3f}")
            logger.info(f"Test score: {test_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error in train: {e}")
            raise

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Make predictions using trained model.
        
        Args:
            features (Dict[str, float]): Dictionary of features
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Convert features to array
            X = np.array([list(features.values())])
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            probability = self.model.predict_proba(X_scaled)[0]
            
            return {
                'prediction': bool(prediction),
                'probability': float(probability[1]),
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            raise
