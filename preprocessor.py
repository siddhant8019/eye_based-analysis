import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Dict, Any, List, Tuple
import ast
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EyeTrackingPreprocessor:
    """Preprocessor for eye tracking data."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.required_columns = [
            'caruncle_2d', 'iris_2d', 'eye_region_details',
            'head_pose', 'lighting_details'
        ]
        logger.info("Initialized EyeTrackingPreprocessor")
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate required columns and data quality.
        
        Args:
            df: Input DataFrame with eye tracking data
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check if all required columns are present
        if not all(col in df.columns for col in self.required_columns):
            logger.warning(f"Missing required columns. Required: {self.required_columns}")
            return False
        
        # Check if data is empty
        if df.empty:
            logger.warning("DataFrame is empty")
            return False
        
        # Check for minimum number of rows
        if len(df) < 30:  # Minimum 1 second at 30 fps
            logger.warning(f"Insufficient data points: {len(df)} (minimum 30 required)")
            return False
        
        return True
    
    def calculate_gaze_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert coordinates to angles.
        
        Args:
            df: Input DataFrame with eye tracking data
            
        Returns:
            DataFrame with gaze angles added
        """
        try:
            # Extract coordinates from string format like '(215.0031, 192.4742, 8.8517)'
            def parse_coordinates(coord_str):
                try:
                    # Use regex to extract numbers from the string
                    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', coord_str)
                    if len(numbers) >= 2:
                        # Convert pixel coordinates to degrees (assuming 96 DPI and 50cm viewing distance)
                        x = float(numbers[0])
                        y = float(numbers[1])
                        # Convert to degrees using arctan
                        x_deg = np.degrees(np.arctan2(x - 320, 500))  # Assuming screen center at 320 pixels
                        y_deg = np.degrees(np.arctan2(y - 240, 500))  # Assuming screen center at 240 pixels
                        return [x_deg, y_deg, float(numbers[2]) if len(numbers) > 2 else 0.0]
                    return [0.0, 0.0, 0.0]
                except:
                    return [0.0, 0.0, 0.0]  # Default values if parsing fails
            
            # Parse caruncle coordinates
            caruncle_coords = df['caruncle_2d'].apply(parse_coordinates)
            df['left_eye_angle_x'] = caruncle_coords.apply(lambda x: x[0])
            df['left_eye_angle_y'] = caruncle_coords.apply(lambda x: x[1])
            
            # Parse iris coordinates
            iris_coords = df['iris_2d'].apply(parse_coordinates)
            df['right_eye_angle_x'] = iris_coords.apply(lambda x: x[0])
            df['right_eye_angle_y'] = iris_coords.apply(lambda x: x[1])
            
            # Add timestamp column (using index as time for now)
            df['time'] = df.index * 1/30  # Assuming 30 fps
            
            # Print sample of processed data for debugging
            logger.debug("Sample of processed coordinates (in degrees):")
            logger.debug(df[['left_eye_angle_x', 'left_eye_angle_y', 'right_eye_angle_x', 'right_eye_angle_y']].head())
            
            return df
        except Exception as e:
            logger.error(f"Error calculating gaze angles: {e}")
            raise ValueError(f"Error calculating gaze angles: {e}")
    
    def calculate_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate eye movement velocities in degrees per second.
        
        Args:
            df: Input DataFrame with eye tracking data
            
        Returns:
            DataFrame with velocities added
        """
        try:
            # Calculate time differences
            df['dt'] = df['time'].diff().fillna(1/30)  # Fill first value with frame time
            
            # Calculate velocities for both eyes
            for eye in ['left', 'right']:
                for coord in ['x', 'y']:
                    col = f'{eye}_eye_angle_{coord}'
                    vel_col = f'{col}_velocity'
                    # Calculate velocity in degrees per second
                    df[vel_col] = df[col].diff() / df['dt']
                    
                    # Apply Savitzky-Golay filter for smoothing
                    window = min(5, len(df) - 1)
                    if window > 3:
                        df[vel_col] = savgol_filter(df[vel_col].fillna(0), window, 2)
            
            # Calculate magnitude velocities (in degrees per second)
            for eye in ['left', 'right']:
                df[f'{eye}_velocity'] = np.sqrt(
                    df[f'{eye}_eye_angle_x_velocity']**2 + 
                    df[f'{eye}_eye_angle_y_velocity']**2
                )
            
            # Print sample of velocity data for debugging
            logger.debug("Sample of velocity data (degrees/second):")
            logger.debug(df[['left_velocity', 'right_velocity']].head())
            
            return df
        except Exception as e:
            logger.error(f"Error calculating velocities: {e}")
            raise ValueError(f"Error calculating velocities: {e}")
    
    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic features common to all models.
        
        Args:
            df: Input DataFrame with eye tracking data
            
        Returns:
            DataFrame with basic features added
        """
        try:
            df = self.calculate_gaze_angles(df)
            df = self.calculate_velocities(df)
            return df
        except Exception as e:
            logger.error(f"Error extracting basic features: {e}")
            raise ValueError(f"Error extracting basic features: {e}")
    
    def detect_fixations_and_saccades(self, df: pd.DataFrame, 
                                     velocity_threshold: float = 30.0,
                                     min_duration: int = 3) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Detect fixations and saccades in eye tracking data.
        
        Args:
            df: Input DataFrame with eye tracking data
            velocity_threshold: Threshold for saccade detection in degrees/second
            min_duration: Minimum duration for a saccade in frames
            
        Returns:
            Tuple of (DataFrame with fixations and saccades marked, metrics dictionary)
        """
        try:
            from scipy.ndimage import binary_opening
            
            metrics = {}
            
            for eye in ['left', 'right']:
                velocity = df[f'{eye}_velocity'].fillna(0)  # Fill NaN with 0
                
                # Detect saccades using velocity threshold
                is_saccade = velocity > velocity_threshold
                
                # Apply minimum duration constraint
                is_saccade = binary_opening(is_saccade, structure=np.ones(min_duration))
                
                # Calculate saccade metrics
                saccade_count = np.sum(is_saccade) / len(velocity)  # Normalize by sequence length
                saccade_peak_velocity = np.max(velocity[is_saccade]) if np.any(is_saccade) else 0
                
                # Detect fixations (periods of low velocity)
                fixation_threshold = velocity_threshold / 3  # Lower threshold for fixations
                is_fixation = velocity < fixation_threshold
                
                # Apply minimum duration constraint for fixations
                min_fixation_duration = min_duration + 2  # Slightly longer for fixations
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
            logger.debug("Saccade and Fixation Detection Results:")
            logger.debug(f"Number of frames: {len(df)}")
            logger.debug(f"Metrics: {metrics}")
            
            return df, metrics
            
        except Exception as e:
            logger.error(f"Error in detect_fixations_and_saccades: {e}")
            raise ValueError(f"Error in detect_fixations_and_saccades: {e}")
    
    def calculate_convergence_angle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate convergence angle between left and right eye gaze vectors.
        
        Args:
            df: Input DataFrame with eye tracking data
            
        Returns:
            DataFrame with convergence angle added
        """
        try:
            # Calculate convergence angle (angle between left and right eye gaze vectors)
            # This is a simplified calculation - in a real implementation, you would use
            # the actual 3D vectors and calculate the angle between them
            
            # For simplicity, we'll use the difference in horizontal angles
            df['convergence_angle'] = df['right_eye_angle_x'] - df['left_eye_angle_x']
            
            # Calculate convergence velocity
            df['convergence_velocity'] = df['convergence_angle'].diff() / df['dt']
            
            # Apply smoothing
            window = min(5, len(df) - 1)
            if window > 3:
                df['convergence_velocity'] = savgol_filter(df['convergence_velocity'].fillna(0), window, 2)
            
            return df
        except Exception as e:
            logger.error(f"Error calculating convergence angle: {e}")
            raise ValueError(f"Error calculating convergence angle: {e}")
    
    def detect_nystagmus(self, df: pd.DataFrame, 
                         velocity_threshold: float = 2.5,
                         min_duration: int = 10) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Detect nystagmus in eye tracking data.
        
        Args:
            df: Input DataFrame with eye tracking data
            velocity_threshold: Threshold for nystagmus detection in degrees/second
            min_duration: Minimum duration for nystagmus in frames
            
        Returns:
            Tuple of (DataFrame with nystagmus marked, metrics dictionary)
        """
        try:
            from scipy.ndimage import binary_opening
            
            metrics = {}
            
            for eye in ['left', 'right']:
                # Use horizontal velocity for nystagmus detection
                velocity = df[f'{eye}_eye_angle_x_velocity'].fillna(0)
                
                # Detect nystagmus using velocity threshold
                is_nystagmus = np.abs(velocity) > velocity_threshold
                
                # Apply minimum duration constraint
                is_nystagmus = binary_opening(is_nystagmus, structure=np.ones(min_duration))
                
                # Calculate nystagmus metrics
                nystagmus_count = np.sum(is_nystagmus) / len(velocity)  # Normalize by sequence length
                nystagmus_peak_velocity = np.max(np.abs(velocity[is_nystagmus])) if np.any(is_nystagmus) else 0
                
                # Calculate nystagmus frequency (simplified)
                if np.any(is_nystagmus):
                    # Count zero crossings as a proxy for frequency
                    zero_crossings = np.sum(np.diff(np.signbit(velocity[is_nystagmus])))
                    nystagmus_frequency = zero_crossings / (2 * np.sum(is_nystagmus) / 30)  # Assuming 30 fps
                else:
                    nystagmus_frequency = 0.0
                
                # Store metrics
                metrics.update({
                    f'{eye}_nystagmus_count': float(nystagmus_count),
                    f'{eye}_nystagmus_peak_velocity': float(nystagmus_peak_velocity),
                    f'{eye}_nystagmus_frequency': float(nystagmus_frequency)
                })
                
                # Add detection results to DataFrame
                df[f'{eye}_is_nystagmus'] = is_nystagmus
            
            # Print debug information
            logger.debug("Nystagmus Detection Results:")
            logger.debug(f"Number of frames: {len(df)}")
            logger.debug(f"Metrics: {metrics}")
            
            return df, metrics
            
        except Exception as e:
            logger.error(f"Error in detect_nystagmus: {e}")
            raise ValueError(f"Error in detect_nystagmus: {e}") 