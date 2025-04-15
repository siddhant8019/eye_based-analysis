import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from typing import Dict, Any, List, Tuple
import ast
import re

class EyeTrackingPreprocessor:
    def __init__(self):
        self.required_columns = [
            'caruncle_2d', 'iris_2d', 'eye_region_details',
            'head_pose', 'lighting_details'
        ]
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate required columns and data quality."""
        return all(col in df.columns for col in self.required_columns)
    
    def calculate_gaze_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert coordinates to angles."""
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
            print("Sample of processed coordinates (in degrees):")
            print(df[['left_eye_angle_x', 'left_eye_angle_y', 'right_eye_angle_x', 'right_eye_angle_y']].head())
            
            return df
        except Exception as e:
            raise ValueError(f"Error calculating gaze angles: {e}")
    
    def calculate_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate eye movement velocities in degrees per second."""
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
            print("Sample of velocity data (degrees/second):")
            print(df[['left_velocity', 'right_velocity']].head())
            
            return df
        except Exception as e:
            raise ValueError(f"Error calculating velocities: {e}")
    
    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic features common to all models."""
        try:
            df = self.calculate_gaze_angles(df)
            df = self.calculate_velocities(df)
            return df
        except Exception as e:
            raise ValueError(f"Error extracting basic features: {e}") 