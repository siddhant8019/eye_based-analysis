#!/usr/bin/env python3
import os
import sys
from typing import Any, Dict
import importlib
import numpy as np
import pandas as pd
import warnings
from enum import Enum
from scipy.signal import savgol_filter, find_peaks
import duckdb
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "pylib/src"))

# Set default database path in the current directory
default_db_path = "gaize_v6.duckdb"

from ddgaize.detection.core.AbstractModel import AbstractModel
from ddgaize.detection.core.DataContext import DataContext
from ddgaize.detection.core import DataAccessor
from ddgaize.detection.core.ModelRegistry import ModelRegistry, ModelVersion
from ddgaize.util import logging_utils

importlib.reload(DataAccessor)

log = logging_utils.get_logger(__name__)

FEATURE_COLUMNS = [
    "mean_convergence",
    "std_convergence",
    "cv_convergence",
    "max_convergence",
    "min_convergence",
    "convergence_range",
]


class CannabisContext(DataContext):
    """Context for Cannabis impairment analysis."""

    class Columns(Enum):
        # Raw eye position data from database
        TIME = "time"
        LEFT_EYE_GAZE_X = "leftEye_gazeRay.x"
        LEFT_EYE_GAZE_Y = "leftEye_gazeRay.y"
        RIGHT_EYE_GAZE_X = "rightEye_gazeRay.x"
        RIGHT_EYE_GAZE_Y = "rightEye_gazeRay.y"

        def __str__(self):
            return str(self.value)

        @staticmethod
        def list():
            return list(map(lambda c: c.value, CannabisContext.Columns))

    def __init__(
        self, dataAccessor: DataAccessor, params: Dict[str, Any], isTrainingData: bool
    ):
        super().__init__(dataAccessor, params, isTrainingData)
        self.frame_data = None
        self.processed_data = None
        self.impairment_labels = dataAccessor.get_impairment_labels()
        self.conn = dataAccessor.dbEngine


class Cannabis(AbstractModel):
    """Model for detecting Cannabis impairment based on eye tracking data.
    This model analyzes eye movements to detect impairment indicators
    which are key indicators of cannabis impairment in standardized field sobriety tests.
    """

    def __init__(self):
        """Initialize the model."""
        try:
            super().__init__()
            self.context = None
            self.params = self._get_default_params()
            self.scaler = None
            self.clf_model = None
            self.is_trained = False

            # Updated thresholds based on data analysis
            self.DETECTION_THRESHOLDS = {
                "mean_convergence": 70.0,
                "std_convergence": 65.0,
                "cv_convergence": 1.15,
                "convergence_range": 150.0,
                "peak_height": 100.0,  # Threshold for peak height
                "peak_width": 2.0,  # Threshold for peak width in seconds
                "peak_variance": 20.0,  # Threshold for variance around peaks
            }

            # Peak timing windows for HGN test (in seconds)
            self.PEAK_WINDOWS = [
                (2, 7),  # Left Peripheral
                (9, 11),  # Center to Right
                (16, 18),  # Right to Center
                (20, 25),  # Left Peripheral
                (27, 29),  # Center to Right
                (34, 35),  # Right Peripheral
            ]

        except Exception as e:
            log.error(f"Initialization error: {e}")
            raise

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default model parameters.
        Returns:
            Dict[str, Any]: A dictionary of default model parameters.
        """
        return {
            "resampling_frequency": 0.01,
            "max_time": 35,
            "smoothing_window": 5,
            "z_threshold": 3.0,
        }

    def _getTrainingParams(self) -> Dict[str, Any]:
        """Get default training parameters.
        Returns:
            Dict[str, Any]: A dictionary of default training parameters.
        """
        return {
            "test_split_ratio": 0.2,
            "random_state": 42,
        }

    def execute(
        self,
        dataAccessor: DataAccessor,
        modelRegistry: ModelRegistry,
        mode: ModelVersion = ModelVersion.PRODUCTION,
    ) -> Dict[str, Any]:
        """Execute model prediction.
        Args:
            dataAccessor (DataAccessor): DataAccessor instance to retrieve data.
            modelRegistry (ModelRegistry): ModelRegistry instance for managing models.
            mode (ModelVersion, optional): The version of the model to use. Defaults to ModelVersion.PRODUCTION.
        Returns:
            Dict[str, Any]: A dictionary containing prediction results or error information.
        """
        try:
            # Pass the dataAccessor to _onExecute
            return self._onExecute(dataAccessor)
        except Exception as e:
            log.error(f"Error in execute: {e}")
            return {"error": str(e)}

    def _onExecute(self, dataAccessor: DataAccessor) -> Dict[str, Any]:
        """Execute model prediction for Cannabis analysis.
        Args:
            dataAccessor (DataAccessor): DataAccessor instance to retrieve data.
        Returns:
            Dict[str, Any]: A dictionary containing Cannabis analysis results.
        """
        try:
            # Use test_json_predictions with the DataAccessor
            json_results = self.test_json_predictions(dataAccessor)

            # Return results
            results = {
                "predictions": json_results,
            }
            return results

        except Exception as e:
            log.error(f"Error in _onExecute: {e}")
            return {"error": f"Model execution failed: {str(e)}"}

    def angle_between_vectors(self, left_vec, right_vec, in_degrees=True):
        """
        Computes the angle between two vectors left_vec and right_vec.
        Both expected to be numpy arrays of shape (N, 3) or (N, 2).
        If in_degrees=True, returns the angle in degrees, else radians.
        """
        # Dot products for each row
        dot_vals = (left_vec * right_vec).sum(axis=1)
        # Magnitudes
        mag_left = np.sqrt((left_vec**2).sum(axis=1))
        mag_right = np.sqrt((right_vec**2).sum(axis=1))

        # Avoid zero-division
        denom = mag_left * mag_right

        # Clip the ratio to [-1, 1] for numerical stability with arccos
        cos_vals = dot_vals / np.where(denom == 0, np.nan, denom)
        cos_vals = np.clip(cos_vals, -1.0, 1.0)

        angles = np.arccos(cos_vals)
        if in_degrees:
            angles = np.degrees(angles)
        return angles

    def compute_convergence_angle(self, left_points, right_points, timestamps):
        """Compute convergence angle between left and right eye gaze vectors.
        Args:
            left_points (list): List of (x,y) tuples for left eye gaze vectors
            right_points (list): List of (x,y) tuples for right eye gaze vectors
            timestamps (list): List of timestamps
        Returns:
            pd.DataFrame: DataFrame with timestamps and convergence angles
        """
        try:
            # Create arrays for calculations
            left_vec = np.array(left_points)
            right_vec = np.array(right_points)

            # Calculate angles
            angles = self.angle_between_vectors(left_vec, right_vec, in_degrees=True)

            # Create result DataFrame
            result_df = pd.DataFrame(
                {
                    "time_s": np.array(timestamps) / 1e7,  # Convert to seconds
                    "convergence_angle": angles,
                }
            )

            return result_df

        except Exception as e:
            log.error(f"Error computing convergence angle: {e}")
            return pd.DataFrame(columns=["time_s", "convergence_angle"])

    def resample_data(self, df, freq=0.01):
        """Resample convergence angle data to uniform time steps.
        Args:
            df (pd.DataFrame): DataFrame with time_s and convergence_angle columns
            freq (float): Resampling frequency in seconds
        Returns:
            pd.DataFrame: Resampled DataFrame
        """
        try:
            # Sort data by time
            df = df.sort_values(by="time_s")

            # Create uniform time series
            t_min = df["time_s"].min()
            t_max = df["time_s"].max()
            new_time = np.arange(t_min, t_max, freq)

            # Interpolate convergence angle
            interpolated = pd.DataFrame(
                {
                    "time_s": new_time,
                    "convergence_angle": np.interp(
                        new_time, df["time_s"], df["convergence_angle"]
                    ),
                }
            )

            return interpolated

        except Exception as e:
            log.error(f"Error resampling data: {e}")
            return df

    def smooth_data(self, df, metric="convergence_angle"):
        """Apply Savitzky-Golay filter to smooth data.

        Args:
            df (pd.DataFrame): DataFrame with time_s and metric columns
            metric (str): Column name to smooth

        Returns:
            pd.DataFrame: DataFrame with smoothed data
        """
        try:
            # Apply Savitzky-Golay filter to smooth data
            window_length = min(101, len(df) - 2)  # Larger window
            window_length = (
                window_length if window_length % 2 == 1 else window_length - 1
            )
            if window_length > 3:
                df[metric] = savgol_filter(
                    df[metric],
                    window_length=window_length,
                    polyorder=2,  # Lower polynomial order
                )
            return df
        except Exception as e:
            log.warning(f"Could not apply smoothing filter: {e}")
            return df

    def remove_outliers(self, df, metric="convergence_angle", z_thresh=3.0):
        """Remove outliers from data using z-score method.

        Args:
            df (pd.DataFrame): DataFrame with metric column
            metric (str): Column name to process
            z_thresh (float): Z-score threshold for outlier detection

        Returns:
            pd.DataFrame: DataFrame with outliers removed
        """
        try:
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            df[f"{metric}_no_outliers"] = df[metric].where(
                abs(df[metric] - mean_val) <= z_thresh * std_val
            )
            return df
        except Exception as e:
            log.error(f"Error removing outliers: {e}")
            return df

    def extract_features(self, df, metric="convergence_angle_no_outliers"):
        """Extract features from the data.

        Args:
            df (pd.DataFrame): DataFrame with metric column
            metric (str): Column name to process

        Returns:
            dict: Dictionary with extracted features
        """
        try:
            # Calculate basic statistics
            features = {
                "mean_convergence": df[metric].mean(),
                "std_convergence": df[metric].std(),
                "max_convergence": df[metric].max(),
                "min_convergence": df[metric].min(),
                "convergence_range": df[metric].max() - df[metric].min(),
            }

            # Calculate coefficient of variation
            if features["mean_convergence"] != 0:
                features["cv_convergence"] = (
                    features["std_convergence"] / features["mean_convergence"]
                )
            else:
                features["cv_convergence"] = 0

            return features
        except Exception as e:
            log.error(f"Error extracting features: {e}")
            return {}

    def normalize_features(self, features):
        """Normalize features using StandardScaler.

        Args:
            features (dict): Dictionary with features

        Returns:
            dict: Dictionary with normalized features
        """
        try:
            # Convert features to DataFrame
            features_df = pd.DataFrame([features])

            # Select numeric columns
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns

            # Apply StandardScaler
            scaler = StandardScaler()
            features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])

            # Convert back to dictionary
            normalized_features = features_df.iloc[0].to_dict()

            return normalized_features
        except Exception as e:
            log.error(f"Error normalizing features: {e}")
            return features

    def add_convergence_angle(self, left_points, right_points, timestamps):
        """
        Add convergence angle to the analysis by calculating the angle between left and right eye gaze vectors.

        Args:
            left_points (np.ndarray): Array of left eye gaze points
            right_points (np.ndarray): Array of right eye gaze points
            timestamps (np.ndarray): Array of timestamps

        Returns:
            pd.DataFrame: DataFrame containing timestamps and convergence angles
        """
        try:
            # Calculate convergence angles using the existing angle_between_vectors function
            convergence_angles = self.angle_between_vectors(left_points, right_points)

            # Create DataFrame with timestamps and convergence angles
            conv_df = pd.DataFrame(
                {"time": timestamps, "convergence_angle": convergence_angles}
            )

            return conv_df

        except Exception as e:
            log.error(f"Error calculating convergence angles: {e}")
            return pd.DataFrame(
                {"time": timestamps, "convergence_angle": np.zeros_like(timestamps)}
            )

    def analyze_peaks(self, convergence_data):
        """Analyze peaks in convergence angle data.

        Args:
            convergence_data (pd.DataFrame): DataFrame with time_s and convergence_angle columns

        Returns:
            dict: Dictionary containing peak analysis results
        """
        try:
            # Find peaks in the convergence angle data
            peaks, properties = find_peaks(
                convergence_data["convergence_angle"],
                height=50,  # Minimum peak height
                distance=50,  # Minimum samples between peaks
                width=10,  # Minimum peak width in samples
            )

            peak_features = {
                "num_peaks": len(peaks),
                "peak_heights": [],
                "peak_widths": [],
                "peak_times": [],
                "peak_variances": [],
                "expected_peaks_found": 0,
            }

            # Analyze each peak
            for i, peak_idx in enumerate(peaks):
                peak_time = convergence_data["time_s"].iloc[peak_idx]
                peak_height = properties["peak_heights"][i]
                peak_width = properties["widths"][i] * np.mean(
                    np.diff(convergence_data["time_s"])
                )

                # Calculate variance around the peak
                window_start = max(0, peak_idx - 10)
                window_end = min(len(convergence_data), peak_idx + 10)
                peak_variance = (
                    convergence_data["convergence_angle"]
                    .iloc[window_start:window_end]
                    .std()
                )

                peak_features["peak_heights"].append(peak_height)
                peak_features["peak_widths"].append(peak_width)
                peak_features["peak_times"].append(peak_time)
                peak_features["peak_variances"].append(peak_variance)

                # Check if peak falls within expected windows
                for start, end in self.PEAK_WINDOWS:
                    if start <= peak_time <= end:
                        peak_features["expected_peaks_found"] += 1
                        break

            # Calculate additional metrics
            if peak_features["peak_heights"]:
                peak_features.update(
                    {
                        "mean_peak_height": np.mean(peak_features["peak_heights"]),
                        "std_peak_height": np.std(peak_features["peak_heights"]),
                        "mean_peak_width": np.mean(peak_features["peak_widths"]),
                        "mean_peak_variance": np.mean(peak_features["peak_variances"]),
                        "peak_timing_score": peak_features["expected_peaks_found"]
                        / len(self.PEAK_WINDOWS),
                    }
                )
            else:
                peak_features.update(
                    {
                        "mean_peak_height": 0,
                        "std_peak_height": 0,
                        "mean_peak_width": 0,
                        "mean_peak_variance": 0,
                        "peak_timing_score": 0,
                    }
                )

            return peak_features

        except Exception as e:
            log.error(f"Error analyzing peaks: {e}")
            return None

    def determine_impairment_status(self, features):
        """Determine impairment status based on Cannabis metrics and peak analysis.

        Args:
            features (dict): Dictionary with Cannabis features and peak analysis

        Returns:
            tuple: (is_impaired, factors)
        """
        try:
            # Check basic convergence factors
            factors = {
                "High Mean Convergence": bool(
                    features["mean_convergence"]
                    > self.DETECTION_THRESHOLDS["mean_convergence"]
                ),
                "High Variability": bool(
                    features["std_convergence"]
                    > self.DETECTION_THRESHOLDS["std_convergence"]
                ),
                "High Coefficient of Variation": bool(
                    features["cv_convergence"]
                    > self.DETECTION_THRESHOLDS["cv_convergence"]
                ),
                "Large Convergence Range": bool(
                    features["convergence_range"]
                    > self.DETECTION_THRESHOLDS["convergence_range"]
                ),
            }

            # Add peak-related factors if available
            if "mean_peak_height" in features:
                factors.update(
                    {
                        "High Peak Heights": bool(
                            features["mean_peak_height"]
                            > self.DETECTION_THRESHOLDS["peak_height"]
                        ),
                        "Wide Peak Variance": bool(
                            features["mean_peak_variance"]
                            > self.DETECTION_THRESHOLDS["peak_variance"]
                        ),
                        "Poor Peak Timing": bool(
                            features.get("peak_timing_score", 1.0) < 0.8
                        ),
                    }
                )

            # Count how many factors indicate impairment
            impairment_count = sum(factors.values())

            # Determine impairment status
            is_impaired = impairment_count > 1

            return is_impaired, factors

        except Exception as e:
            log.error(f"Error determining impairment status: {e}")
            return False, {}

    def test_json_predictions(self, dataAccessor: DataAccessor) -> Dict[str, Any]:
        """Generate Cannabis predictions from JSON data.

        Args:
            dataAccessor (DataAccessor): DataAccessor instance to retrieve data.

        Returns:
            Dict[str, Any]: Dictionary with Cannabis prediction results
        """
        try:
            # Extract frames from JSON data
            data = dataAccessor.json
            frames = []

            for frame in data.get("data", []):
                if not isinstance(frame, dict):
                    continue

                left_eye = frame.get("leftEye", {})
                right_eye = frame.get("rightEye", {})
                left_gaze = left_eye.get("gazeRay", {})
                right_gaze = right_eye.get("gazeRay", {})

                if all(
                    [
                        left_gaze.get("x") is not None,
                        left_gaze.get("y") is not None,
                        right_gaze.get("x") is not None,
                        right_gaze.get("y") is not None,
                        not left_eye.get("isBlink", True),
                        not right_eye.get("isBlink", True),
                    ]
                ):
                    frame_data = {
                        "time": frame["deviceTimeStamp"],
                        "left_gaze_x": float(left_gaze["x"]),
                        "left_gaze_y": float(left_gaze["y"]),
                        "right_gaze_x": float(right_gaze["x"]),
                        "right_gaze_y": float(right_gaze["y"]),
                    }
                    frames.append(frame_data)

            if len(frames) < 30:  # Minimum frames required
                return {
                    "error": f"Insufficient valid frames: {len(frames)} (minimum 30 required)"
                }

            # Create DataFrame from frames
            df = pd.DataFrame(frames)

            # Prepare data for convergence angle calculation
            left_points = df[["left_gaze_x", "left_gaze_y"]].values
            right_points = df[["right_gaze_x", "right_gaze_y"]].values
            timestamps = df["time"].values

            # Compute convergence angle
            conv_df = self.compute_convergence_angle(
                left_points, right_points, timestamps
            )

            # Resample data for uniform time steps
            resampled_df = self.resample_data(
                conv_df, freq=self.params["resampling_frequency"]
            )

            # Apply smoothing filter
            resampled_df = self.smooth_data(resampled_df)

            # Remove outliers
            resampled_df = self.remove_outliers(resampled_df)

            # Extract features
            features = self.extract_features(resampled_df)

            # Normalize features
            normalized_features = self.normalize_features(features)

            # Analyze peaks
            peak_features = self.analyze_peaks(resampled_df)

            # Determine impairment status
            is_impaired, factors = self.determine_impairment_status(
                {**features, **peak_features}
            )

            # Create result dictionary
            results = {
                "is_impaired": is_impaired,
                "factors": factors,
                "features": features,
                "normalized_features": normalized_features,
            }

            return results

        except Exception as e:
            log.error(f"Error in test_json_predictions: {e}")
            return {"error": str(e)}

    def experiment(self, database_path: str = None) -> Dict[str, Any]:
        """Experiment with the model using provided database.

        Args:
            database_path (str, optional): Path to the database. Defaults to None.

        Returns:
            Dict[str, Any]: A dictionary containing experiment results or error information.
        """
        return self._onExperiment(database_path)

    def _onExperiment(self, conn) -> Dict[str, Any]:
        """Experiment with the model using provided database.

        Args:
            conn: Database connection object.

        Returns:
            Dict[str, Any]: A dictionary containing experiment results or error information.
        """
        try:
            log.info("Starting Cannabis experiment")

            # First, check the database schema to understand what tables and columns exist
            print("\nChecking database schema:")
            schema_query = "SELECT table_name, column_name, data_type FROM information_schema.columns"
            try:
                schema = conn.execute(schema_query).fetchdf()
                print(schema)
            except Exception as e:
                print(f"Could not retrieve schema: {e}")
                # Try an alternative approach to get table structure
                tables = conn.execute("SHOW TABLES").fetchall()
                for table in tables:
                    table_name = table[0]
                    try:
                        print(f"\nStructure of table {table_name}:")
                        table_info = conn.execute(f"DESCRIBE {table_name}").fetchdf()
                        print(table_info)
                    except Exception as e:
                        print(f"Could not describe table {table_name}: {e}")

            # Check if the required tables exist
            tables = conn.execute("SHOW TABLES").fetchall()
            required_tables = ["frame_data", "test_session"]
            missing_tables = [
                table
                for table in required_tables
                if table not in [t[0] for t in tables]
            ]

            if missing_tables:
                print(f"Missing required tables: {missing_tables}")
                print("Creating sample data for testing...")

                # Create sample data for testing
                sample_data = []
                for i in range(10):
                    subject_id = f"subject_{i}"
                    impairment_level = i % 2  # Alternate between 0 and 1

                    # Generate sample eye tracking data
                    frames = []
                    for j in range(50):  # 50 frames per subject
                        time = j * 10000000  # Timestamp in nanoseconds
                        # Generate slightly different gaze points for left and right eyes
                        left_gaze_x = (
                            0.5 + 0.1 * np.sin(j / 5) + 0.05 * np.random.randn()
                        )
                        left_gaze_y = (
                            0.5 + 0.1 * np.cos(j / 5) + 0.05 * np.random.randn()
                        )
                        right_gaze_x = (
                            0.5 + 0.1 * np.sin(j / 5 + 0.1) + 0.05 * np.random.randn()
                        )
                        right_gaze_y = (
                            0.5 + 0.1 * np.cos(j / 5 + 0.1) + 0.05 * np.random.randn()
                        )

                        frames.append(
                            {
                                "subject_id": subject_id,
                                "impairment_level": impairment_level,
                                "time": time,
                                "left_gaze_x": left_gaze_x,
                                "left_gaze_y": left_gaze_y,
                                "right_gaze_x": right_gaze_x,
                                "right_gaze_y": right_gaze_y,
                            }
                        )

                    sample_data.extend(frames)

                # Convert to DataFrame
                sample_df = pd.DataFrame(sample_data)

                # Process each subject
                subject_results = []
                for subject_id in sample_df["subject_id"].unique():
                    subject_data = sample_df[sample_df["subject_id"] == subject_id]
                    impairment_level = subject_data["impairment_level"].iloc[0]
                    session_id = (
                        f"session_{subject_id}"  # Generate a session ID for sample data
                    )

                    print(
                        f"\nProcessing subject {subject_id} (Impairment Level: {impairment_level})"
                    )

                    # Prepare data for convergence angle calculation
                    left_points = subject_data[["left_gaze_x", "left_gaze_y"]].values
                    right_points = subject_data[["right_gaze_x", "right_gaze_y"]].values
                    timestamps = subject_data["time"].values

                    # Calculate convergence angles
                    convergence_data = self.add_convergence_angle(
                        left_points, right_points, timestamps
                    )

                    # Calculate metrics
                    mean_conv = convergence_data["convergence_angle"].mean()
                    std_conv = convergence_data["convergence_angle"].std()
                    cv_conv = std_conv / mean_conv if mean_conv != 0 else 0
                    max_conv = convergence_data["convergence_angle"].max()
                    min_conv = convergence_data["convergence_angle"].min()
                    conv_range = max_conv - min_conv

                    # Create features dictionary
                    features = {
                        "mean_convergence": mean_conv,
                        "std_convergence": std_conv,
                        "cv_convergence": cv_conv,
                        "max_convergence": max_conv,
                        "min_convergence": min_conv,
                        "convergence_range": conv_range,
                    }

                    # Analyze peaks
                    peak_features = self.analyze_peaks(convergence_data)

                    # Determine impairment status with severity
                    is_impaired, factors = self.determine_impairment_status(features)

                    # Calculate confidence score based on how many thresholds are exceeded
                    threshold_count = sum(
                        [
                            mean_conv > self.DETECTION_THRESHOLDS["mean_convergence"],
                            std_conv > self.DETECTION_THRESHOLDS["std_convergence"],
                            cv_conv > self.DETECTION_THRESHOLDS["cv_convergence"],
                            conv_range > self.DETECTION_THRESHOLDS["convergence_range"],
                        ]
                    )

                    confidence_score = min(
                        100, threshold_count * 25
                    )  # 25% per threshold exceeded

                    # Create result dictionary
                    result = {
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "impairment_level": impairment_level,
                        "is_impaired": is_impaired,
                        "confidence_score": confidence_score,
                        "factors": factors,
                        "metrics": {
                            "mean_convergence": float(mean_conv),
                            "std_convergence": float(std_conv),
                            "cv_convergence": float(cv_conv),
                            "max_convergence": float(max_conv),
                            "min_convergence": float(min_conv),
                            "convergence_range": float(conv_range),
                        },
                    }

                    subject_results.append(result)

                # Analyze results
                if subject_results:
                    # Extract metrics into separate columns for easier analysis
                    metrics_data = []
                    for result in subject_results:
                        metrics_data.append(
                            {
                                "subject_id": result["subject_id"],
                                "session_id": result["session_id"],
                                "impairment_level": result["impairment_level"],
                                "is_impaired": result["is_impaired"],
                                "confidence_score": result["confidence_score"],
                                "mean_convergence": result["metrics"][
                                    "mean_convergence"
                                ],
                                "std_convergence": result["metrics"]["std_convergence"],
                                "cv_convergence": result["metrics"]["cv_convergence"],
                                "max_convergence": result["metrics"]["max_convergence"],
                                "min_convergence": result["metrics"]["min_convergence"],
                                "convergence_range": result["metrics"][
                                    "convergence_range"
                                ],
                            }
                        )

                    # Convert to DataFrame
                    results_df = pd.DataFrame(metrics_data)

                    # Save results
                    output_file = "cannabis_analysis_results.csv"
                    results_df.to_csv(output_file, index=False)
                    print(f"\nAnalysis results saved to {output_file}")

                    # Print summary statistics
                    print("\nSummary Statistics:")
                    print(f"Total subjects processed: {len(results_df)}")
                    print("\nImpairment level distribution:")
                    print(results_df.groupby("impairment_level").size())
                    print("\nClassification results:")
                    print(results_df.groupby("is_impaired").size())
                else:
                    print("No subjects could be processed successfully")

                return {
                    "total_subjects": 10,
                    "processed_subjects": len(subject_results),
                    "results": subject_results,
                }

        except Exception as e:
            log.error(f"Error in _onExperiment: {e}")
            return {"error": str(e)}


def main():
    """Main function to run Cannabis analysis on database and output to CSV."""
    try:
        # Initialize model
        model = Cannabis()

        # Use the database path
        db_path = "gaize_v6.duckdb"
        # db_path = "gaize-alcohol-db-v8.db"
        print(f"Starting Cannabis analysis with database: {db_path}")

        # Try to use DuckDB database
        print("Attempting to use DuckDB database")
        try:
            # Create a simple connection test
            conn = duckdb.connect(db_path)
            tables = conn.execute("SHOW TABLES").fetchall()
            print(f"Database tables: {tables}")

            if not tables:
                raise Exception("No tables found in database")

            # Create our wrapper for executing the experiment
            class DuckDBConnection:
                def __init__(self, db_path):
                    self.conn = duckdb.connect(db_path)

                def execute(self, query, params=None):
                    class ResultWrapper:
                        def __init__(self, result):
                            self.result = result

                        def fetchall(self):
                            return self.result.fetchall()

                        def fetchdf(self):
                            return self.result.df()

                    # Replace sqlite placeholders ? with duckdb placeholders $1, $2, etc.
                    if params:
                        for i, param in enumerate(params):
                            query = query.replace("?", f"${i + 1}", 1)
                        result = self.conn.execute(query, params)
                    else:
                        result = self.conn.execute(query)

                    return ResultWrapper(result)

            # Create the connection
            conn = DuckDBConnection(db_path)
            print("DuckDB connection created successfully")

            # Query for sessions with sufficient data
            sessions = conn.execute("""
                SELECT DISTINCT 
                    ts.id as session_id,
                    ts.subject_id,
                    ts.impairmentLevel
                FROM frame_data f
                JOIN test_session ts ON f.test_session_id = ts.id
                WHERE EXISTS (
                    SELECT 1 FROM frame_data f2 
                    WHERE f2.test_session_id = f.test_session_id 
                    GROUP BY f2.test_session_id 
                    HAVING COUNT(*) >= 30
                )
                GROUP BY ts.id, ts.subject_id, ts.impairmentLevel
                ORDER BY ts.id
            """).fetchall()

            if not sessions:
                print("No valid sessions found in database")
                return

            print(f"\nFound {len(sessions)} valid sessions")

            session_results = []
            for session in sessions:
                session_id = session[0]
                subject_id = session[1]
                impairment_level = session[2]

                try:
                    # Query for all frame data for this session
                    query = """
                    SELECT 
                        deviceTimeStamp as time,
                        leftEye.gazeRay.x as left_gaze_x,
                        leftEye.gazeRay.y as left_gaze_y,
                        rightEye.gazeRay.x as right_gaze_x,
                        rightEye.gazeRay.y as right_gaze_y
                    FROM frame_data f
                    JOIN test_session ts ON f.test_session_id = ts.id
                    WHERE ts.id = ?
                    AND NOT leftEye.isBlink
                    AND NOT rightEye.isBlink
                    ORDER BY time
                    """

                    df = conn.execute(query, [session_id]).fetchdf()

                    if len(df) < 30:
                        continue

                    print(
                        f"\nProcessing session {session_id} (Subject: {subject_id}, Impairment Level: {impairment_level})"
                    )

                    # Prepare data for convergence angle calculation
                    left_points = df[["left_gaze_x", "left_gaze_y"]].values
                    right_points = df[["right_gaze_x", "right_gaze_y"]].values
                    timestamps = df["time"].values

                    # Calculate convergence angles
                    convergence_data = model.add_convergence_angle(
                        left_points, right_points, timestamps
                    )

                    # Calculate metrics
                    mean_conv = convergence_data["convergence_angle"].mean()
                    std_conv = convergence_data["convergence_angle"].std()
                    cv_conv = std_conv / mean_conv if mean_conv != 0 else 0
                    max_conv = convergence_data["convergence_angle"].max()
                    min_conv = convergence_data["convergence_angle"].min()
                    conv_range = max_conv - min_conv

                    # Create features dictionary
                    features = {
                        "mean_convergence": mean_conv,
                        "std_convergence": std_conv,
                        "cv_convergence": cv_conv,
                        "max_convergence": max_conv,
                        "min_convergence": min_conv,
                        "convergence_range": conv_range,
                    }

                    # Analyze peaks
                    peak_features = model.analyze_peaks(convergence_data)

                    # Determine impairment status with severity
                    is_impaired, factors = model.determine_impairment_status(features)

                    # Calculate confidence score based on how many thresholds are exceeded
                    threshold_count = sum(
                        [
                            mean_conv > model.DETECTION_THRESHOLDS["mean_convergence"],
                            std_conv > model.DETECTION_THRESHOLDS["std_convergence"],
                            cv_conv > model.DETECTION_THRESHOLDS["cv_convergence"],
                            conv_range
                            > model.DETECTION_THRESHOLDS["convergence_range"],
                        ]
                    )

                    confidence_score = min(
                        100, threshold_count * 25
                    )  # 25% per threshold exceeded

                    # Create result dictionary
                    result = {
                        "session_id": session_id,
                        "subject_id": subject_id,
                        "impairment_level": impairment_level,
                        "is_impaired": is_impaired,
                        "confidence_score": confidence_score,
                        "factors": factors,
                        "metrics": {
                            "mean_convergence": float(mean_conv),
                            "std_convergence": float(std_conv),
                            "cv_convergence": float(cv_conv),
                            "max_convergence": float(max_conv),
                            "min_convergence": float(min_conv),
                            "convergence_range": float(conv_range),
                        },
                    }

                    session_results.append(result)

                except Exception as e:
                    print(f"Error processing session {session_id}: {e}")
                    continue

            # Analyze results
            if session_results:
                # Extract metrics into separate columns for easier analysis
                metrics_data = []
                for result in session_results:
                    metrics_data.append(
                        {
                            "session_id": result["session_id"],
                            "subject_id": result["subject_id"],
                            "impairment_level": result["impairment_level"],
                            "is_impaired": result["is_impaired"],
                            "confidence_score": result["confidence_score"],
                            "mean_convergence": result["metrics"]["mean_convergence"],
                            "std_convergence": result["metrics"]["std_convergence"],
                            "cv_convergence": result["metrics"]["cv_convergence"],
                            "max_convergence": result["metrics"]["max_convergence"],
                            "min_convergence": result["metrics"]["min_convergence"],
                            "convergence_range": result["metrics"]["convergence_range"],
                        }
                    )

                # Convert to DataFrame
                results_df = pd.DataFrame(metrics_data)

                # Save results
                output_file = "cannabis_analysis_results.csv"
                results_df.to_csv(output_file, index=False)
                print(f"\nAnalysis results saved to {output_file}")

                # Print summary statistics
                print("\nSummary Statistics:")
                print(f"Total sessions processed: {len(results_df)}")
                print("\nImpairment level distribution:")
                print(results_df.groupby("impairment_level").size())
                print("\nClassification results:")
                print(results_df.groupby("is_impaired").size())
            else:
                print("No sessions could be processed successfully")

        except Exception as e:
            print(f"Error accessing database: {e}")
            import traceback

            print(traceback.format_exc())
            return

    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    main()
