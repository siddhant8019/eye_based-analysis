import os
import sys
from typing import Any, Dict, Tuple, List
import numpy as np
import pandas as pd
from scipy import signal
import warnings
from enum import Enum
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../..")
)
sys.path.insert(0, os.path.join(project_root, "pylib/src"))

from ddgaize.detection.core.AbstractModel import AbstractModel
from ddgaize.detection.core.DataContext import DataContext
from ddgaize.detection.core import DataAccessor
from ddgaize.util import logging_utils
from ddgaize.detection.core.ModelRegistry import ModelRegistry, ModelVersion

log = logging_utils.get_logger(__name__)


class NystagmusContext(DataContext):
    """Context for Horizontal Gaze Nystagmus analysis."""

    class Columns(Enum):
        TIME = "deviceTimeStamp"
        LEFT_GAZE_X = "left_gaze_x"
        RIGHT_GAZE_X = "right_gaze_x"
        LEFT_GAZE_Y = "left_gaze_y"
        RIGHT_GAZE_Y = "right_gaze_y"

        def __str__(self):
            return str(self.value)

        @staticmethod
        def list():
            return list(map(lambda c: c.value, NystagmusContext.Columns))

    def __init__(
        self, dataAccessor: DataAccessor, params: Dict[str, Any], isTrainingData: bool
    ):
        super().__init__(dataAccessor, params, isTrainingData)
        self.frame_data = None
        self.processed_data = None
        self.conn = getattr(dataAccessor, "dbEngine", None)


class NystagmusDetection_v1(AbstractModel):
    """Model for detecting Horizontal Gaze Nystagmus (HGN) based on research standards."""

    THRESHOLDS = {
        "velocity": {
            "saccade": 32.0,
            "nystagmus": 2.5,
        },
        "angle": {
            "max_deviation": 45.0,
            "onset_check": [30.0, 35.0, 40.0, 45.0],
        },
        "frequency": {
            "min": 1.0,
            "max": 9.0,
            "typical": 4.0,
        },
        "duration": {
            "fixation": 3.0,
            "observation": 1.2,
        },
    }

    def __init__(self):
        """Initialize the model."""
        super().__init__()
        self.context = None
        self.params = self._get_default_params()
        self.calibration_data = None
        self.model = None
        self.scaler = None
        self.is_trained = False  # Initialize is_trained attribute
        self.all_results = []  # Add this line to store results for combined plotting

        # Set up MLflow with simplified configuration
        experiment_name = "NystagmusDetection_v1_prod"

        # Set experiment without using deprecated functions
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Error setting experiment: {e}")

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default model parameters."""
        return {
            "sampling_rate": 90,  # Hz - based on Gaize hardware
            "smoothing_window": 3,  # frames
            "min_fixation_samples": 360,  # 4 seconds * 90 Hz
            "calibration_required": True,
        }

    def _calculate_gaze_angles(
        self, gaze_x: float, gaze_y: float
    ) -> Tuple[float, float]:
        """Convert gaze coordinates to angles using arctangent.

        Args:
            gaze_x: x-coordinate from eye tracker
            gaze_y: y-coordinate from eye tracker

        Returns:
            Tuple of (horizontal_angle, vertical_angle) in degrees
        """
        if not isinstance(gaze_x, (float, np.ndarray)) or not isinstance(
            gaze_y, (float, np.ndarray)
        ):
            raise ValueError("Invalid gaze coordinates")

        # Scale factor to map gaze coordinates to appropriate angle ranges
        # Based on typical HGN test range of ±45 degrees
        scale_factor = 1.25

        # Normalize and scale gaze vectors
        gaze_x_scaled = gaze_x * scale_factor
        gaze_y_scaled = gaze_y * scale_factor

        # Ensure we don't exceed valid range for arctan2
        gaze_x_scaled = np.clip(gaze_x_scaled, -0.99, 0.99)
        gaze_y_scaled = np.clip(gaze_y_scaled, -0.99, 0.99)

        # Convert scaled gaze vector to angles using arctangent
        # Using arctan2 for proper quadrant handling
        horizontal_angle = np.degrees(
            np.arctan2(gaze_x_scaled, np.sqrt(1 - gaze_x_scaled**2))
        )
        vertical_angle = np.degrees(
            np.arctan2(gaze_y_scaled, np.sqrt(1 - gaze_y_scaled**2))
        )

        return horizontal_angle, vertical_angle

    def _calculate_slow_phase_velocity(
        self, angles: np.ndarray, time: np.ndarray
    ) -> np.ndarray:
        """Calculate slow phase velocity of nystagmus."""
        # Calculate velocities
        dt = np.diff(time) / 1e7  # Convert to seconds (consistent with other methods)
        velocities = np.diff(angles) / dt

        # Identify fast phases (saccades) using velocity threshold
        saccade_mask = np.abs(velocities) > self.THRESHOLDS["velocity"]["saccade"]

        # Extract slow phase segments
        slow_phase_velocities = velocities.copy()
        slow_phase_velocities[saccade_mask] = np.nan

        return slow_phase_velocities

    def _analyze_nystagmus_at_angle(
        self, angles: np.ndarray, time: np.ndarray, target_angle: float
    ) -> Dict[str, Any]:
        """Analyze nystagmus characteristics at a specific gaze angle."""
        if len(angles) != len(time):
            raise ValueError("Angles and time arrays must have same length")
        if len(angles) < self.params["sampling_rate"]:
            return {
                "nystagmus_detected": False,
                "spv": 0.0,
                "frequency": 0.0,
                "duration": 0.0,
            }

        # Find segments where gaze is held at target angle
        # Use NARROWER windows to be more strict
        angle_window = 5.0 if target_angle >= 40.0 else 3.0
        angle_mask = np.abs(angles - target_angle) < angle_window
        n_samples_at_angle = np.sum(angle_mask)

        if not np.any(angle_mask):
            return {
                "nystagmus_detected": False,
                "spv": 0.0,
                "frequency": 0.0,
                "duration": 0.0,
            }

        # Get slow phase velocity
        spv = self._calculate_slow_phase_velocity(angles[angle_mask], time[angle_mask])
        mean_spv = float(np.nanmean(np.abs(spv)))

        # Calculate nystagmus frequency using FFT
        peak_freq = 0.0
        if len(spv) > self.params["sampling_rate"]:
            # Remove NaN values for frequency analysis
            valid_spv = spv[~np.isnan(spv)]
            if len(valid_spv) > self.params["sampling_rate"]:
                freqs, power = signal.welch(
                    valid_spv,
                    fs=self.params["sampling_rate"],
                    nperseg=self.params["sampling_rate"],
                )
                # Find peak frequency in the nystagmus range
                nystagmus_range = (
                    self.THRESHOLDS["frequency"]["min"],
                    self.THRESHOLDS["frequency"]["max"],
                )
                mask = (freqs >= nystagmus_range[0]) & (freqs <= nystagmus_range[1])
                if np.any(mask) and np.any(power[mask]):
                    peak_freq = float(freqs[mask][np.argmax(power[mask])])

        # Determine if nystagmus is present based on research criteria
        duration = float(n_samples_at_angle / self.params["sampling_rate"])

        # More lenient criteria
        min_spv = self.THRESHOLDS["velocity"]["nystagmus"] * 1.5

        # More lenient frequency requirements
        min_freq = self.THRESHOLDS["frequency"]["min"]
        max_freq = self.THRESHOLDS["frequency"]["max"]

        # More lenient minimum observation time
        min_duration = self.THRESHOLDS["duration"]["observation"]

        # Require valid frequency measurement (no zero values)
        nystagmus_detected = (
            mean_spv >= min_spv
            and peak_freq >= min_freq  # Must have valid frequency
            and peak_freq <= max_freq
            and duration >= min_duration
        )

        return {
            "nystagmus_detected": nystagmus_detected,
            "spv": mean_spv,
            "frequency": float(peak_freq),
            "duration": duration,
            "angle": target_angle,
        }

    def _determine_onset_angle(self, df: pd.DataFrame, eye: str) -> float:
        """Determine the angle of onset for nystagmus."""
        onset_angle = 45.0

        # Check each angle from 30° to 45° in steps
        for angle in reversed(self.THRESHOLDS["angle"]["onset_check"]):
            result = self._analyze_nystagmus_at_angle(
                df[f"{eye}_gaze_angle"].values,
                df[str(self.context.Columns.TIME.value)].values,
                angle,
            )

            if result["nystagmus_detected"]:
                onset_angle = angle
                break

        return onset_angle

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate the input data meets minimum requirements."""
        if len(df) < self.params["sampling_rate"]:
            return False

        required_columns = [
            str(self.context.Columns.TIME.value),
            str(self.context.Columns.LEFT_GAZE_X.value),
            str(self.context.Columns.RIGHT_GAZE_X.value),
            str(self.context.Columns.LEFT_GAZE_Y.value),
            str(self.context.Columns.RIGHT_GAZE_Y.value),
        ]

        for col in required_columns:
            if col not in df.columns:
                log.warning(f"Missing required column: {col}")
                return False

        return True

    def load_pretrained_results(self):
        """Load pre-trained models from MLflow model registry."""
        try:
            # Initialize MLflow client
            client = mlflow.tracking.MlflowClient()

            try:
                # Get the latest version of the model
                latest_version = client.get_latest_versions(
                    "NystagmusDetection_v1", stages=["None"]
                )[0]
                run_id = latest_version.run_id

                # Load the preprocessor from the same run
                self.scaler = mlflow.sklearn.load_model(f"runs:/{run_id}/preprocessor")

                # Load the XGBoost model
                self.model = mlflow.xgboost.load_model(
                    model_uri="models:/NystagmusDetection_v1/latest"
                )

                self.is_trained = True
                print(
                    f"Successfully loaded model version {latest_version.version} from MLflow"
                )
                return True

            except Exception as e:
                print(f"Error loading model and preprocessor: {e}")
                return False

        except Exception as e:
            print(f"Error loading pre-trained models from MLflow: {e}")
            return False

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
            # Initialize context
            self.context = NystagmusContext(dataAccessor, {}, False)

            # Load the model and scaler if not already loaded
            if not self.is_trained:
                if not self.load_pretrained_results():
                    print("Will use rule-based detection since no model is available")

            return self._onExecute(dataAccessor)
        except Exception as e:
            log.error(f"Error in execute: {e}")
            return {"error": str(e), "HGN_prod": False}

    def _onExecute(self, dataAccessor: DataAccessor) -> Dict[str, Any]:
        """Execute model prediction using trained model."""
        try:
            data = dataAccessor.json
            if not data or "data" not in data:
                return {"error": "Invalid data format", "HGN_pred": False}

            # Process frames and extract features
            frames = self._process_frames(data)
            if not frames:
                return {"error": "No valid frames found", "HGN_pred": False}

            df = pd.DataFrame(frames)

            # Calculate features
            left_angles_h, _ = self._calculate_gaze_angles(
                df["left_gaze_x"].astype(float).values,
                df["left_gaze_y"].astype(float).values,
            )
            right_angles_h, _ = self._calculate_gaze_angles(
                df["right_gaze_x"].astype(float).values,
                df["right_gaze_y"].astype(float).values,
            )

            left_metrics = self._analyze_nystagmus_at_angle(
                left_angles_h, df["deviceTimeStamp"].values, 45.0
            )
            right_metrics = self._analyze_nystagmus_at_angle(
                right_angles_h, df["deviceTimeStamp"].values, 45.0
            )

            # If we have a trained model and scaler, use them for prediction
            if self.is_trained and hasattr(self, "model") and hasattr(self, "scaler"):
                # Prepare features for prediction
                features = np.array(
                    [
                        [
                            left_metrics["spv"],
                            right_metrics["spv"],
                            left_metrics["frequency"],
                            right_metrics["frequency"],
                            left_metrics["duration"],
                            right_metrics["duration"],
                        ]
                    ]
                )

                try:
                    # Scale features
                    features_scaled = self.scaler.transform(features)
                    hgn_pred_algo = (
                        left_metrics["nystagmus_detected"]
                        or right_metrics["nystagmus_detected"]
                    ) and (
                        left_metrics["spv"] >= self.THRESHOLDS["velocity"]["nystagmus"]
                        or right_metrics["spv"]
                        >= self.THRESHOLDS["velocity"]["nystagmus"]
                    )
                    # Make prediction
                    hgn_prob = self.model.predict_proba(features_scaled)[0][1]
                    hgn_pred_model = bool(hgn_prob >= 0.5)

                    if 0.0 < hgn_prob < 0.3:
                        hgn_pred = hgn_pred_algo
                    else:
                        hgn_pred = hgn_pred_model

                except Exception as e:
                    print(f"Error in model prediction: {e}")
                    # Fallback to rule-based detection
                    hgn_pred = (
                        left_metrics["nystagmus_detected"]
                        or right_metrics["nystagmus_detected"]
                    ) and (
                        left_metrics["spv"] >= self.THRESHOLDS["velocity"]["nystagmus"]
                        or right_metrics["spv"]
                        >= self.THRESHOLDS["velocity"]["nystagmus"]
                    )
                    hgn_prob = 1.0 if hgn_pred else 0.0
            else:
                # Use rule-based detection
                hgn_pred = (
                    left_metrics["nystagmus_detected"]
                    or right_metrics["nystagmus_detected"]
                ) and (
                    left_metrics["spv"] >= self.THRESHOLDS["velocity"]["nystagmus"]
                    or right_metrics["spv"] >= self.THRESHOLDS["velocity"]["nystagmus"]
                )
                hgn_prob = 1.0 if hgn_pred else 0.0

            return {
                "HGN_pred": hgn_pred,
                "HGN_pred_model": hgn_pred_model,
                "HGN_pred_algo": hgn_pred_algo,
                "metrics": {"left": left_metrics, "right": right_metrics},
            }

        except Exception as e:
            log.error(f"Error in nystagmus detection: {e}")
            return {"error": str(e), "HGN_pred": False}

    def _process_frames(self, data: Dict) -> List[Dict]:
        """Process raw frames and extract relevant data."""
        frames = []
        for frame in data.get("data", []):
            if not isinstance(frame, dict):
                continue

            left_eye = frame.get("leftEye", {})
            right_eye = frame.get("rightEye", {})

            if all(
                [
                    not left_eye.get("isBlink"),
                    not right_eye.get("isBlink"),
                    left_eye.get("gazeRay", {}).get("x") is not None,
                    left_eye.get("gazeRay", {}).get("y") is not None,
                    right_eye.get("gazeRay", {}).get("x") is not None,
                    right_eye.get("gazeRay", {}).get("y") is not None,
                ]
            ):
                frames.append(
                    {
                        "deviceTimeStamp": frame["deviceTimeStamp"],
                        "left_gaze_x": float(left_eye["gazeRay"]["x"]),
                        "left_gaze_y": float(left_eye["gazeRay"]["y"]),
                        "right_gaze_x": float(right_eye["gazeRay"]["x"]),
                        "right_gaze_y": float(right_eye["gazeRay"]["y"]),
                    }
                )

        return frames

    def experiment(self, database_path) -> Dict[str, Any]:
        """Train the model using provided training data.

        Args:
            database_path: Path to the database for training.

        Returns:
            Dict[str, Any]: A dictionary containing training metrics or error information.
        """
        # Simply pass the connection object to _onExperiment
        return self._onExperiment(database_path)

    def _onExperiment(self, conn):
        """Train the model using provided database.

        Args:
            conn: DuckDB connection object
        """
        print("Starting experiment")
        try:
            # First, ensure no runs are active
            try:
                mlflow.end_run()
            except Exception:
                pass

            # Create experiment if it doesn't exist
            experiment_name = "NystagmusDetection_v1_prod"
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)
            except Exception as e:
                print(f"Error setting up experiment: {e}")
                return {"error": str(e)}

            with mlflow.start_run(run_name="nystagmus-analysis") as run:
                # Get list of test sessions with HGN information
                sessions = conn.execute("""
                    SELECT DISTINCT 
                        f.test_session_id,
                        ts.subject_id,
                        p.hgn,
                        p.hgn_45
                    FROM frame_data f
                    JOIN test_session ts ON f.test_session_id = ts.id
                    JOIN participant_data p ON ts.subject_id = p.subject_id
                    WHERE p.hgn IS NOT NULL
                    AND EXISTS (
                        SELECT 1 FROM frame_data f2 
                        WHERE f2.test_session_id = f.test_session_id 
                        GROUP BY f2.test_session_id 
                        HAVING COUNT(*) >= 30
                    )
                    ORDER BY f.test_session_id
                """).fetchall()

                print(f"\nFound {len(sessions)} test sessions in the database")
                mlflow.log_param("total_sessions", len(sessions))

                # Process sessions and extract features
                session_data = []
                for session in sessions:
                    session_id, subject_id, hgn, _ = session
                    try:
                        # Get frame data for this session with proper filtering
                        df = conn.execute(
                            """
                            WITH valid_frames AS (
                                SELECT 
                                    deviceTimeStamp,
                                    leftEye.gazeRay.x as left_gaze_x,
                                    leftEye.gazeRay.y as left_gaze_y,
                                    rightEye.gazeRay.x as right_gaze_x,
                                    rightEye.gazeRay.y as right_gaze_y
                                FROM frame_data 
                                WHERE test_session_id = ?
                                AND NOT leftEye.isBlink
                                AND NOT rightEye.isBlink
                                AND leftEye.gazeRay.x IS NOT NULL
                                AND leftEye.gazeRay.y IS NOT NULL
                                AND rightEye.gazeRay.x IS NOT NULL
                                AND rightEye.gazeRay.y IS NOT NULL
                                ORDER BY deviceTimeStamp
                            )
                            SELECT * FROM valid_frames
                            """,
                            [session_id],
                        ).fetchdf()

                        if len(df) < 30:
                            print(f"Insufficient data for session {session_id}")
                            continue

                        # Convert gaze coordinates to angles
                        left_angles_h, _ = self._calculate_gaze_angles(
                            df["left_gaze_x"].astype(float).values,
                            df["left_gaze_y"].astype(float).values,
                        )
                        right_angles_h, _ = self._calculate_gaze_angles(
                            df["right_gaze_x"].astype(float).values,
                            df["right_gaze_y"].astype(float).values,
                        )

                        # Calculate metrics for each eye at 45 degrees
                        left_metrics = self._analyze_nystagmus_at_angle(
                            left_angles_h, df["deviceTimeStamp"].values, 45.0
                        )
                        right_metrics = self._analyze_nystagmus_at_angle(
                            right_angles_h, df["deviceTimeStamp"].values, 45.0
                        )

                        # Store features and target
                        features = {
                            "session_id": session_id,
                            "subject_id": subject_id,
                            "left_spv": left_metrics["spv"],
                            "right_spv": right_metrics["spv"],
                            "left_frequency": left_metrics["frequency"],
                            "right_frequency": right_metrics["frequency"],
                            "left_duration": left_metrics["duration"],
                            "right_duration": right_metrics["duration"],
                            "hgn": float(hgn),
                        }
                        session_data.append(features)

                    except Exception as e:
                        print(f"Error processing session {session_id}: {str(e)}")
                        continue

                if not session_data:
                    raise ValueError("No sessions could be processed successfully")

                # Create DataFrame and prepare for training
                df = pd.DataFrame(session_data)

                # Prepare features and target
                X = df[
                    [
                        "left_spv",
                        "right_spv",
                        "left_frequency",
                        "right_frequency",
                        "left_duration",
                        "right_duration",
                    ]
                ]
                y = df["hgn"]

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train XGBoost classifier with optimized parameters
                xgb_params = {
                    "objective": "binary:logistic",
                    "n_estimators": 100,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                }
                self.model = xgb.XGBClassifier(**xgb_params)
                self.model.fit(X_train_scaled, y_train)
                self.scaler = scaler

                # Make predictions for all data
                X_all_scaled = scaler.transform(X)
                y_pred_all = self.model.predict(X_all_scaled)
                y_prob_all = self.model.predict_proba(X_all_scaled)[:, 1]

                # Create comprehensive DataFrame with all results
                comprehensive_df = df.copy()
                comprehensive_df["predicted_hgn"] = y_pred_all
                comprehensive_df["prediction_probability"] = y_prob_all
                comprehensive_df["dataset_split"] = "train"  # Default to train
                comprehensive_df.loc[X_test.index, "dataset_split"] = (
                    "test"  # Mark test samples
                )

                # Drop duplicates based on session_id, keeping the first occurrence
                comprehensive_df = comprehensive_df.drop_duplicates(
                    subset=["session_id"], keep="first"
                )

                # Reorder columns to put actual and predicted HGN first
                column_order = [
                    "subject_id",
                    "session_id",
                    "hgn",  # actual HGN
                    "predicted_hgn",
                    "prediction_probability",
                    "dataset_split",
                    "left_spv",
                    "right_spv",
                    "left_frequency",
                    "right_frequency",
                    "left_duration",
                    "right_duration",
                ]
                comprehensive_df = comprehensive_df[column_order]

                # Save comprehensive dataset
                artifacts_dir = Path(project_root) / "artifacts"
                artifacts_dir.mkdir(exist_ok=True)
                comprehensive_path = artifacts_dir / "nystagmus_dataset.csv"
                comprehensive_df.to_csv(comprehensive_path, index=False)
                mlflow.log_artifact(str(comprehensive_path))

                # Calculate metrics (using test set only)
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred_all[X_test.index]),
                    "precision": precision_score(y_test, y_pred_all[X_test.index]),
                    "recall": recall_score(y_test, y_pred_all[X_test.index]),
                    "f1": f1_score(y_test, y_pred_all[X_test.index]),
                }

                # Create results DataFrame
                results_df = pd.DataFrame(
                    {
                        "subject_id": df.loc[X_test.index, "subject_id"],
                        "actual_hgn": y_test,
                        "predicted_hgn": y_pred_all[X_test.index],
                        "prediction_probability": y_prob_all[X_test.index],
                    }
                )

                # Add features to results
                for col in X.columns:
                    results_df[col] = X_test[col].values

                # Create a single combined visualization plot
                plt.style.use("default")
                fig, ax1 = plt.subplots(figsize=(75, 58))

                # Sort by actual HGN for better visualization
                sorted_indices = results_df["actual_hgn"].argsort()
                x_indices = results_df["subject_id"].values[
                    sorted_indices
                ]  # Use subject_id for x-axis

                # Plot HGN predictions and actual values
                ax1.plot(
                    range(len(x_indices)),  # Use range for positioning
                    results_df["actual_hgn"].values[sorted_indices],
                    "k--",
                    label="Actual HGN",
                    alpha=0.7,
                    linewidth=1.5,
                )
                ax1.scatter(
                    range(len(x_indices)),  # Use range for positioning
                    results_df["predicted_hgn"].values[sorted_indices],
                    c="purple",
                    label="HGN Prediction",
                    alpha=0.4,
                    s=50,
                )
                ax1.set_ylabel("HGN Detection", color="k")
                ax1.tick_params(axis="y", labelcolor="k")

                # Plot SPV metrics
                ax2 = ax1.twinx()
                ax2.plot(
                    range(len(x_indices)),  # Use range for positioning
                    results_df["left_spv"].values[sorted_indices],
                    "b-",
                    label="Left SPV",
                    linewidth=1.5,
                    alpha=0.7,
                )
                ax2.plot(
                    range(len(x_indices)),  # Use range for positioning
                    results_df["right_spv"].values[sorted_indices],
                    "g-",
                    label="Right SPV",
                    linewidth=1.5,
                    alpha=0.7,
                )
                ax2.set_ylabel("Slow Phase Velocity (deg/s)", color="b")
                ax2.tick_params(axis="y", labelcolor="b")

                # Plot frequency metrics
                ax3 = ax1.twinx()
                ax3.spines["right"].set_position(("outward", 60))
                ax3.plot(
                    range(len(x_indices)),  # Use range for positioning
                    results_df["left_frequency"].values[sorted_indices],
                    "m-",
                    label="Left Frequency",
                    linewidth=1.5,
                    alpha=0.7,
                )
                ax3.plot(
                    range(len(x_indices)),  # Use range for positioning
                    results_df["right_frequency"].values[sorted_indices],
                    "c-",
                    label="Right Frequency",
                    linewidth=1.5,
                    alpha=0.7,
                )
                ax3.set_ylabel("Frequency (Hz)", color="m")
                ax3.tick_params(axis="y", labelcolor="m")

                # Plot duration metrics
                ax4 = ax1.twinx()
                ax4.spines["right"].set_position(("outward", 120))
                ax4.plot(
                    range(len(x_indices)),  # Use range for positioning
                    results_df["left_duration"].values[sorted_indices],
                    "y-",
                    label="Left Duration",
                    linewidth=1.5,
                    alpha=0.7,
                )
                ax4.plot(
                    range(len(x_indices)),  # Use range for positioning
                    results_df["right_duration"].values[sorted_indices],
                    "r-",
                    label="Right Duration",
                    linewidth=1.5,
                    alpha=0.7,
                )
                ax4.set_ylabel("Duration (s)", color="r")
                ax4.tick_params(axis="y", labelcolor="r")

                # Set title and x-label
                plt.title("Combined Nystagmus Analysis", pad=20, fontsize=14)
                ax1.set_xlabel("Subject ID")

                # Set x-ticks to show subject IDs
                ax1.set_xticks(range(len(x_indices)))
                ax1.set_xticklabels(x_indices, rotation=45, ha="right")

                # Add grid
                ax1.grid(True, alpha=0.2)

                # Combine legends from all axes
                all_axes = [ax for ax in fig.axes]
                lines, labels = [], []
                for ax in all_axes:
                    ax_lines, ax_labels = ax.get_legend_handles_labels()
                    lines.extend(ax_lines)
                    labels.extend(ax_labels)

                ax1.legend(
                    lines,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=(0.05, 1.15),
                    ncol=4,
                )

                # Adjust layout and save
                plt.tight_layout()
                plot_path = artifacts_dir / "nystagmus_analysis_plot.png"
                plt.savefig(plot_path, bbox_inches="tight", dpi=300)
                mlflow.log_artifact(str(plot_path))
                plt.close()

                # Log metrics and artifacts
                mlflow.log_metrics(metrics)

                # Save preprocessor as an artifact
                mlflow.sklearn.log_model(
                    self.scaler,
                    "preprocessor",
                    input_example=X_test_scaled[:1],
                )

                # Save model
                mlflow.xgboost.log_model(
                    self.model,
                    "nystagmus_model",
                    registered_model_name="NystagmusDetection_v1",
                )

                return metrics

        except Exception as e:
            if mlflow.active_run():
                mlflow.end_run()
            raise ValueError(f"Model training failed: {str(e)}")
        finally:
            if mlflow.active_run():
                mlflow.end_run()

    def plot_results(
        self, file_name: str, results: Dict[str, Any], artifacts_dir: Path
    ) -> None:
        """Store results for combined plotting."""
        # Store the results with the filename
        self.all_results.append({"file_name": file_name, "results": results})

    def create_combined_plot(self, artifacts_dir: Path) -> None:
        """Create a combined plot of all processed JSON files.

        Args:
            artifacts_dir (Path): Directory to save the plot
        """
        try:
            if not self.all_results:
                print("No results to plot")
                return

            plt.figure(figsize=(15, 8))

            # Number of files
            n_files = len(self.all_results)

            # Create arrays for each metric
            hgn_pred = []
            hgn_algo = []
            left_spv = []
            right_spv = []
            file_names = []

            # Collect data from all results
            for result in self.all_results:
                file_names.append(result["file_name"])
                r = result["results"]
                hgn_pred.append(float(r["HGN_pred"]))
                hgn_algo.append(float(r["HGN_pred_algo"]))
                left_spv.append(r["metrics"]["left"]["spv"])
                right_spv.append(r["metrics"]["right"]["spv"])

            # Create x positions for bars
            x = np.arange(n_files)
            width = 0.2  # Width of bars

            # Create bars
            plt.bar(
                x - width * 1.5,
                hgn_pred,
                width,
                label="HGN Pred",
                color="blue",
                alpha=0.7,
            )
            plt.bar(
                x - width / 2, hgn_algo, width, label="HGN Algo", color="red", alpha=0.7
            )
            plt.bar(
                x + width / 2,
                left_spv,
                width,
                label="Left SPV",
                color="green",
                alpha=0.7,
            )
            plt.bar(
                x + width * 1.5,
                right_spv,
                width,
                label="Right SPV",
                color="orange",
                alpha=0.7,
            )

            # Customize the plot
            plt.title("Combined HGN Analysis Results", pad=20, fontsize=14)
            plt.xlabel("JSON Files")
            plt.ylabel("Values")

            # Set x-axis labels (file names)
            plt.xticks(x, file_names, rotation=45, ha="right")

            # Add legend
            plt.legend()

            # Add grid
            plt.grid(True, alpha=0.3)

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

            # Save plot
            plot_path = artifacts_dir / "combined_hgn_analysis.png"
            plt.savefig(plot_path, bbox_inches="tight", dpi=300)
            plt.close()

            print(f"Combined plot saved as: {plot_path}")

        except Exception as e:
            print(f"Error creating combined plot: {e}")
