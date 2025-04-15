import os
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from smooth_pursuit_model import SmoothPursuitModel
import warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """Load and prepare data from CSV file."""
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from {csv_path}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Warning: Found {missing_values.sum()} missing values in the data")
            print("Missing values by column:")
            for col, count in missing_values.items():
                if count > 0:
                    print(f"  {col}: {count}")
        
        return df
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def create_training_data(model, df, num_samples=10):
    """Create training data from the DataFrame."""
    print(f"Creating {num_samples} training samples")
    
    # Split data into chunks
    chunk_size = len(df) // num_samples
    training_data = []
    
    for i in range(num_samples):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        # Get chunk of data
        chunk = df.iloc[start_idx:end_idx].copy()
        
        try:
            # Preprocess data
            processed_chunk = model.preprocess(chunk)
            
            # Extract features
            features = model.extract_features(processed_chunk)
            
            # Add random label (0 or 1) for testing
            features['label'] = np.random.randint(0, 2)
            
            training_data.append(features)
        except Exception as e:
            print(f"Warning: Error processing chunk {i}: {e}")
            continue
    
    if not training_data:
        raise ValueError("No valid training samples could be created")
    
    return training_data

def process_data_in_windows(model, df, window_size=150):
    """Process data in windows and return predictions for each window."""
    all_results = []
    num_windows = len(df) // window_size
    
    print(f"\nProcessing {num_windows} windows of size {window_size}")
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        
        # Get window of data
        window_df = df.iloc[start_idx:end_idx].copy()
        window_df.reset_index(drop=True, inplace=True)
        
        try:
            # Make predictions for this window
            results = model.predict(window_df)
            
            # Add window information
            results['window_start'] = start_idx
            results['window_end'] = end_idx
            results['window_index'] = i
            
            all_results.append(results)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_windows} windows")
                
        except Exception as e:
            print(f"Warning: Error processing window {i}: {e}")
            continue
    
    return all_results

def save_results_to_csv(results, output_path):
    """Save all results to CSV file."""
    # Create list of dictionaries for DataFrame
    rows = []
    for result in results:
        row = {
            'probability': result['probability'],
            'under_influence': result['probability'] >= 0.6,
            'prediction': result['prediction'],
        }
        # Add all features
        row.update(result['features'])
        rows.append(row)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_path, index=False)
    return results_df

def main():
    try:
        # Initialize paths
        data_path = "gaze.csv"
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)

        # Load data
        print("Loading data...")
        df = load_and_prepare_data(data_path)

        # Initialize model
        print("Initializing model...")
        model = SmoothPursuitModel()
        
        # Create training data
        print("Creating training data...")
        training_data = create_training_data(model, df, num_samples=10)
        
        # Prepare data for training
        print("Preparing data for training...")
        X, y = model.prepare_training_data(training_data)
        
        # Train model
        print("Training model...")
        model.train(X, y)
        print("Model training completed")

        # Process data in windows
        print("Processing data in windows...")
        window_size = 150  # 5 seconds at 30 fps
        all_results = process_data_in_windows(model, df, window_size)

        # Save results
        results_path = results_dir / "smooth_pursuit_results.csv"
        results_df = save_results_to_csv(all_results, results_path)
        print(f"\nResults saved to {results_path}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total windows processed: {len(results_df)}")
        print(f"Impaired windows: {results_df['prediction'].sum()} ({results_df['prediction'].mean()*100:.1f}%)")
        print("\nFeature Statistics:")
        for col in results_df.columns:
            if col not in ['window_index', 'window_start', 'window_end', 'prediction']:
                print(f"{col}:")
                print(f"  Mean: {results_df[col].mean():.2f}")
                print(f"  Std: {results_df[col].std():.2f}")

    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main() 