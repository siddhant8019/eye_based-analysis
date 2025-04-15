# 👁️ Eye-Track-Self: Advanced Impairment Detection System

## 🎯 Vision Statement

Eye-Track-Self is a cutting-edge machine learning system that analyzes eye movement patterns to detect potential impairment. Using advanced gaze tracking and sophisticated algorithms, it processes eye movement data to identify patterns associated with various forms of impairment.

```
👁️ → 📊 → 🤖 → 🎯
Data → Analysis → ML → Results
```

## ✨ Key Features

### 🔍 Multi-Model Analysis

- **Smooth Pursuit Detection**
  - Tracks fluid eye movements
  - Analyzes saccade patterns
  - Measures fixation stability
  - Real-time velocity calculations

### 📊 Data Processing

- Processes data in 5-second windows (150 frames)
- Calculates:
  - Gaze angles (X/Y coordinates)
  - Eye velocity
  - Saccade detection
  - Fixation stability

### 📈 Output Metrics

- Probability scores
- Binary impairment classification
- Detailed feature analysis
- Window-by-window breakdown

## 🚀 Getting Started

### Prerequisites

```bash
python 3.x
pandas
numpy
scikit-learn
```

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/eye-track-self.git
cd eye-track-self
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the main analysis:

```bash
python main.py
```

## 📊 Output Format

The system generates a CSV file containing:

- Probability scores (0-1)
- Under influence classification (threshold: 0.6)
- Feature metrics:
  - Eye velocity
  - Saccade counts
  - Fixation stability
  - Peak velocities

## 🔬 Technical Details

### Feature Extraction

- **Gaze Analysis**

  - X/Y coordinate tracking
  - Velocity calculations
  - Acceleration patterns

- **Saccade Detection**

  - Peak velocity analysis
  - Direction changes
  - Frequency patterns

- **Fixation Stability**
  - Stability measurements
  - Drift analysis
  - Micro-movement detection

### Model Performance

- Window Size: 150 frames (5 seconds)
- Processing Rate: ~30 fps
- Classification Threshold: 0.6

## 📈 Sample Results

```
Feature Statistics:
- Eye Velocity: 44.28 ± 1.96 deg/sec
- Velocity Stability: 23.87 ± 1.29
- Saccade Count (Left): 0.24 ± 0.07
- Saccade Count (Right): 0.71 ± 0.06
```

## 🛠️ Future Enhancements

- Real-time processing capabilities
- Additional impairment detection models
- Enhanced visualization tools
- Mobile device integration

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and support, please open an issue in the repository.

---

_Made with ❤️ by the Eye-Track-Self Team_
