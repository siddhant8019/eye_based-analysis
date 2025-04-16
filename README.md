# 👁️ Eye-Track-Self

A machine learning system that analyzes eye movement patterns to detect potential impairment through multiple specialized models.

## 🎯 Overview

Eye-Track-Self processes eye tracking data to identify patterns associated with various forms of impairment using three specialized models:

- Smooth Pursuit Detection
- Nystagmus Analysis
- Cannabis Influence Detection

## 🚀 Features

- **Multi-Model Analysis**: Combines three specialized models for comprehensive impairment detection
- **Real-time Processing**: Analyzes data in 5-second windows (150 frames)
- **Advanced Metrics**: Tracks gaze angles, velocity, saccades, and fixation stability
- **Scalable Architecture**: Modular design with base model class for easy extension

## 🛠️ Tech Stack

- Python 3.x
- NumPy & Pandas for data processing
- Scikit-learn & XGBoost for ML models
- Matplotlib for visualization

## 🚦 Getting Started

1. Clone and setup:

```bash
git clone https://github.com/yourusername/eye-track-self.git
cd eye-track-self
pip install -r requirements.txt
```

2. Run analysis:

```bash
python main.py
```

## 📊 Output

The system generates detailed analysis including:

- Probability scores for each model
- Binary impairment classification
- Feature metrics (velocity, saccades, stability)
- Window-by-window analysis

## 🏗️ Architecture

- `base_model.py`: Core model functionality
- `smooth_pursuit_model.py`: Smooth pursuit detection
- `nystagmus_model.py`: Nystagmus analysis
- `cannabis_model.py`: Cannabis influence detection
- `preprocessor.py`: Data preprocessing pipeline
- `main.py`: Orchestration and execution

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---
