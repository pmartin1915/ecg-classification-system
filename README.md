# ECG Classification System

An AI-powered system for classifying ECG patterns into 5 cardiac conditions using advanced machine learning techniques.

## 🚀 Features

- **87.3% accuracy** using Random Forest algorithm
- Real-time ECG classification
- Support for 12-lead ECG (works with fewer leads)
- Clinical decision support interface
- Comprehensive monitoring dashboard

## 🩺 Supported Conditions

1. **NORM** - Normal sinus rhythm
2. **MI** - Myocardial Infarction (Heart Attack)
3. **STTC** - ST/T Changes
4. **CD** - Conduction Disorders
5. **HYP** - Hypertrophy

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/pmartin1915/ecg-classification-system.git
cd ecg-classification-system
```

2. Create a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application

### Local Development
```bash
streamlit run app/main.py
```

### With ngrok (for sharing)
```bash
python app/main.py --use-ngrok
```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 87.3% |
| F1-Score | 86.8% |
| Precision | 88.1% |
| Recall | 85.6% |

## 📁 Project Structure

```
ecg-classification-system/
├── app/                    # Streamlit application
│   ├── components/         # UI components
│   └── utils/             # Helper functions
├── models/                # ML models and training
│   ├── trained_models/    # Saved model files
│   └── training/          # Training scripts
├── data/                  # Data directory
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed data
│   └── features/         # Extracted features
├── notebooks/            # Jupyter notebooks
├── config/              # Configuration files
└── tests/               # Unit tests
```

## 🗂️ Datasets

- **Primary**: PTB-XL dataset (21,388 samples)
- **Secondary**: MIT-BIH Arrhythmia Database (for enhanced training)

## 👨‍⚕️ Clinical Context

This system is designed as a decision support tool for healthcare professionals. It should not be used as the sole basis for clinical decisions.

## 🚧 Development Status

- [x] Phase 1: Foundation & Data Loading
- [x] Phase 2: Preprocessing & Signal Filtering
- [x] Phase 3: Feature Engineering
- [x] Phase 4: Model Training & Evaluation
- [x] Phase 5: Deployment Interface
- [ ] Phase 6: Production Deployment
- [ ] Phase 7: Clinical Validation

## 🤝 Contributing

This is a DNP student project. Contributions and feedback are welcome!

## 📝 License

This project is for educational and research purposes.

## 👤 Author

DNP Student - pmartin1915

## 🙏 Acknowledgments

- PTB-XL Database providers
- MIT-BIH Arrhythmia Database
- Streamlit community