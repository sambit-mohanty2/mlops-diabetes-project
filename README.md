# Diabetic Risk Prediction Model - MLOps Project

A comprehensive machine learning operations (MLOps) project for predicting diabetic risk using industry best practices, automated pipelines, and production-ready code.

## 🎯 Project Overview

This project implements an end-to-end diabetic risk prediction model with:
- **Data Pipeline**: Automated data loading, validation, and preprocessing
- **Feature Engineering**: Intelligent feature selection and transformation
- **Model Training**: Multiple ML algorithms with hyperparameter optimization
- **Model Evaluation**: Comprehensive metrics, visualization, and interpretation
- **MLOps Integration**: Experiment tracking, model versioning, and CI/CD pipelines
- **Deployment**: API endpoints for real-time predictions
- **Monitoring**: Performance tracking and data drift detection

## 📁 Project Structure

```
mlops-diabetes-project/
├── data/
│   ├── raw/                     # Original raw datasets
│   ├── processed/               # Cleaned and processed data
│   └── external/                # External data sources
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_final_model.ipynb
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py          # Configuration management
│   │   └── paths.py             # Path definitions
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # Data loading utilities
│   │   ├── preprocessor.py      # Data preprocessing
│   │   ├── validator.py         # Data validation
│   │   └── splitter.py          # Train/test splitting
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineer.py          # Feature engineering
│   │   ├── selector.py          # Feature selection
│   │   └── transformer.py       # Feature transformation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── builder.py           # Model builders
│   │   ├── trainer.py           # Training logic
│   │   ├── predictor.py         # Prediction utilities
│   │   └── persistence.py       # Model save/load
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # Evaluation metrics
│   │   ├── visualizer.py        # Result visualization
│   │   └── report.py            # Report generation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py            # Logging setup
│   │   ├── helpers.py           # Helper functions
│   │   └── monitoring.py        # Monitoring utilities
│   └── api/
│       ├── __init__.py
│       ├── main.py              # FastAPI app
│       ├── routes.py            # API routes
│       └── schemas.py           # Request/response schemas
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
├── models/                      # Trained model artifacts
├── mlruns/                      # MLflow experiment tracking
├── outputs/
│   ├── reports/                 # Generated reports
│   ├── visualizations/          # Plots and charts
│   └── predictions/             # Prediction results
├── .github/workflows/
│   ├── ci.yml                   # CI pipeline
│   ├── tests.yml                # Test pipeline
│   └── deploy.yml               # Deployment pipeline
├── config/
│   ├── config.yaml              # Main configuration
│   ├── model_config.yaml        # Model configuration
│   └── data_config.yaml         # Data configuration
├── docker/
│   ├── Dockerfile               # Docker image for app
│   └── Dockerfile.train         # Docker image for training
├── scripts/
│   ├── train_model.py           # Training script
│   ├── evaluate_model.py        # Evaluation script
│   ├── serve_model.py           # Model serving script
│   └── predict_batch.py         # Batch prediction script
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── pyproject.toml
├── CONTRIBUTING.md
├── LICENSE
└── README.md

```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker (optional)
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/sambit-mohanty2/mlops-diabetes-project.git
cd mlops-diabetes-project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Pipeline

1. **Data Processing**
```bash
python scripts/train_model.py --mode prepare
```

2. **Train Model**
```bash
python scripts/train_model.py --mode train
```

3. **Evaluate Model**
```bash
python scripts/evaluate_model.py
```

4. **Serve Model**
```bash
python scripts/serve_model.py
```

## 📊 Key Features

### Data Pipeline
- Automated data validation with Pandera schemas
- Handling missing values and outliers
- Stratified train/test splitting
- Data versioning with DVC

### Feature Engineering
- Automated feature selection
- Polynomial and interaction features
- Feature scaling and normalization
- Feature importance analysis

### Model Training
- Multiple algorithms: Logistic Regression, XGBoost, LightGBM, CatBoost
- Hyperparameter optimization with Optuna
- Cross-validation with stratified folds
- Experiment tracking with MLflow

### Evaluation & Monitoring
- Comprehensive metrics: AUC-ROC, F1, Precision, Recall
- Confusion matrix and classification reports
- Feature importance visualization
- Model comparison dashboard
- Data drift detection

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Model parameters
- Feature settings
- Data paths
- Training hyperparameters

## 📈 MLOps Features

### Experiment Tracking
```bash
mlflow ui  # View experiments on http://localhost:5000
```

### Model Registry
- Automatic model versioning
- Production model promotion
- Model metadata tracking

### CI/CD Pipelines
- Automated testing on push
- Model validation before deployment
- Automatic model deployment

## 🧪 Testing

```bash
pytest tests/ -v --cov=src
```

## 🐳 Docker

### Build Image
```bash
docker build -f docker/Dockerfile -t diabetic-model:latest .
```

### Run Container
```bash
docker run -p 8000:8000 diabetic-model:latest
```

### Docker Compose
```bash
docker-compose up
```

## 📡 API Usage

### Start API Server
```bash
python scripts/serve_model.py
```

### Make Predictions
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "age": 45,
        "bmi": 25.5,
        "glucose": 120,
        # ... other features
    }
)
print(response.json())
```

## 📚 Documentation

- [Data Dictionary](docs/DATA_DICTIONARY.md)
- [Model Architecture](docs/MODEL_ARCHITECTURE.md)
- [API Documentation](docs/API.md)
- [Contributing Guide](CONTRIBUTING.md)

## 📊 Model Performance

| Model | AUC-ROC | F1-Score | Accuracy |
|-------|---------|----------|----------|
| Logistic Regression | 0.82 | 0.78 | 0.80 |
| XGBoost | 0.88 | 0.85 | 0.86 |
| LightGBM | 0.89 | 0.86 | 0.87 |
| **CatBoost** | **0.91** | **0.88** | **0.89** |

## 🔐 Security & Privacy

- Input validation for all API requests
- HIPAA considerations for health data
- Secure model storage and versioning
- Audit logging for predictions

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details

## 👤 Author

**Sambit Mohanty**
- GitHub: [@sambit-mohanty2](https://github.com/sambit-mohanty2)

## 🙏 Acknowledgments

- Dataset: [UCI Machine Learning Repository]
- Inspiration: MLOps best practices and industry standards
- Community: Open source ML/MLOps tools

## 📞 Contact & Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Contact the maintainers

---

**Last Updated**: June 2026
