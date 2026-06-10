from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mlops-diabetes-model",
    version="0.1.0",
    author="Sambit Mohanty",
    description="Diabetic Risk Prediction Model with MLOps Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sambit-mohanty2/mlops-diabetes-project",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "lightgbm>=4.0.0",
        "catboost>=1.2.0",
        "fastapi>=0.103.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.2.0",
        "mlflow>=2.7.0",
        "optuna>=3.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "mlops": [
            "mlflow>=2.7.0",
            "dvc>=3.27.0",
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "diabetes-train=scripts.train_model:main",
            "diabetes-evaluate=scripts.evaluate_model:main",
            "diabetes-serve=scripts.serve_model:main",
            "diabetes-predict=scripts.predict_batch:main",
        ],
    },
)
