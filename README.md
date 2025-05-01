# Blood Glucose Forecaster

## Overview
This project focuses on forecasting blood glucose levels using machine learning techniques, specifically leveraging the PyTorch Forecasting library. The goal is to predict future glucose trends to aid in the management of diabetes, utilizing time-series data from continuous glucose monitoring (CGM) systems.

## Features
**Time-Series Forecasting:** Implements models to predict future blood glucose levels based on historical data.

**Model Training and Evaluation:** Provides scripts and notebooks for training models and evaluating their performance.

**Hyperparameter Tuning:** Includes configurations for conducting hyperparameter sweeps to optimize model performance.

**Data Analysis:** Offers tools for analyzing predictions and comparing them with actual glucose readings.

<!-- ## Repository Structure
blood_glucose_forecaster/
├── artifacts/                   # Saved model artifacts and checkpoints
├── blood-glucose-v2/            # Additional model versions and experiments
├── configs/                     # Configuration files for experiments
├── Project_Milestone.ipynb      # Milestone report notebook
├── analyze.ipynb                # Notebook for analyzing model predictions
├── pytorch_forecast.ipynb       # Main notebook for model training
├── sweep_config.yaml            # Configuration for hyperparameter sweeps
├── sweep_forecast.ipynb         # Notebook for running hyperparameter sweeps
├── test.py                      # Script for testing the trained model
├── test_predictions_with_truth.csv  # CSV file with test predictions and actual values
├── train.py                     # Script for training the model
└── README.md                    # Project documentation -->

## Getting Started
### Prerequisites
- Python 3.8 or higher
- _Recommended:_ Create a virtual environment using venv or conda

### Installation
1. Clone the Repository
````bash
`git clone https://github.com/rjaisinghani99/blood_glucose_forecaster.git
cd blood_glucose_forecaster
git checkout
```

2. Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

1. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage
### Clustering

### Neural Network
#### Training the Model
To train the model using the provided script:
```bash
python train.py
```

Alternatively, use the Jupyter notebook:
```bash
jupyter notebook pytorch_forecast.ipynb
```

### Hyperparameter Tuning
Conduct hyperparameter sweeps using the configuration file:
```bash
jupyter notebook sweep_forecast.ipynb
```
Ensure sweep_config.yaml is properly configured before running the sweep.

### Testing the Model
After training, evaluate the model's performance:
```bash
python test.py
```

This will generate predictions and save them to ```test_predictions_with_truth.csv```.

### Analyzing Results
Use the analysis notebook to visualize and interpret the model's predictions:
```bash
jupyter notebook analyze.ipynb
```

# Data
_Note: The dataset used for training and evaluation is not included in this repository. Please ensure you have access to appropriate blood glucose time-series data formatted for use with PyTorch Forecasting._

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments
Inspired by ongoing research in blood glucose prediction and time-series forecasting.

Utilizes the PyTorch Forecasting library for model implementation.

For any questions or support, please contact the repository maintainer.