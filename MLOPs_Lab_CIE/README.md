# MLOPs Lab CIE

## Project Overview
This repository contains the implementation of an MLOps pipeline for managing machine learning workflows. The project includes data preprocessing, model training, monitoring, and retraining, along with API endpoints for predictions and traffic simulation.

## Directory Structure
```
requirements.txt       # Python dependencies
src/                   # Source code for the project
  api.py               # API for serving predictions
  monitor.py           # Monitoring script
  retrain.py           # Script for retraining the model
  simulate_traffic.py  # Traffic simulation script
  train.py             # Model training script
data/                  # Data files
  new_data.csv         # New data for predictions
  training_data.csv    # Training dataset
logs/                  # Logs for predictions and monitoring
  predictions.jsonl    # JSONL file with prediction logs
mlruns/                # MLflow tracking directory
models/                # Directory for storing trained models
results/               # Results of the pipeline steps
  step1_s1.json        # Step 1 results
  step2_s4.json        # Step 2 results
  step3_s5.json        # Step 3 results
  step4_s8.json        # Step 4 results
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MLOPs_Lab_CIE
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Training the Model
Run the training script to train the model:
```bash
python src/train.py
```

#### Serving Predictions
Start the API server:
```bash
python src/api.py
```
The API will be available at `http://localhost:5000`.

#### Monitoring
Run the monitoring script to track model performance:
```bash
python src/monitor.py
```

#### Retraining
Trigger the retraining process:
```bash
python src/retrain.py
```

#### Simulating Traffic
Simulate traffic for the API:
```bash
python src/simulate_traffic.py
```

## MLflow Tracking
The `mlruns/` directory contains MLflow tracking data for experiments. Metrics such as MAE, MAPE, R2, and RMSE are logged for each run.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- [MLflow](https://mlflow.org/) for experiment tracking
- [Flask](https://flask.palletsprojects.com/) for the API framework
- [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data processing