# ⚡ Smart Electricity Consumption Predictor

An end-to-end beginner machine learning project that predicts **hourly office electricity consumption (kWh)** using **Linear Regression**. It covers the full workflow: generating data, exploring it, training and evaluating a model, saving it with Pickle, and serving predictions through a terminal script and a Streamlit web app.

> **Note:** The dataset is **synthetic** and generated for learning. It is not real smart-meter data, and the model's scores should not be read as real-world accuracy.

---

## Features

- Generates 2,000 synthetic hourly office records
- Exploratory data analysis (shape, types, missing values, statistics, correlations, scatter plots)
- Trains and evaluates a scikit-learn `LinearRegression` model (MAE, MSE, RMSE, R²)
- Saves the trained model and its feature order with Pickle
- Predicts from terminal input
- Interactive Streamlit web app

## Tech stack

Python · Pandas · NumPy · scikit-learn · Matplotlib · Pickle · Streamlit

## Input features and target

| Column | Meaning |
|---|---|
| `temperature_c` | Outdoor temperature in °C |
| `people_count` | Number of people present |
| `ac_hours` | Fraction of the hour the AC ran (0–1) |
| `computer_count` | Number of operating computers |
| `lighting_hours` | Fraction of the hour the lights ran (0–1) |
| `energy_consumption_kwh` | **Target:** electricity used in the hour (kWh) |

## Project structure

```
electricbill-prediction/
├── 01_create_data.py      # Generate electricity_data.csv
├── 02_explore_data.py     # Explore data, statistics, charts
├── 03_train_model.py      # Train, evaluate, save electricity_model.pkl
├── 04_predict.py          # Terminal predictions
├── 05_app.py              # Streamlit web app
├── requirements.txt
├── README.md
├── electricity_data.csv   # Created by script 01
└── electricity_model.pkl  # Created by script 03
```

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/AdwayRakshit/electricbill-prediction.git
cd electricbill-prediction

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows (Command Prompt)
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:

```
pandas
numpy
scikit-learn
matplotlib
streamlit
```

## How to run

Run the scripts **in order** from inside the project folder:

| Step | Command | Result |
|---|---|---|
| 1 | `python 01_create_data.py` | Creates `electricity_data.csv` (2,000 rows, 6 columns) |
| 2 | `python 02_explore_data.py` | Prints statistics and shows charts |
| 3 | `python 03_train_model.py` | Trains and evaluates the model, creates `electricity_model.pkl` |
| 4 | `python 04_predict.py` | Asks for office conditions and prints an estimate |
| 5 | `streamlit run 05_app.py` | Opens the web app at http://localhost:8501 |

If `streamlit` is not recognized, use `python -m streamlit run 05_app.py`.

## How it works

1. **Data creation.** Features are drawn from random ranges, and consumption is computed from a linear formula plus random noise (standard deviation 0.8 kWh).
2. **Exploration.** Checks for missing values, ranges, and relationships between features and the target.
3. **Training.** Data is split 80% / 20% into training and test sets. `LinearRegression` learns one coefficient per feature plus an intercept.
4. **Evaluation.** Predictions on the unseen test set are scored with MAE, MSE, RMSE and R².
5. **Saving.** The model and its feature names are saved together in `electricity_model.pkl`.
6. **Prediction.** The terminal script and the Streamlit app load the saved model and predict for new inputs without retraining.

## Example

Inputs: 34 °C, 12 people, AC 1.0, 8 computers, lighting 1.0

Expected estimate: about **9.7 kWh** for the hour.

## Evaluation metrics

| Metric | Meaning |
|---|---|
| MAE | Average absolute error, in kWh |
| MSE | Average squared error (kWh²) |
| RMSE | Square root of MSE, in kWh |
| R² | Share of variation in consumption explained by the model |

Because noise with a standard deviation of 0.8 kWh was added to the data, an RMSE near 0.8 is about the best any model can achieve here.

## Troubleshooting

| Problem | Fix |
|---|---|
| `electricity_data.csv` not found | Run script 01 first, from the project folder |
| `electricity_model.pkl` not found | Run script 03 after creating the dataset |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt` |
| `streamlit` command not found | Use `python -m streamlit run 05_app.py` |
| App shows old results after retraining | Restart the app or clear the Streamlit cache |

## Limitations and responsible use

- The data is synthetic and the target comes from a near-linear formula, so the strong scores demonstrate the workflow, not real-world accuracy.
- The model only saw temperatures of 10–40 °C, 0–30 people and 0–20 computers. Inputs far outside these ranges give unreliable estimates.
- Outputs are estimates, not measured meter readings.
- The terminal script does not validate every input range.
- **Only load Pickle files you trust.** Loading a malicious Pickle file can execute code. Use the same scikit-learn version for training and loading.

## Ideas for improvement

- Replace the synthetic data with real timestamped meter data
- Validate inputs in the terminal script
- Compare against a baseline and other models (e.g. Random Forest)
- Use cross-validation and time-based train/test splits
- Show coefficients and charts in the Streamlit app
- Deploy on Streamlit Community Cloud

## License

Educational project. Add a license of your choice (for example MIT) before publishing.
