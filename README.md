# 🚲 Bike Sharing Demand Prediction

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A small, end-to-end machine learning pipeline that predicts **hourly bike rental demand** from calendar and weather information. It covers data download, preprocessing, model training, and a command-line tool for making predictions.

The data comes from the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (Capital Bikeshare, Washington D.C., 2011–2012). The model is a scikit-learn `RandomForestRegressor`.

## Table of Contents

- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Features](#features)
- [Model & Evaluation](#model--evaluation)
- [Limitations & Next Steps](#limitations--next-steps)
- [License](#license)

## Project Structure

```
.
├── src/
│   ├── fetch_data.sh     # Downloads and unzips the raw dataset
│   ├── preprocess.py     # Cleans data and creates train/test splits
│   ├── train.py          # Trains the model, reports metrics, saves it
│   └── predict.py        # CLI for single-row predictions
├── config/
│   ├── requirements.txt  # Python dependencies
│   └── model.pkl         # Trained model (generated, git-ignored)
├── data/
│   ├── raw/              # Downloaded dataset (generated, git-ignored)
│   └── processed/        # train.csv / test.csv (generated, git-ignored)
├── .gitignore
├── LICENSE
└── README.md
```

> Datasets and trained model files are git-ignored. You regenerate them by running the pipeline below.

## Getting Started

### Prerequisites

- Python 3.9 or newer
- `wget` and `unzip` (used by the data download script)

### Installation

```bash
git clone https://github.com/gds-workshop/cli-ml-data-pipeline.git
cd cli-ml-data-pipeline

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r config/requirements.txt
```

## Usage

Run the pipeline from the **repository root**, in order. All paths in the scripts are relative to it.

### 1. Download the data

```bash
mkdir -p data/raw
bash src/fetch_data.sh
```

This downloads the dataset zip, extracts it into `data/raw/` (including `hour.csv`, which is used here), and removes the zip.

### 2. Preprocess

```bash
python src/preprocess.py
```

Drops non-predictive and leaky columns, removes missing values, and writes an 80/20 train/test split to `data/processed/train.csv` and `data/processed/test.csv`.

### 3. Train

```bash
python src/train.py
```

Fits the Random Forest, prints MSE and R² on the test set, and saves the model to `config/model.pkl`.

### 4. Predict

```bash
python src/predict.py \
  --season 3 --yr 1 --mnth 7 --hr 17 \
  --holiday 0 --weekday 3 --workingday 1 \
  --weathersit 1 --temp 0.7 --atemp 0.68 \
  --hum 0.45 --windspeed 0.15
```

Example output:

```
🚀 [Prediction Result] Estimated Demand Count: 815 bikes/hour
```

*(Your exact number will vary with the trained model.)*

## Features

All arguments to `predict.py` are required.

| Flag | Type | Description |
|------|------|-------------|
| `--season` | int | 1 = spring, 2 = summer, 3 = fall, 4 = winter |
| `--yr` | int | Year offset: 0 = 2011, 1 = 2012 |
| `--mnth` | int | Month, 1–12 |
| `--hr` | int | Hour of day, 0–23 |
| `--holiday` | int | 1 if holiday, else 0 |
| `--weekday` | int | Day of week, 0–6 |
| `--workingday` | int | 1 if neither weekend nor holiday, else 0 |
| `--weathersit` | int | Weather rating, 1 (clear) to 4 (heavy rain/snow) |
| `--temp` | float | Normalized temperature, 0.0–1.0 |
| `--atemp` | float | Normalized "feels like" temperature, 0.0–1.0 |
| `--hum` | float | Normalized humidity, 0.0–1.0 |
| `--windspeed` | float | Normalized wind speed, 0.0–1.0 |

**Target:** `cnt` — total bike rentals in that hour.

## Model & Evaluation

- **Algorithm:** `RandomForestRegressor` (100 trees, `random_state=42`)
- **Split:** random 80% train / 20% test
- **Metrics:** Mean Squared Error (MSE) and R², printed at the end of `train.py`
- **Leakage prevention:** `casual` and `registered` are dropped because they sum exactly to the target `cnt`. `instant` (row index) and `dteday` (date string) are also removed.

## Limitations & Next Steps

- The train/test split is random, so it can overstate real-world performance for time-series data. A chronological split would be a more honest evaluation.
- No hyperparameter tuning or cross-validation yet.
- Pickle files should only be loaded from sources you trust.
- Ideas: add feature importance plots (`matplotlib` is already in the requirements), compare against gradient boosting, and add a `Makefile` to run the whole pipeline in one command.

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

Dataset: Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.

@article{
	year={2013},
	issn={2192-6352},
	journal={Progress in Artificial Intelligence},
	doi={10.1007/s13748-013-0040-3},
	title={Event labeling combining ensemble detectors and background knowledge},
	url={http://dx.doi.org/10.1007/s13748-013-0040-3},
	publisher={Springer Berlin Heidelberg},
	keywords={Event labeling; Event detection; Ensemble learning; Background knowledge},
	author={Fanaee-T, Hadi and Gama, Joao},
	pages={1-15}
}
