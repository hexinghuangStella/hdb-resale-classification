# HDB Resale Price Classification

Predict whether an HDB resale transaction is priced above or below the median using property attributes. The target is `price_category`: **Below Median = 0**, **Above Median = 1**.

## Project Structure

```text
root/
├── eda.ipynb
├── README.md
├── requirements.txt
├── data/
│   └── data.csv
├── src/
│   ├── data_preparation.py
│   ├── model_training.py
│   └── config.yaml
└── main.py
```

## Setup and Usage

Python 3.10 or later is required. Run the following commands from the project root:

```powershell
python -m venv .venv
# Activate the environment in Windows PowerShell:
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt

# Train both configured models:
python main.py

# Use a custom configuration:
python main.py --config src/config.yaml

# Train only one model:
python main.py --model decision_tree
python main.py --model logistic_regression

# Open the exploratory notebook:
jupyter lab eda.ipynb
```

## Data

The original dataset, `resale_transactions_categorised.csv`, has been renamed to `data/data.csv`. Data files are excluded from version control by `.gitignore`; place the CSV at this path before running the project on another machine.

Required columns:

```text
month, flat_type, block, street_name, storey_range,
floor_area_sqm, lease_commence_date, remaining_lease,
town_id, flatm_id, price_category
```

## Data Preparation

- Normalize column names and remove entirely empty rows and exact duplicates. Missing or unknown target labels raise an error.
- Convert remaining lease duration to months, extract the transaction year and month, and convert storey ranges to their midpoints.
- Treat `town_id` and `flatm_id` as categorical features. Exclude record `id` and redundant name columns, avoiding ID-to-name mappings learned from the entire dataset.
- Use the same stratified training, validation, and test split for both models: **80% / 10% / 10%**, with a default random seed of **42**.
- Fit missing-value imputation and one-hot encoding on the training set only. Unknown categories are supported during prediction.
- Standardize numeric features for logistic regression; skip standardization for the decision tree. Each model fits its own preprocessing pipeline.

## Models and Configuration

Configure each model under `models` in `src/config.yaml`. Each entry specifies a `type` and a `params` mapping, passed directly to the corresponding scikit-learn classifier.

Supported types are `logistic_regression` and `decision_tree`. Each classifier inherits `split.random_state` unless its own `random_state` parameter is provided.

The decision tree uses the best parameters supplied from the notebook experiment:

```yaml
models:
  decision_tree:
    type: decision_tree
    params:
      criterion: gini
      max_depth: 20
      min_samples_leaf: 10
      min_samples_split: 50
```

The original experiment reported a best cross-validation ROC AUC of **0.9709776364989262**. This score has not been reproduced through cross-validation in the current training entry point, which independently reports validation and test metrics.

The original search used names such as `classifier__max_depth`. Remove the `classifier__` prefix here because these parameters are passed directly to the classifier rather than to the enclosing pipeline.

Use cross-validation on the training set or the validation set for model and parameter selection. Reserve the test set for final evaluation.

## Evaluation and Saved Artifacts

Each model reports accuracy, F1, ROC AUC, a confusion matrix, and a classification report for both validation and test sets. Confusion matrix rows represent actual classes and columns represent predicted classes, ordered as `[Below Median, Above Median]`.

Artifacts are saved separately under `artifacts/<model_name>/`:

| File | Contents |
| --- | --- |
| `pipeline.joblib` | Fitted preprocessing and classifier pipeline; recommended for prediction. |
| `preprocessor.joblib` | Fitted preprocessing pipeline saved separately. |
| `model.joblib` | Fitted classifier; inputs must first be transformed by its matching preprocessor. |
| `metrics.json` | Validation and test metrics, model name, and sample counts. |
| `config.yaml` | Data path, split settings, and selected model configuration for the run. |

Running a model again overwrites its artifacts. Legacy files directly under `artifacts/` are no longer updated. Generated artifacts are excluded from version control.

## Prediction

Run this example from the project root after training:

```python
import joblib
import pandas as pd
from src.data_preparation import engineer_features

model = joblib.load("artifacts/decision_tree/pipeline.joblib")
raw = pd.read_csv("data/data.csv").head()
predictions = model.predict(engineer_features(raw))  # No target column required.
```

Use `artifacts/logistic_regression/pipeline.joblib` to load the logistic regression pipeline instead.

## Review Notes and Limitations

The refactoring fills the previously empty training modules, configuration, dependency list, and README. It excludes record IDs from model features, validates target labels, and makes cleaning and splitting functions reusable from the EDA notebook. Exploratory display-name cleaning remains in the notebook and does not affect training.

The random holdout split measures performance within the same data distribution. For future transaction prediction, use a chronological split and verify how and when the median price threshold was calculated. The available CSV cannot establish whether target generation used future information.

Only exact duplicates are removed; similar records with different IDs are not assumed to represent the same transaction. Dependencies use compatible version ranges rather than a lock file. Pin the installed versions when reproducing experiments.
