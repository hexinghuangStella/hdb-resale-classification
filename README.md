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
├── tests/
│   ├── test_data_preparation.py
│   └── test_model_workflow.py
└── main.py
```

## Setup and Usage

Python 3.10 or later is required. Run the following commands from the project root:

```powershell
python -m venv .venv
# Activate the environment in Windows PowerShell:
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt

# Train both models; report training CV and validation metrics only:
python main.py

# Use a custom configuration:
python main.py --config src/config.yaml

# Train only one model:
python main.py --model decision_tree
python main.py --model logistic_regression

# Explicit final test evaluation after freezing model/parameter choices:
python main.py --model logistic_regression --final-test

# Open the exploratory notebook:
jupyter lab eda.ipynb
```

## Data

The original dataset, `resale_transactions_categorised.csv`, has been renamed to `data/data.csv`. Data files are excluded from version control by `.gitignore`; place the CSV at this path before running the project on another machine.

Required columns for training:

```text
month, flat_type, block, street_name, storey_range,
floor_area_sqm, remaining_lease,
town_id, flatm_id, price_category
```

The EDA notebook additionally uses `lease_commence_date`, `town_name`, and `flatm_name` from the original dataset for auditing and readable plots.

## Data Preparation

- Normalize column names and remove entirely empty rows and exact duplicates. Missing or unknown target labels raise an error.
- Correct negative lease commencement years by reversing their signs only when the result is an integer between 1900 and the transaction year. Unrecoverable negative years become missing. This repairs 8,446 records in the current cleaned dataset without changing the source CSV; this is an audit/EDA correction when the optional year column is present. The current baseline does not use commencement year, so prediction does not require it.
- Convert remaining lease duration to months, extract the transaction year and month, and convert storey ranges to their midpoints.
- Canonicalize `town_id` and `flatm_id` so integer, float, and numeric-string forms such as `17`, `17.0`, and `"17.0"` all become `"17"`. Missing values remain missing for training-fitted imputation; non-integer, negative, or malformed IDs raise an error.
- Treat `town_id` and `flatm_id` as categorical features. Exclude record `id` and redundant name columns, avoiding ID-to-name mappings learned from the entire dataset.
- Use the same stratified training, validation, and test split for both models: **80% / 10% / 10%**, with a default random seed of **42**.
- Fit missing-value imputation and one-hot encoding on the training set only. Unknown categories are supported during prediction.
- Standardize numeric features for logistic regression; skip standardization for the decision tree. Each model fits its own preprocessing pipeline.

The baseline excludes `lease_commence_date` because of its strong overlap with `remaining_lease_months`. It retains remaining lease, sale year, and sale month. Commencement year is optional for baseline prediction and is retained in the source data for cleaning audits and EDA (`engineer_features(df, include_eda=True)`).

## Models and Configuration

Configure each model under `models` in `src/config.yaml`. Each entry specifies a `type` and a `params` mapping, passed directly to the corresponding scikit-learn classifier.

Supported types are `logistic_regression` and `decision_tree`. Each classifier inherits `split.random_state` unless its own `random_state` parameter is provided.

The decision tree starts with parameters supplied from a historical notebook experiment. They have not been re-tuned for the current feature set:

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

The original experiment reported a best cross-validation ROC AUC of **0.9709776364989262**. This is a historical score, not a current result. The training entry point now independently evaluates the configured pipeline using training-only cross-validation and validation metrics; it does not run a hyperparameter search.

The original search used names such as `classifier__max_depth`. Remove the `classifier__` prefix here because these parameters are passed directly to the classifier rather than to the enclosing pipeline.

Use training-set cross-validation and the validation set for model and parameter selection. Configure the number of stratified, shuffled folds with `cross_validation.folds` (default: 5); the fold seed comes from `split.random_state`, so both models use the same folds. Imputation, scaling, encoding, and classifier fitting all happen inside each fold. Reported standard deviations use the population convention (`ddof=0`). The final saved pipeline is then fitted on the full training partition only.

Test labels are not scored by default. Enable held-out evaluation explicitly with `--final-test` after freezing the model and parameter choices. Merely changing configuration cannot enable test scoring. Earlier development runs have already inspected this test partition, so it should not be described as a completely untouched final test set; the switch prevents routine future test inspection rather than undoing that history.

## Evaluation and Saved Artifacts

By default each model reports training CV accuracy, F1, and ROC AUC (individual fold scores, mean, and standard deviation), plus validation accuracy, F1, ROC AUC, a confusion matrix, and a classification report. Test metrics are generated only with `--final-test`. Confusion matrix rows represent actual classes and columns represent predicted classes, ordered as `[Below Median, Above Median]`.

Artifacts are saved separately under `artifacts/<model_name>/`:

| File | Contents |
| --- | --- |
| `pipeline.joblib` | Fitted preprocessing and classifier pipeline; recommended for prediction. |
| `preprocessor.joblib` | Fitted preprocessing pipeline saved separately. |
| `model.joblib` | Fitted classifier; inputs must first be transformed by its matching preprocessor. |
| `metrics.json` | Training CV and validation results, model name, sample counts, and `test_evaluated`; `test` metrics appear only when explicitly enabled. |
| `config.yaml` | Data path, split/CV settings, selected model configuration, and the `final_test` flag used for the run. |

Running a model again overwrites its artifacts. A default run replaces any previous test scores in that model's `metrics.json`; a test row count is partition metadata, not a test evaluation. Legacy files directly under `artifacts/` are no longer updated. Generated artifacts are excluded from version control.

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

## Regression Checks

Run `python -m unittest discover -s tests -v` to check lease-year correction, canonical and invalid IDs, and prediction-label/probability invariance across integer, float, and string IDs for both models. A synthetic-data CLI test checks that default runs skip held-out scoring, `--final-test` enables it, and subsequent default runs remove stale test metrics. These checks do not evaluate the real held-out dataset.
