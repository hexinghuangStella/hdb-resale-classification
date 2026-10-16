"""Training-only preprocessing, classification, and held-out evaluation."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data_preparation import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_model(config, random_state):
    model_type = config["type"]
    params = dict(config.get("params", {}))
    params.setdefault("random_state", random_state)
    if model_type == "logistic_regression":
        params.setdefault("solver", "liblinear")
        classifier = LogisticRegression(**params)
    elif model_type == "decision_tree":
        classifier = DecisionTreeClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type!r}")
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler() if model_type == "logistic_regression" else "passthrough"),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def evaluate_cross_validation(model, X, y, config, random_state):
    """Clone and fit the entire pipeline inside each training-only CV fold."""
    folds = config["folds"]
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("cross_validation.folds must be an integer >= 2")
    if y.value_counts().min() < folds:
        raise ValueError("Each training class must have at least one sample per CV fold")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = cross_validate(model, X, y, cv=splitter,
                            scoring=["accuracy", "f1", "roc_auc"],
                            n_jobs=1, error_score="raise")
    return {
        "folds": folds, "shuffle": True, "random_state": random_state,
        "metrics": {metric: {
            "mean": float(scores[f"test_{metric}"].mean()),
            "std": float(scores[f"test_{metric}"].std(ddof=0)),
            "fold_scores": scores[f"test_{metric}"].tolist(),
        } for metric in ["accuracy", "f1", "roc_auc"]},
    }


def evaluate(model, X, y):
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, prediction, labels=[0, 1], target_names=["Below Median", "Above Median"], output_dict=True, zero_division=0),
    }
