"""Run the configured classification experiment from any working directory."""
import argparse
import json
from pathlib import Path

import joblib
import yaml

from src.data_preparation import load_data, split_data
from src.model_training import build_model, evaluate, evaluate_cross_validation

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Train the HDB resale price classifier")
    parser.add_argument("--config", type=Path, default=ROOT / "src/config.yaml")
    parser.add_argument("--model", help="Train one configured model; by default train all models")
    parser.add_argument("--final-test", action="store_true", help="Explicitly evaluate the held-out test set after model selection")
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    models = config["models"]
    if not isinstance(models, dict) or not models:
        parser.error("config.models must be a nonempty mapping")
    if args.model and args.model not in models:
        parser.error(f"Unknown model {args.model!r}; choose from {', '.join(models)}")
    selected = {args.model: models[args.model]} if args.model else models
    for name in selected:
        if not name or name in {".", ".."} or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
            parser.error(f"Invalid model name: {name!r}; use letters, digits, underscores or hyphens")
    pipelines = {name: build_model(settings, config["split"]["random_state"]) for name, settings in selected.items()}
    df = load_data(ROOT / config["data_path"])
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, config["split"])
    for name, model in pipelines.items():
        cv_metrics = evaluate_cross_validation(model, X_train, y_train,
                                               config.get("cross_validation", {"folds": 5}),
                                               config["split"]["random_state"])
        model.fit(X_train, y_train)
        metrics = {
            "model": name,
            "rows": {"cleaned": len(df), "train": len(y_train), "validation": len(y_val), "test": len(y_test)},
            "cross_validation": cv_metrics,
            "validation": evaluate(model, X_val, y_val),
            "test_evaluated": args.final_test,
        }
        if args.final_test:
            metrics["test"] = evaluate(model, X_test, y_test)
        output = ROOT / config["output_dir"] / name
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output / "pipeline.joblib")
        joblib.dump(model.named_steps["preprocessing"], output / "preprocessor.joblib")
        joblib.dump(model.named_steps["classifier"], output / "model.joblib")
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        run_config = {**config, "models": {name: selected[name]}, "final_test": args.final_test}
        (output / "config.yaml").write_text(yaml.safe_dump(run_config), encoding="utf-8")
        print(f"{name}: training CV ROC AUC={cv_metrics['metrics']['roc_auc']['mean']:.4f} "
              f"(+/- {cv_metrics['metrics']['roc_auc']['std']:.4f}), validation F1={metrics['validation']['f1']:.4f}; saved to {output}")
        if args.final_test:
            print(f"{name}: final test F1={metrics['test']['f1']:.4f}")


if __name__ == "__main__":
    main()
