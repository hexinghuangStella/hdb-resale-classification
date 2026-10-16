"""Regression tests for canonical IDs and the final-test evaluation gate."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

import main as training_cli
from src.data_preparation import engineer_features, normalize_category_id
from src.model_training import build_model


def sample_data():
    rows = np.arange(100)
    return pd.DataFrame({
        "month": ["2018-06"] * 100, "floor_area_sqm": 60 + rows,
        "remaining_lease": ["70 years 03 months"] * 100,
        "storey_range": ["04 TO 06"] * 100, "flat_type": ["4 ROOM"] * 100,
        "block": ["101"] * 100, "street_name": ["TEST STREET"] * 100,
        "town_id": rows % 3 + 1, "flatm_id": rows % 2 + 1,
        "price_category": np.where(rows % 2, "Above Median", "Below Median"),
    })


class CategoryIdTests(unittest.TestCase):
    def test_canonical_and_missing_ids(self):
        for value in [17, 17.0, "17", "17.0", " 017.00 "]:
            self.assertEqual(normalize_category_id(value), "17")
        for value in [None, np.nan, pd.NA, ""]:
            self.assertTrue(pd.isna(normalize_category_id(value)))
        for value in ["bad", 1.5, -1, float("inf")]:
            with self.assertRaises(ValueError):
                normalize_category_id(value)

    def test_prediction_invariant_to_id_representation(self):
        raw = sample_data()
        X = engineer_features(raw)
        y = raw.price_category.eq("Above Median").astype(int)
        for model_type in ["logistic_regression", "decision_tree"]:
            with self.subTest(model=model_type):
                model = build_model({"type": model_type, "params": {}}, 42).fit(X, y)
                expected = model.predict(X)
                expected_proba = model.predict_proba(X)
                for dtype in [int, float, str]:
                    variant = raw.copy()
                    for col in ["town_id", "flatm_id"]:
                        variant[col] = variant[col].astype(dtype)
                    actual_X = engineer_features(variant)
                    pd.testing.assert_frame_equal(actual_X, X)
                    np.testing.assert_array_equal(model.predict(actual_X), expected)
                    np.testing.assert_allclose(model.predict_proba(actual_X), expected_proba)
                variant = raw.copy()
                for col in ["town_id", "flatm_id"]:
                    variant[col] = variant[col].map(lambda x: f" {x}.0 ")
                np.testing.assert_allclose(model.predict_proba(engineer_features(variant)), expected_proba)


class EvaluationGateTests(unittest.TestCase):
    def test_default_and_explicit_final_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.csv"
            sample_data().to_csv(data_path, index=False)
            config = {
                "data_path": str(data_path), "output_dir": str(root / "artifacts"),
                "split": {"validation_size": .1, "test_size": .1, "random_state": 42},
                "cross_validation": {"folds": 2},
                "models": {"decision_tree": {"type": "decision_tree", "params": {"max_depth": 3}}},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            for final_test in [True, False]:
                with self.subTest(final_test=final_test):
                    argv = ["main.py", "--config", str(config_path)] + (["--final-test"] if final_test else [])
                    with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), patch.object(
                        training_cli, "evaluate", wraps=training_cli.evaluate
                    ) as evaluate_spy:
                        training_cli.main()
                    self.assertEqual(evaluate_spy.call_count, 2 if final_test else 1)
                    report = json.loads((root / "artifacts/decision_tree/metrics.json").read_text())
                    self.assertEqual(report["test_evaluated"], final_test)
                    self.assertEqual("test" in report, final_test)
                    self.assertIn("validation", report)
                    self.assertEqual(report["cross_validation"]["folds"], 2)
                    self.assertEqual(len(report["cross_validation"]["metrics"]["roc_auc"]["fold_scores"]), 2)
                    # Validation evaluation must have no overlap with training rows.
                    self.assertEqual(len(evaluate_spy.call_args_list[0].args[1]), 10)


if __name__ == "__main__":
    unittest.main()
