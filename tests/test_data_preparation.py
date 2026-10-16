"""Regression checks for lease-year sign correction."""
import unittest

import numpy as np
import pandas as pd

from src.data_preparation import clean_data, correct_lease_year, engineer_features


class LeaseYearTests(unittest.TestCase):
    def test_bounds_missing_and_idempotence(self):
        years = pd.Series([-1995, 1995, -2020, -1800, -1995.5, np.nan, -1990])
        months = pd.Series(["2016-12"] * 6 + ["invalid"])
        actual = correct_lease_year(years, months)
        expected = pd.Series([1995., 1995., np.nan, np.nan, np.nan, np.nan, np.nan])
        pd.testing.assert_series_equal(actual, expected)
        pd.testing.assert_series_equal(correct_lease_year(actual, months), actual)

    def test_training_and_raw_prediction_use_same_correction(self):
        raw = pd.DataFrame({
            "month": ["2016-12"], "lease_commence_date": [-1995],
            "floor_area_sqm": [90], "remaining_lease": ["78"],
            "storey_range": ["01 TO 03"], "flat_type": ["4 ROOM"],
            "block": ["1"], "street_name": ["TEST"], "town_id": [1],
            "flatm_id": [1], "price_category": ["Below Median"],
        })
        cleaned = clean_data(raw)
        self.assertEqual(cleaned.loc[0, "lease_commence_date"], 1995)
        self.assertEqual(raw.loc[0, "lease_commence_date"], -1995)
        pd.testing.assert_frame_equal(engineer_features(raw), engineer_features(cleaned))
        self.assertNotIn("lease_commence_date", engineer_features(raw).columns)
        pd.testing.assert_frame_equal(engineer_features(raw), engineer_features(raw.drop(columns="lease_commence_date")))
        self.assertEqual(engineer_features(raw, include_eda=True).loc[0, "lease_commence_date"], 1995)


if __name__ == "__main__":
    unittest.main()
