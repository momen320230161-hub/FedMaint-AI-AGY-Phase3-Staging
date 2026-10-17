"""Synthetic acceptance tests for reusable Azure EDA summaries."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedmaint.data.azure import AzureDataError
from fedmaint.data.eda import analyze_azure


class AzureEDATests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.write("PdM_machines.csv", ["machineID", "model", "age"], [
            [1, "model1", 5], [2, "model2", 10],
        ])
        self.write("PdM_telemetry.csv",
                   ["datetime", "machineID", "volt", "rotate", "pressure", "vibration"], [
                       ["2015-01-01 00:00:00", 1, 1, 2, 5, 7],
                       ["2015-01-01 01:00:00", 2, 2, 4, 5, 8],
                       ["2015-01-01 01:00:00", 2, 2, 4, 5, 8],
                   ])
        self.write("PdM_errors.csv", ["datetime", "machineID", "errorID"], [
            ["2015-01-02 00:00:00", 1, "error1"],
        ])
        self.write("PdM_maint.csv", ["datetime", "machineID", "comp"], [
            ["2015-01-03 00:00:00", 2, "comp2"],
        ])
        self.write("PdM_failures.csv", ["datetime", "machineID", "failure"], [
            ["2015-01-04 00:00:00", 1, "comp3"],
        ])

    def write(self, filename, columns, records):
        with (self.directory / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(records)

    def test_counts_quality_models_and_events(self):
        result = analyze_azure(self.directory, sample_size=2)
        self.assertEqual(result["tables"]["telemetry"]["rows"], 3)
        self.assertEqual(result["tables"]["telemetry"]["exact_duplicate_rows"], 1)
        self.assertEqual(sum(result["tables"]["telemetry"]["missing_values"].values()), 0)
        self.assertEqual(result["machines_by_model"], {"model1": 1, "model2": 1})
        self.assertEqual(result["event_counts"]["errors"], {"error1": 1})
        self.assertEqual(result["event_counts"]["maint"], {"comp2": 1})
        self.assertEqual(result["event_counts"]["failures"], {"comp3": 1})
        self.assertEqual(result["monthly_counts"]["telemetry"], {"2015-01": 3})
        self.assertEqual(result["tables"]["telemetry"]["first_timestamp"].hour, 0)
        self.assertEqual(result["tables"]["telemetry"]["last_timestamp"].hour, 1)
        self.assertEqual(len(result["telemetry_plot_sample"]["volt"]), 2)

    def test_full_data_sensor_statistics_and_constant_correlation(self):
        result = analyze_azure(self.directory)
        self.assertAlmostEqual(result["sensor_stats"]["volt"]["mean"], 5 / 3)
        self.assertEqual(result["sensor_stats"]["volt"]["min"], 1)
        self.assertEqual(result["sensor_stats"]["volt"]["max"], 2)
        self.assertAlmostEqual(result["sensor_correlation"]["volt"]["rotate"], 1.0)
        self.assertIsNone(result["sensor_correlation"]["pressure"]["volt"])

    def test_deterministic_bounded_sample_and_invalid_size(self):
        first = analyze_azure(self.directory, sample_size=1, seed=123)
        second = analyze_azure(self.directory, sample_size=1, seed=123)
        self.assertEqual(first["telemetry_plot_sample"], second["telemetry_plot_sample"])
        self.assertEqual(len(first["telemetry_plot_sample"]["volt"]), 1)
        for size in (0, -1, True, 1.5):
            with self.subTest(size=size), self.assertRaises(ValueError):
                analyze_azure(self.directory, sample_size=size)

    def test_loader_validation_is_not_bypassed(self):
        self.write("PdM_errors.csv", ["datetime", "machineID", "errorID"], [
            ["not-a-timestamp", 1, "error1"],
        ])
        with self.assertRaisesRegex(AzureDataError, "PdM_errors.csv: CSV row 2"):
            analyze_azure(self.directory)


if __name__ == "__main__":
    unittest.main()
