"""Synthetic tests for the Azure CSV readers."""
import csv
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedmaint.data.azure import AzureDataError, iter_azure_rows, load_machines, validate_azure_files


HEADERS = {
    "telemetry": ("PdM_telemetry.csv", ["datetime", "machineID", "volt", "rotate", "pressure", "vibration"]),
    "errors": ("PdM_errors.csv", ["datetime", "machineID", "errorID"]),
    "maint": ("PdM_maint.csv", ["datetime", "machineID", "comp"]),
    "failures": ("PdM_failures.csv", ["datetime", "machineID", "failure"]),
    "machines": ("PdM_machines.csv", ["machineID", "model", "age"]),
}
ROWS = {
    "telemetry": ["2015-01-01 06:00:00", "1", "176.2", "418.5", "113.0", "45.0"],
    "errors": ["2015-01-03 07:00:00", "1", "error1"],
    "maint": ["2014-06-01 06:00:00", "1", "comp2"],
    "failures": ["2015-01-05 06:00:00", "1", "comp4"],
    "machines": ["1", "model3", "18"],
}


class AzureDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        for table in HEADERS:
            self.write_table(table, [ROWS[table]])

    def write_table(self, table, rows, header=None):
        filename, standard_header = HEADERS[table]
        with (self.directory / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(standard_header if header is None else header)
            writer.writerows(rows)

    def assert_row_error(self, table, filename, row_number, message):
        with self.assertRaises(AzureDataError) as caught:
            list(iter_azure_rows(table, self.directory))
        value = str(caught.exception)
        self.assertIn(filename, value)
        self.assertIn(f"CSV row {row_number}", value)
        self.assertIn(message, value)

    def test_valid_tables_and_typed_values(self):
        self.assertEqual(load_machines(self.directory)[1]["age"], 18)
        telemetry = list(iter_azure_rows("telemetry", self.directory))
        self.assertEqual(len(telemetry), 1)
        self.assertEqual(telemetry[0]["datetime"], datetime(2015, 1, 1, 6))
        self.assertIsInstance(telemetry[0]["volt"], float)
        self.assertEqual(telemetry[0]["machineID"], 1)
        for table in ("errors", "maint", "failures"):
            self.assertEqual(len(list(iter_azure_rows(table, self.directory))), 1)
        self.assertEqual(
            validate_azure_files(self.directory),
            {"machines": 1, "telemetry": 1, "errors": 1, "maint": 1, "failures": 1},
        )

    def test_missing_and_duplicate_headers(self):
        filename, header = HEADERS["telemetry"]
        self.write_table("telemetry", [ROWS["telemetry"][:-1]], header=header[:-1])
        self.assert_row_error("telemetry", filename, 1, "expected columns")
        duplicate = list(header)
        duplicate[-1] = "volt"
        self.write_table("telemetry", [ROWS["telemetry"]], header=duplicate)
        self.assert_row_error("telemetry", filename, 1, "expected columns")

    def test_schema_validation_covers_every_table(self):
        for table, (filename, header) in HEADERS.items():
            with self.subTest(table=table):
                self.write_table(table, [ROWS[table][:-1]], header=header[:-1])
                self.assert_row_error(table, filename, 1, "expected columns")
                self.write_table(table, [ROWS[table]])

    def test_wrong_field_count(self):
        filename, _ = HEADERS["errors"]
        self.write_table("errors", [ROWS["errors"] + ["extra"]])
        self.assert_row_error("errors", filename, 2, "expected 3 fields")

    def test_invalid_timestamp_and_calendar_date(self):
        filename, _ = HEADERS["errors"]
        for bad in ("2015/01/03 07:00:00", "2015-02-30 07:00:00", ""):
            with self.subTest(bad=bad):
                row = list(ROWS["errors"])
                row[0] = bad
                self.write_table("errors", [row])
                self.assert_row_error("errors", filename, 2, "datetime")

    def test_invalid_sensor_and_machine_types(self):
        filename, _ = HEADERS["telemetry"]
        for bad in ("not-a-number", "nan", "inf", ""):
            with self.subTest(sensor=bad):
                row = list(ROWS["telemetry"])
                row[2] = bad
                self.write_table("telemetry", [row])
                self.assert_row_error("telemetry", filename, 2, "volt")
        self.write_table("telemetry", [ROWS["telemetry"]])
        row = list(ROWS["machines"])
        row[0] = "1.5"
        self.write_table("machines", [row])
        self.assert_row_error("machines", "PdM_machines.csv", 2, "machineID")

    def test_invalid_age_and_empty_category(self):
        self.write_table("machines", [["1", "model3", "-1"]])
        self.assert_row_error("machines", "PdM_machines.csv", 2, "age")
        self.write_table("machines", [ROWS["machines"]])
        self.write_table("maint", [["2014-06-01 06:00:00", "1", "  "]])
        self.assert_row_error("maint", "PdM_maint.csv", 2, "comp")

    def test_invalid_references_in_all_event_tables_and_telemetry(self):
        for table in ("telemetry", "errors", "maint", "failures"):
            with self.subTest(table=table):
                row = list(ROWS[table])
                row[1] = "999"
                self.write_table(table, [row])
                self.assert_row_error(table, HEADERS[table][0], 2, "absent from PdM_machines.csv")
                self.write_table(table, [ROWS[table]])

    def test_duplicate_machine_identifier_reports_second_record(self):
        self.write_table("machines", [ROWS["machines"], ROWS["machines"]])
        self.assert_row_error("machines", "PdM_machines.csv", 3, "duplicate machineID")

    def test_streaming_delays_later_row_validation(self):
        first = list(ROWS["telemetry"])
        later = list(ROWS["telemetry"])
        later[2] = "not-a-number"
        self.write_table("telemetry", [first, later])
        rows = iter_azure_rows("telemetry", self.directory)
        self.assertEqual(next(rows)["volt"], 176.2)
        with self.assertRaises(AzureDataError) as caught:
            next(rows)
        self.assertIn("PdM_telemetry.csv: CSV row 3", str(caught.exception))

    def test_missing_source_file_identifies_path(self):
        (self.directory / "PdM_failures.csv").unlink()
        with self.assertRaises(AzureDataError) as caught:
            list(iter_azure_rows("failures", self.directory))
        self.assertIn("PdM_failures.csv", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
