import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import observe_agenda_run


class ServiceTransactionHygieneTests(unittest.TestCase):
    def test_observer_rolls_back_after_a_successful_tick(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "observation.jsonl"
            args = [
                "observe_agenda_run.py",
                "--agenda",
                "11",
                "--hours",
                "0",
                "--out",
                str(output),
            ]
            rollback = mock.Mock()
            with (
                mock.patch("sys.argv", args),
                mock.patch.object(
                    observe_agenda_run,
                    "snapshot",
                    return_value={"at": "2026-08-12T00:00:00+00:00"},
                ),
                mock.patch.object(observe_agenda_run.db, "rollback", rollback),
            ):
                self.assertEqual(observe_agenda_run.main(), 0)

            rollback.assert_called_once_with()
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
