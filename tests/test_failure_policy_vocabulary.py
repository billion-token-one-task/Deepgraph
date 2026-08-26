from __future__ import annotations

import unittest

from meta_harness.failure_policy import measured_nothing


class LaneDependencyVocabularyTests(unittest.TestCase):
    """A lane without the packages says nothing about the science.

    The policy already reasoned about this -- a package pre-provisioned on one
    host and absent on another is a property of the host -- but only under the
    word "dependency_install_blocked". The runner reports it as
    "dependency_missing", so three pilot attempts were charged to the science
    budget for a host that had torch and no transformers.
    """

    def test_the_runners_own_word_is_recognised(self):
        for reason in (
            "dependency_missing",
            'RUNNER_ERROR: {"reason_code": "dependency_missing", '
            '"detail": "dependency_missing:datasets"}',
        ):
            with self.subTest(reason=reason[:40]):
                self.assertTrue(measured_nothing(reason))

    def test_a_scientific_failure_still_spends_the_science_budget(self):
        for reason in (
            "candidate scored below baseline",
            "metric regressed on the held-out split",
        ):
            with self.subTest(reason=reason):
                self.assertFalse(measured_nothing(reason))


if __name__ == "__main__":
    unittest.main()
