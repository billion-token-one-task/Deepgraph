"""A run that already failed must report why, not "artifact collection failed".

Request 66 (2026-08-20) exited 78 with a clean, actionable message:

    dependency_install_blocked: reviewed runtime must be pre-provisioned;
    missing pot

The remote had correctly refused to install a package at runtime. But the
executor went on to collect artifacts that a failed run never produced, spent
four download retries on them, and then raised "Colab artifact collection
failed after 4 attempts: File /content/deepgraph-artifacts.tar.gz not found".
That exception is caught into a `transport:` failure reason.

The misclassification is not cosmetic. Transport-class failures deliberately
draw on a larger infrastructure retry budget (they measured nothing, so they
say nothing about the science) -- so a deterministic environment mismatch was
handed the budget reserved for problems a different lane might fix. It would
be retried until that budget ran out, every attempt failing identically.

The exit code is the cause; missing artifacts are the consequence.
"""

import unittest

from meta_harness.failure_policy import is_transport_class_failure


class FailedRunReportsItsOwnExitTests(unittest.TestCase):
    OBSERVED_MASK = (
        "transport:ColabCLIError:Colab artifact collection failed after 4 "
        'attempts: File "/content/deepgraph-artifacts.tar.gz" not found.'
    )

    def test_the_mask_was_classified_as_transport(self):
        # why this mattered: the wrong retry budget, not just a wrong label
        self.assertTrue(is_transport_class_failure(self.OBSERVED_MASK))

    def test_the_truthful_reason_is_not_transport(self):
        self.assertFalse(is_transport_class_failure("experiment_exit_78"))

    def test_collection_is_skipped_when_the_run_already_failed(self):
        import inspect

        from meta_harness.backends import colab_cli

        source = inspect.getsource(colab_cli.ColabCLIExecutor.run_request)
        self.assertIn(
            "collected = returncode == 0 or bool(embedded_archive)", source
        )
        # and the artifacts of a SUCCESSFUL run are still collected with
        # retries -- run 163 lost 2.8 hours to a single failed download
        self.assertIn("for attempt in range(1, 5):", source)

    def test_an_embedded_archive_still_rescues_a_failed_run(self):
        # a non-zero exit that still emitted its archive over stdout keeps it:
        # the run measured something even though it ended badly
        import inspect

        from meta_harness.backends import colab_cli

        source = inspect.getsource(colab_cli.ColabCLIExecutor.run_request)
        self.assertIn("or bool(embedded_archive)", source)


if __name__ == "__main__":
    unittest.main()


class DependencyPresenceTests(unittest.TestCase):
    """Ask the installer what is installed; do not guess an import name.

    The remote preflight mapped distribution names to import names with a
    hand-maintained dict of five entries, falling back to
    name.replace("-", "_"). POT installs the module "ot", so it was reported
    missing on 2026-08-20 IMMEDIATELY AFTER being installed successfully --
    idea 157 lost runs 193, 194 and 196 to it. Pillow/PIL,
    scikit-learn/sklearn and opencv-python/cv2 are the same shape, so the
    map was always going to be one package behind.

    Verified against the live A10G runtime when this shipped: POT, pot,
    scipy, networkx and scikit-learn all resolve through
    importlib.metadata, while a genuinely absent distribution still reports
    missing.
    """

    def _script(self):
        import inspect

        from meta_harness.backends import colab_cli

        return inspect.getsource(colab_cli)

    def test_the_hand_maintained_name_map_is_gone(self):
        self.assertNotIn("_DISTRIBUTION_MODULES", self._script())

    def test_presence_is_decided_by_distribution_metadata(self):
        script = self._script()
        self.assertIn("importlib.metadata.distribution(name)", script)

    def test_find_spec_survives_as_the_fallback(self):
        # a module present without distribution metadata must still count
        script = self._script()
        self.assertIn("importlib.util.find_spec", script)

    def test_the_refusal_itself_is_unchanged(self):
        # the remote must still never install at runtime
        self.assertIn("dependency_install_blocked", self._script())


class LaneMismatchIsNotAScienceFailureTests(unittest.TestCase):
    """A lane that lacks a declared dependency judges no hypothesis.

    Request 97 was refused on colab-pro-2 with exit 78 for a package that had
    been pre-provisioned on aws-g5-1, where requests 90-96 all succeeded
    (2026-08-20). The lanes are not interchangeable: the same experiment
    succeeds or fails on where it happens to land.

    Charging that to the science retry budget punishes a candidate for the
    scheduler's routing. It is the same evidential situation as a transport
    death -- no examples evaluated, so nothing learned -- and belongs in the
    same category, while still terminating through a bounded budget.
    """

    def test_the_observed_refusal_is_a_lane_mismatch(self):
        from meta_harness.failure_policy import is_lane_mismatch_failure

        for reason in (
            "experiment_exit_78",
            "dependency_install_blocked: reviewed runtime must be "
            "pre-provisioned; missing POT",
        ):
            self.assertTrue(is_lane_mismatch_failure(reason), reason)

    def test_a_real_code_failure_is_not_a_lane_mismatch(self):
        from meta_harness.failure_policy import is_lane_mismatch_failure

        for reason in ("experiment_exit_2", "required_artifacts_missing", "", None):
            self.assertFalse(is_lane_mismatch_failure(reason), reason)

    def test_measured_nothing_covers_both_causes(self):
        from meta_harness.failure_policy import measured_nothing

        self.assertTrue(measured_nothing("transport:ColabCLIError:colab provision failed"))
        self.assertTrue(measured_nothing("experiment_exit_78"))
        # but a run that actually evaluated examples and failed does count
        self.assertFalse(measured_nothing("experiment_exit_2"))

    def test_both_retry_budgets_use_the_shared_predicate(self):
        import inspect

        import scripts.auto_advance as aa
        from meta_harness import evidence_audit

        self.assertIn("measured_nothing", inspect.getsource(aa))
        self.assertIn("measured_nothing", inspect.getsource(evidence_audit))
