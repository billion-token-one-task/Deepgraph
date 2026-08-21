"""An unset field is not an alias that matches every dataset in existence.

`_known_protocol_for` matches by substring in both directions, and five
registered protocols carry an empty `hf_dataset`. `"" in anything` is True, so
each of those empty strings matched every name ever asked about, and the first
in registry order answered for all of them:

    BIG-Bench object_counting   -> CIFAR-10
    tasksource/bigbench         -> CIFAR-10
    my_totally_made_up_dataset  -> CIFAR-10
    xyzzy                       -> CIFAR-10
    openai/gsm8k                -> GSM8K          (registered, so correct)

This was not two object-recognition tasks being confused. **Every unregistered
dataset in the system resolved to CIFAR-10**, and inherited its protocol along
with the name: requires_harness (which blocked execution outright until
floors106), minimum_repeats 3, primary_metric accuracy, and cs.toronto.edu as
the official source.

The M3 session reported it as three separate defects -- the dataset resolving to
CIFAR-10, the requirements inflating, and the harness gate blocking. All three
were this one line.

What it did NOT corrupt is what ran. The execution contract comes from
execution_requirements, and run 240 really did run Qwen/Qwen2.5-1.5B on
tasksource/bigbench at revision 210c156. What was corrupted is the record --
the part a manuscript quotes.
"""

import unittest

from agents.benchmark_protocol import (
    KNOWN_BENCHMARK_PROTOCOLS,
    _known_protocol_for,
)


def _name_of(protocol):
    return (protocol or {}).get("canonical_name")


class EmptyAliasTests(unittest.TestCase):
    def test_an_unregistered_dataset_matches_nothing(self):
        for name in (
            "BIG-Bench object_counting",
            "tasksource/bigbench",
            "my_totally_made_up_dataset",
            "xyzzy",
        ):
            with self.subTest(name=name):
                self.assertIsNone(
                    _known_protocol_for(name),
                    f"{name!r} resolved to {_name_of(_known_protocol_for(name))}",
                )

    def test_nothing_resolves_to_cifar_by_accident(self):
        # The specific symptom, stated as its own assertion so a regression
        # names the thing that actually happened.
        self.assertNotEqual(_name_of(_known_protocol_for("xyzzy")), "CIFAR-10")

    def test_an_empty_name_matches_nothing(self):
        for name in ("", "   ", None):
            with self.subTest(name=name):
                self.assertIsNone(_known_protocol_for(name or ""))

    def test_the_registry_still_contains_empty_hf_dataset_entries(self):
        # The fix guards against these; if they were simply filled in, this
        # test's premise would be gone and the guard would rot untested.
        blank = [
            key
            for key, value in KNOWN_BENCHMARK_PROTOCOLS.items()
            if not str(value.get("hf_dataset") or "").strip()
        ]
        self.assertTrue(blank, "no protocol has a blank hf_dataset any more")


class RegisteredProtocolsStillResolveTests(unittest.TestCase):
    """The fix must not have narrowed matching into uselessness."""

    def test_matching_by_hf_dataset(self):
        self.assertEqual(_name_of(_known_protocol_for("openai/gsm8k")), "GSM8K")

    def test_matching_by_short_name(self):
        self.assertEqual(_name_of(_known_protocol_for("gsm8k")), "GSM8K")

    def test_matching_by_canonical_name_when_hf_dataset_is_blank(self):
        # HarmBench and LongBench have no hf_dataset; they must still match on
        # their canonical name, which is the case the guard could have broken.
        for name in ("HarmBench", "LongBench"):
            with self.subTest(name=name):
                self.assertEqual(_name_of(_known_protocol_for(name)), name)

    def test_every_registered_protocol_resolves_to_itself(self):
        for key, value in KNOWN_BENCHMARK_PROTOCOLS.items():
            with self.subTest(protocol=key):
                self.assertEqual(
                    _name_of(_known_protocol_for(value["canonical_name"])),
                    value["canonical_name"],
                )


if __name__ == "__main__":
    unittest.main()
