"""Presentation-only aliases for the disabled CGGR/CRPP example.

This module has no default registration and must never fill missing values or
rewrite scientific caveats.
"""

METHOD_ALIASES = {
    "Vanilla Direct Answering": "Direct",
    "Certified Residual Policy Packets": "CRPP",
    "Confidence Gate": "Conf. Gate",
    "Confidence Routing": "Conf. Gate",
    "Disagreement Routing": "Disagree",
    "Random Budget-Matched Routing": "Rand. Budget",
    "CAR-Style Certainty Adaptive Routing": "CAR",
    "Self-Route-Style Mode Routing": "Self-Route",
    "Rational-Metareasoning VOC Routing": "VOC",
    "Always-Reason Chain-of-Thought": "Always-CoT",
    "Self-Consistency Reasoning": "Self-Cons.",
    "Least-to-Most Prompting": "LtM",
}

ROW_GROUPS = (
    ("Direct and packet-based methods", ("Direct", "CRPP")),
    ("Adaptive routing baselines", ("Conf. Gate", "Disagree", "Rand. Budget", "CAR", "Self-Route", "VOC")),
    ("High-compute reasoning baselines", ("Always-CoT", "Self-Cons.", "LtM")),
)
