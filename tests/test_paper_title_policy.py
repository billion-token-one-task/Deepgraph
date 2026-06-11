from agents.paper_idea_agent import EXPERIMENT_DESIGN_SYSTEM
from agents.paper_title_policy import TITLE_NAMING_STANDARD_TEXT, normalize_paper_title


def test_normalizes_raw_fixed_point_claim_to_symbolic_title():
    title = normalize_paper_title(
        "Benchmark-conditioned consensus/refinement as a shared fixed-point operator on answer distributions",
        method_name="Benchmark-conditioned consensus/refinement as a shared fixed-point operator on answer distributions",
        context={"full_benchmark_completed": False},
    )

    assert title == "Attractor: Benchmark-Conditioned Refinement for Answer Distribution Consensus"
    assert "/" not in title
    assert "Fixed-Point" not in title


def test_preserves_acronym_title_spacing():
    title = normalize_paper_title(
        "Q-VAE:Q-Guided Value-Gradient Matching for Flow-Matching VLA Policies",
        context={"full_benchmark_completed": True},
    )

    assert title == "Q-VAE: Q-Guided Value-Gradient Matching for Flow-Matching VLA Policies"


def test_experiment_design_prompt_embeds_title_standard():
    assert TITLE_NAMING_STANDARD_TEXT in EXPERIMENT_DESIGN_SYSTEM
    assert "{TITLE_NAMING_STANDARD_TEXT}" not in EXPERIMENT_DESIGN_SYSTEM
