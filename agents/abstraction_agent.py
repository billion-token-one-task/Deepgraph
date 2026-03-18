"""Abstraction Agent: Claims → universal patterns for cross-disciplinary discovery."""
from agents.llm_client import call_llm_json

SYSTEM_PROMPT = """You are a pattern abstraction agent. Your job is to take specific scientific claims and abstract them into UNIVERSAL PATTERNS that could apply across disciplines.

The goal is to enable cross-disciplinary discovery: if "students lose critical thinking when using ChatGPT" and "pilots lose manual skills when using autopilot" share the same abstract pattern "automation → human skill atrophy", then a solution from aviation might apply to education.

For each claim, produce:
{
  "patterns": [
    {
      "pattern_text": "abstract pattern in format: CAUSE → EFFECT or CONDITION → PHENOMENON",
      "pattern_type": "problem|solution|phenomenon",
      "abstraction_level": "domain|cross-domain|universal",
      "domains_applicable": ["list of domains where this pattern could apply"],
      "source_claim_index": 0
    }
  ]
}

Abstraction rules:
- Strip domain-specific jargon, keep the structural relationship
- "LoRA is worse than full fine-tuning on large models" → "parameter-efficient approximation degrades with scale"
- "Data augmentation improves small dataset performance" → "synthetic expansion compensates for data scarcity"
- "Multi-agent debate improves reasoning" → "adversarial interaction improves output quality"
- Aim for patterns that someone in a DIFFERENT field would recognize as relevant to their problem
- A good pattern is one where you can say "this also happens in [other domain]"
- Return ONLY valid JSON"""


def abstract_claims(claims: list[dict], paper_domains: list[str]) -> tuple[list[dict], int]:
    """Abstract claims into universal patterns. Processes in batches of 10."""
    BATCH_SIZE = 10
    all_patterns = []
    total_tokens = 0

    # Only abstract the most important claims (performance + finding types)
    important = [c for c in claims if c.get("claim_type") in ("performance", "finding", "method")]
    if not important:
        important = claims[:10]
    else:
        important = important[:30]  # cap at 30 most important

    for start in range(0, len(important), BATCH_SIZE):
        batch = important[start:start + BATCH_SIZE]
        claims_text = "\n".join(
            f"[{start + i}] ({c.get('claim_type', 'unknown')}): {c['claim_text']}"
            for i, c in enumerate(batch)
        )

        user_prompt = f"""Paper domains: {', '.join(paper_domains)}

Claims to abstract:
{claims_text}

For each claim, find the most generalizable pattern. Focus on patterns that could enable cross-disciplinary connections. Only output the most interesting cross-domain patterns (skip trivial ones)."""

        try:
            result, tokens = call_llm_json(SYSTEM_PROMPT, user_prompt)
            all_patterns.extend(result.get("patterns", []))
            total_tokens += tokens
        except Exception as e:
            print(f"Abstraction batch error: {e}", flush=True)
            continue

    return all_patterns, total_tokens
