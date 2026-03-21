"""Extraction Agent: scientific paper -> taxonomy classification, evidence rows, and graph evidence."""
from agents.llm_client import call_llm_json
from db.taxonomy import get_all_leaf_ids

SYSTEM_PROMPT = """You are a scientific paper analysis agent. Extract structured, concrete information from research papers.

You will be given a paper and a taxonomy of scientific research areas (leaf node IDs). Your job:

1. CLASSIFY the paper into 1-3 leaf taxonomy nodes (pick the most specific match)
2. EXTRACT every (method, dataset, metric, value) result tuple from the paper
3. EXTRACT methods proposed or used
4. NOTE key findings and limitations

Return a JSON object:
{
  "taxonomy_nodes": [
    {"node_id": "ml.dl.cv.detection", "confidence": 0.95}
  ],
  "paper_overview": {
    "plain_summary": "In simple language, what this paper is trying to do and why",
    "problem_statement": "What concrete problem or bottleneck the paper addresses",
    "approach_summary": "How the paper approaches the problem in plain language",
    "work_type": "model|training_method|benchmark|dataset|application|system|analysis|theory",
    "key_findings": ["Finding 1", "Finding 2"],
    "limitations": ["Limitation 1", "Limitation 2"],
    "open_questions": ["Open question 1"]
  },
  "knowledge_graph": {
    "entities": [
      {
        "name": "Transformer",
        "entity_type": "method",
        "description": "Sequence model based on attention",
        "aliases": ["attention-only model"],
        "mention_role": "used",
        "confidence": 0.95,
        "evidence_location": "Section 2",
        "source_text": "We use a Transformer backbone..."
      }
    ],
    "relations": [
      {
        "subject": "Transformer",
        "subject_type": "method",
        "predicate": "evaluated_on",
        "object": "WMT14 En-De",
        "object_type": "dataset",
        "confidence": 0.94,
        "evidence_location": "Table 1",
        "source_text": "Transformer is evaluated on WMT14 En-De."
      }
    ]
  },
  "results": [
    {
      "method_name": "YOLOv9",
      "dataset_name": "COCO val2017",
      "metric_name": "mAP@0.5:0.95",
      "metric_value": 55.6,
      "metric_unit": "%",
      "is_sota": 1,
      "evidence_location": "Table 2"
    }
  ],
  "methods": [
    {
      "name": "YOLOv9",
      "category": "object detection",
      "description": "Real-time object detector with PGI and GELAN",
      "key_innovation": "Programmable Gradient Information for better training",
      "builds_on": ["YOLOv7", "ELAN"]
    }
  ],
  "claims": [
    {
      "claim_text": "YOLOv9 achieves 55.6% mAP on COCO with 50% fewer parameters than YOLOv8",
      "claim_type": "performance",
      "method_name": "YOLOv9",
      "dataset_name": "COCO val2017",
      "metric_name": "mAP@0.5:0.95",
      "metric_value": 55.6,
      "evidence_location": "Table 2",
      "conditions": {"model_size": "large", "input_resolution": "640x640"}
    }
  ],
  "key_findings": ["Finding 1", "Finding 2"],
  "limitations_stated": ["Limitation 1"]
}

Rules:
- For taxonomy_nodes: use ONLY the provided leaf node IDs. Pick 1-3 most relevant.
- For paper_overview: write for a technically curious non-expert. Avoid unexplained jargon.
- work_type must be one of: model, training_method, benchmark, dataset, application, system, analysis, theory
- For knowledge_graph entities, use one of:
  concept, method, task, dataset, metric, artifact, material, gene, protein, disease, organism, theory
- For knowledge_graph predicates, prefer:
  uses, builds_on, evaluated_on, measured_by, compares_with, applied_to,
  improves_over, part_of, studies, predicts, treats, interacts_with, derived_from, related_to
- Keep entities canonical and reusable across papers. Do not create separate entities for trivial ablations unless the paper treats them as named systems.
- Add source_text only as a short grounding snippet, not a long quote.
- For results: extract EVERY quantitative result from tables and text. Each row in a results table = one result entry.
- method_name should be the exact name used in the paper (e.g. "GPT-4o", "LLaMA-3-70B", "ResNet-50")
- dataset_name should be the standard benchmark name (e.g. "ImageNet-1K", "COCO val2017", "MMLU")
- metric_name should be standard (e.g. "accuracy", "mAP@0.5", "BLEU", "F1", "perplexity")
- metric_value should be numeric. If a percentage, store the number (95.2 not 0.952)
- is_sota=1 only if the paper explicitly claims state-of-the-art
- Extract ALL results, not just the best ones
- Limitations and open questions should be grounded in the paper, not invented.
- Return ONLY valid JSON, no markdown formatting"""


def extract_paper(paper_id: str, title: str, text: str) -> tuple[dict, int]:
    """Extract structured info from a paper. Returns (extraction_dict, tokens_used)."""
    leaf_ids = get_all_leaf_ids()
    taxonomy_hint = "Available taxonomy leaf nodes:\n" + "\n".join(
        f"  {nid}" for nid in leaf_ids
    )

    user_prompt = f"""Paper ID: {paper_id}
Title: {title}

{taxonomy_hint}

Full text:
{text}"""

    result, tokens = call_llm_json(SYSTEM_PROMPT, user_prompt)
    return result, tokens
