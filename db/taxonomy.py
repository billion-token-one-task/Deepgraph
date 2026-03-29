"""Predefined ML taxonomy tree and CRUD operations."""
import json
from collections import Counter, defaultdict
from config import PAPER_CLUSTER_MIN_PAPERS, ROOT_NODE_ID
from db import database as db
from db import evidence_graph as graph

# ── Predefined taxonomy ────────────────────────────────────────────
# Format: (id, name, parent_id, depth, description, sort_order)
TAXONOMY = [
    # Root
    ("ml", "Machine Learning", None, 0, "All machine learning research", 0),

    # ── Level 1 ────────────────────────────────────────────────────
    ("ml.classical", "Classical ML", "ml", 1,
     "Traditional machine learning: SVM, trees, clustering, etc.", 0),
    ("ml.dl", "Deep Learning", "ml", 1,
     "Neural network based approaches", 1),
    ("ml.rl", "Reinforcement Learning", "ml", 1,
     "Learning from environment interaction and rewards", 2),
    ("ml.agents", "AI Agents", "ml", 1,
     "Autonomous agents: tool use, multi-agent, code generation", 3),
    ("ml.ai4sci", "AI for Science", "ml", 1,
     "ML applied to scientific discovery", 4),
    ("ml.theory", "ML Theory", "ml", 1,
     "Theoretical foundations: optimization, generalization, learning theory", 5),

    # ── Classical ML (Level 2) ─────────────────────────────────────
    ("ml.classical.svm", "Support Vector Machines", "ml.classical", 2,
     "Kernel methods and SVMs", 0),
    ("ml.classical.trees", "Decision Trees & Ensembles", "ml.classical", 2,
     "Random forests, gradient boosting, XGBoost", 1),
    ("ml.classical.clustering", "Clustering", "ml.classical", 2,
     "K-means, DBSCAN, hierarchical clustering", 2),
    ("ml.classical.dim_reduction", "Dimensionality Reduction", "ml.classical", 2,
     "PCA, t-SNE, UMAP", 3),
    ("ml.classical.bayes", "Bayesian Methods", "ml.classical", 2,
     "Bayesian inference, Gaussian processes", 4),

    # ── Deep Learning (Level 2) ────────────────────────────────────
    ("ml.dl.cv", "Computer Vision", "ml.dl", 2,
     "Visual understanding with deep learning", 0),
    ("ml.dl.nlp", "Natural Language Processing", "ml.dl", 2,
     "Language understanding and generation", 1),
    ("ml.dl.embodied", "Embodied Intelligence", "ml.dl", 2,
     "Manipulation, navigation, sim-to-real", 2),
    ("ml.dl.multimodal", "Multimodal Learning", "ml.dl", 2,
     "Vision-language, cross-modal learning", 3),
    ("ml.dl.audio", "Audio & Speech", "ml.dl", 2,
     "Speech recognition, synthesis, audio understanding", 4),
    ("ml.dl.graph", "Graph Neural Networks", "ml.dl", 2,
     "Learning on graph-structured data", 5),
    ("ml.dl.generative", "Generative Models", "ml.dl", 2,
     "VAEs, GANs, diffusion models, flow matching", 6),
    ("ml.dl.foundation", "Foundation Models", "ml.dl", 2,
     "Large-scale pretrained models, scaling laws", 7),
    ("ml.dl.efficiency", "Efficient Deep Learning", "ml.dl", 2,
     "Pruning, quantization, distillation, NAS", 8),

    # ── Computer Vision (Level 3) ──────────────────────────────────
    ("ml.dl.cv.classification", "Image Classification", "ml.dl.cv", 3,
     "Image-level recognition and categorization", 0),
    ("ml.dl.cv.detection", "Object Detection", "ml.dl.cv", 3,
     "Localizing and classifying objects in images", 1),
    ("ml.dl.cv.segmentation", "Segmentation", "ml.dl.cv", 3,
     "Semantic, instance, and panoptic segmentation", 2),
    ("ml.dl.cv.generation", "Image Generation", "ml.dl.cv", 3,
     "Image synthesis, editing, inpainting", 3),
    ("ml.dl.cv.video", "Video Understanding", "ml.dl.cv", 3,
     "Action recognition, video generation, tracking", 4),
    ("ml.dl.cv.3d", "3D Vision", "ml.dl.cv", 3,
     "NeRF, 3D reconstruction, point clouds, Gaussian splatting", 5),
    ("ml.dl.cv.reid", "Re-Identification", "ml.dl.cv", 3,
     "Matching identities across cameras, domains, or species", 6),
    ("ml.dl.cv.medical", "Medical Imaging", "ml.dl.cv", 3,
     "Radiology, pathology, medical image analysis", 7),
    ("ml.dl.cv.face", "Face Analysis", "ml.dl.cv", 3,
     "Face recognition, detection, generation", 8),
    ("ml.dl.cv.document", "Document Understanding", "ml.dl.cv", 3,
     "OCR, layout analysis, document parsing", 9),

    # ── Re-Identification (Level 4) ──────────────────────────────
    ("ml.dl.cv.reid.person", "Person Re-Identification", "ml.dl.cv.reid", 4,
     "Matching the same person across cameras or times", 0),
    ("ml.dl.cv.reid.cross_domain", "Cross-Domain Re-Identification", "ml.dl.cv.reid", 4,
     "Re-identification under domain shift across environments or datasets", 1),
    ("ml.dl.cv.reid.animal", "Animal Re-Identification", "ml.dl.cv.reid", 4,
     "Matching individual animals across images, videos, or sensors", 2),

    # ── NLP (Level 3) ─────────────────────────────────────────────
    ("ml.dl.nlp.lm", "Language Modeling", "ml.dl.nlp", 3,
     "Autoregressive and masked language models, perplexity", 0),
    ("ml.dl.nlp.translation", "Machine Translation", "ml.dl.nlp", 3,
     "Neural machine translation, multilingual models", 1),
    ("ml.dl.nlp.ie", "Information Extraction", "ml.dl.nlp", 3,
     "NER, relation extraction, event extraction", 2),
    ("ml.dl.nlp.qa", "Question Answering", "ml.dl.nlp", 3,
     "Reading comprehension, open-domain QA", 3),
    ("ml.dl.nlp.summarization", "Summarization", "ml.dl.nlp", 3,
     "Abstractive and extractive summarization", 4),
    ("ml.dl.nlp.reasoning", "Reasoning & Math", "ml.dl.nlp", 3,
     "Chain-of-thought, mathematical reasoning, logic", 5),
    ("ml.dl.nlp.dialogue", "Dialogue Systems", "ml.dl.nlp", 3,
     "Conversational AI, chatbots", 6),
    ("ml.dl.nlp.sentiment", "Sentiment Analysis", "ml.dl.nlp", 3,
     "Opinion mining, emotion detection", 7),
    ("ml.dl.nlp.alignment", "RLHF & Alignment", "ml.dl.nlp", 3,
     "Reinforcement learning from human feedback, safety", 8),

    # ── Embodied Intelligence (Level 3) ────────────────────────────
    ("ml.dl.embodied.manipulation", "Manipulation", "ml.dl.embodied", 3,
     "Robotic grasping, dexterous manipulation", 0),
    ("ml.dl.embodied.navigation", "Navigation", "ml.dl.embodied", 3,
     "Visual navigation, path planning", 1),
    ("ml.dl.embodied.sim2real", "Sim-to-Real Transfer", "ml.dl.embodied", 3,
     "Transferring from simulation to real world", 2),
    ("ml.dl.embodied.autonomous", "Autonomous Driving", "ml.dl.embodied", 3,
     "Self-driving, end-to-end driving", 3),

    # ── Multimodal (Level 3) ───────────────────────────────────────
    ("ml.dl.multimodal.vl", "Vision-Language", "ml.dl.multimodal", 3,
     "CLIP, image captioning, visual QA", 0),
    ("ml.dl.multimodal.t2i", "Text-to-Image", "ml.dl.multimodal", 3,
     "Stable Diffusion, DALL-E, text-guided generation", 1),
    ("ml.dl.multimodal.t2v", "Text-to-Video", "ml.dl.multimodal", 3,
     "Video generation from text", 2),
    ("ml.dl.multimodal.embodied", "Multimodal Embodied", "ml.dl.multimodal", 3,
     "Vision-language-action models", 3),

    # ── Audio (Level 3) ────────────────────────────────────────────
    ("ml.dl.audio.asr", "Speech Recognition", "ml.dl.audio", 3,
     "Automatic speech recognition", 0),
    ("ml.dl.audio.tts", "Text-to-Speech", "ml.dl.audio", 3,
     "Speech synthesis, voice cloning", 1),
    ("ml.dl.audio.music", "Music Generation", "ml.dl.audio", 3,
     "Music composition, audio generation", 2),

    # ── Reinforcement Learning (Level 2) ───────────────────────────
    ("ml.rl.model_free", "Model-Free RL", "ml.rl", 2,
     "Policy gradient, Q-learning, actor-critic", 0),
    ("ml.rl.model_based", "Model-Based RL", "ml.rl", 2,
     "World models, planning with learned dynamics", 1),
    ("ml.rl.offline", "Offline RL", "ml.rl", 2,
     "Learning from fixed datasets without interaction", 2),
    ("ml.rl.marl", "Multi-Agent RL", "ml.rl", 2,
     "Cooperative and competitive multi-agent learning", 3),
    ("ml.rl.safe", "Safe RL", "ml.rl", 2,
     "Constrained optimization, risk-sensitive RL", 4),

    # ── AI Agents (Level 2) ────────────────────────────────────────
    ("ml.agents.tool_use", "Tool Use", "ml.agents", 2,
     "LLM tool calling, function calling, API agents", 0),
    ("ml.agents.multi_agent", "Multi-Agent Systems", "ml.agents", 2,
     "Agent collaboration, debate, orchestration", 1),
    ("ml.agents.code_gen", "Code Generation", "ml.agents", 2,
     "Code synthesis, program repair, code agents", 2),
    ("ml.agents.web", "Web Agents", "ml.agents", 2,
     "Browser automation, web navigation", 3),
    ("ml.agents.planning", "Planning & Reasoning", "ml.agents", 2,
     "Task decomposition, plan generation, search", 4),

    # ── AI for Science (Level 2) ───────────────────────────────────
    ("ml.ai4sci.drug", "Drug Discovery", "ml.ai4sci", 2,
     "Molecular generation, protein-ligand binding", 0),
    ("ml.ai4sci.protein", "Protein Science", "ml.ai4sci", 2,
     "Protein folding, design, function prediction", 1),
    ("ml.ai4sci.materials", "Materials Science", "ml.ai4sci", 2,
     "Materials discovery, property prediction", 2),
    ("ml.ai4sci.climate", "Climate & Weather", "ml.ai4sci", 2,
     "Weather forecasting, climate modeling", 3),
    ("ml.ai4sci.physics", "Physics Simulation", "ml.ai4sci", 2,
     "Neural PDE solvers, physics-informed NN", 4),
    ("ml.ai4sci.bio", "Biology & Genomics", "ml.ai4sci", 2,
     "Genomics, single-cell, biological sequence modeling", 5),

    # ── ML Theory (Level 2) ────────────────────────────────────────
    ("ml.theory.optimization", "Optimization", "ml.theory", 2,
     "SGD variants, Adam, learning rate schedules", 0),
    ("ml.theory.generalization", "Generalization Theory", "ml.theory", 2,
     "PAC learning, VC theory, generalization bounds", 1),
    ("ml.theory.robustness", "Robustness & Adversarial", "ml.theory", 2,
     "Adversarial examples, certified robustness", 2),
    ("ml.theory.fairness", "Fairness & Bias", "ml.theory", 2,
     "Algorithmic fairness, debiasing", 3),
    ("ml.theory.interpretability", "Interpretability", "ml.theory", 2,
     "Explainable AI, mechanistic interpretability", 4),
]

SCIENCE_TAXONOMY = [
    ("science", "Science", None, 0, "Scientific research across computational, physical, life, and engineering domains", 0),
    ("science.math", "Mathematics & Statistics", "science", 1,
     "Optimization, probability, statistics, and applied mathematics", 0),
    ("science.physics", "Physics", "science", 1,
     "Theoretical, computational, and experimental physics", 1),
    ("science.chemistry", "Chemistry & Materials", "science", 1,
     "Chemistry, materials discovery, and molecular systems", 2),
    ("science.life", "Life Sciences", "science", 1,
     "Biology, genomics, neuroscience, and ecosystems", 3),
    ("science.medicine", "Medicine & Health", "science", 1,
     "Diagnostics, therapies, medical imaging, and public health", 4),
    ("science.earth", "Earth & Climate", "science", 1,
     "Climate, geoscience, weather, remote sensing, and oceans", 5),
    ("science.engineering", "Engineering", "science", 1,
     "Control, robotics, signal processing, and energy systems", 6),
    ("science.computing", "Computing & AI", "science", 1,
     "Machine learning, AI systems, and computational science", 7),

    ("science.math.optimization", "Optimization", "science.math", 2,
     "Optimization methods, numerical solvers, and control-oriented math", 0),
    ("science.math.statistics", "Statistics & Probability", "science.math", 2,
     "Statistical inference, uncertainty, and probabilistic modeling", 1),
    ("science.math.applied", "Applied Mathematics", "science.math", 2,
     "Mathematical modeling, PDEs, inverse problems, and numerical analysis", 2),

    ("science.physics.comp", "Computational Physics", "science.physics", 2,
     "Simulation, inverse modeling, and scientific machine learning for physics", 0),
    ("science.physics.bio", "Biophysics", "science.physics", 2,
     "Physics-informed studies of living systems", 1),
    ("science.physics.matter", "Condensed Matter & Materials Physics", "science.physics", 2,
     "Structure, properties, and behavior of materials", 2),
    ("science.physics.astro", "Astrophysics & Space", "science.physics", 2,
     "Astronomy, cosmology, and space science", 3),

    ("science.chemistry.molecular", "Molecular & Computational Chemistry", "science.chemistry", 2,
     "Reaction modeling, molecular simulation, and chemical design", 0),
    ("science.chemistry.materials", "Materials Discovery", "science.chemistry", 2,
     "Property prediction, synthesis planning, and materials screening", 1),
    ("science.chemistry.drug", "Drug & Molecule Design", "science.chemistry", 2,
     "Small molecules, binding, and therapeutic candidate discovery", 2),

    ("science.life.genomics", "Genomics & Single-Cell Biology", "science.life", 2,
     "Sequencing, regulatory biology, and omics analysis", 0),
    ("science.life.protein", "Proteins & Structural Biology", "science.life", 2,
     "Protein design, folding, and biological structure prediction", 1),
    ("science.life.neuro", "Neuroscience", "science.life", 2,
     "Neural data analysis, cognition, and brain-inspired models", 2),
    ("science.life.ecology", "Ecology & Evolution", "science.life", 2,
     "Population, ecosystem, and biodiversity research", 3),

    ("science.medicine.imaging", "Medical Imaging", "science.medicine", 2,
     "Radiology, pathology, and imaging-based diagnosis", 0),
    ("science.medicine.clinical", "Clinical AI", "science.medicine", 2,
     "Prediction, triage, records, and decision support in healthcare", 1),
    ("science.medicine.public_health", "Public Health", "science.medicine", 2,
     "Population health, surveillance, and epidemiological modeling", 2),

    ("science.earth.climate", "Climate & Weather", "science.earth", 2,
     "Forecasting, climate risk, and Earth-system modeling", 0),
    ("science.earth.geo", "Geoscience", "science.earth", 2,
     "Geophysics, subsurface modeling, and Earth observation", 1),
    ("science.earth.remote", "Remote Sensing", "science.earth", 2,
     "Satellite, aerial, and sensor-based observation of Earth", 2),
    ("science.earth.ocean", "Ocean & Hydrology", "science.earth", 2,
     "Ocean processes, water systems, and environmental sensing", 3),

    ("science.engineering.robotics", "Robotics & Embodied Systems", "science.engineering", 2,
     "Manipulation, navigation, autonomy, and embodied control", 0),
    ("science.engineering.control", "Control Systems", "science.engineering", 2,
     "Dynamics, planning, and system optimization", 1),
    ("science.engineering.signal", "Signal Processing", "science.engineering", 2,
     "Audio, imaging, communications, and time-series systems", 2),
    ("science.engineering.energy", "Energy Systems", "science.engineering", 2,
     "Power systems, batteries, grids, and sustainable engineering", 3),

    ("science.computing.ml", "Machine Learning", "science.computing", 2,
     "Learning systems, benchmarks, and model development", 0),
    ("science.computing.ai_agents", "AI Agents", "science.computing", 2,
     "Tool-using systems, orchestration, and autonomous workflows", 1),
    ("science.computing.cv", "Computer Vision", "science.computing", 2,
     "Image, video, and visual understanding", 2),
    ("science.computing.nlp", "Language & Reasoning", "science.computing", 2,
     "Language modeling, retrieval, reasoning, and dialogue", 3),
    ("science.computing.scientific_ml", "Scientific Machine Learning", "science.computing", 2,
     "Physics-informed and simulation-aware learning systems", 4),
    ("science.computing.cv.reid", "Re-Identification", "science.computing.cv", 3,
     "Matching the same identity across views, domains, or species", 0),
    ("science.computing.cv.reid.cross_domain", "Cross-Domain Re-Identification", "science.computing.cv.reid", 4,
     "Identity matching across different domains or environments", 0),
    ("science.computing.cv.reid.animal", "Animal Re-Identification", "science.computing.cv.reid", 4,
     "Identity matching for individual animals", 1),
]

ALL_TAXONOMY = TAXONOMY + SCIENCE_TAXONOMY

LOWER_IS_BETTER_KEYWORDS = (
    "error",
    "loss",
    "wer",
    "cer",
    "edit distance",
    "perplexity",
    "fid",
    "gfid",
    "rfid",
    "mae",
    "mse",
    "rmse",
    "nll",
    "latency",
    "runtime",
    "time",
    "memory",
    "flops",
)


def _loads_list(value: str | None) -> list:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _count_subtree_papers(node_id: str) -> int:
    row = db.fetchone(
        """SELECT COUNT(DISTINCT pt.paper_id) as c
           FROM paper_taxonomy pt
           JOIN taxonomy_nodes t ON pt.node_id = t.id
           WHERE t.id = ? OR t.id LIKE ? || '.%'""",
        (node_id, node_id),
    )
    return row["c"] if row else 0


def _count_subtree_results(node_id: str) -> int:
    row = db.fetchone(
        """SELECT COUNT(DISTINCT r.id) as c
           FROM results r
           JOIN result_taxonomy rt ON rt.result_id = r.id
           JOIN taxonomy_nodes t ON rt.node_id = t.id
           WHERE t.id = ? OR t.id LIKE ? || '.%'""",
        (node_id, node_id),
    )
    return row["c"] if row else 0


def _metric_prefers_lower(metric_name: str | None) -> bool:
    metric = (metric_name or "").strip().lower()
    return any(keyword in metric for keyword in LOWER_IS_BETTER_KEYWORDS)


def _is_better_result(candidate: dict, existing: dict | None) -> bool:
    if existing is None:
        return True

    candidate_value = candidate.get("metric_value")
    existing_value = existing.get("value")
    if candidate_value is None:
        return existing_value is None and candidate.get("paper_id", "") > existing.get("paper_id", "")
    if existing_value is None:
        return True

    if _metric_prefers_lower(candidate.get("metric_name")):
        if candidate_value != existing_value:
            return candidate_value < existing_value
    else:
        if candidate_value != existing_value:
            return candidate_value > existing_value

    return candidate.get("paper_id", "") > existing.get("paper_id", "")


def seed_taxonomy():
    """Insert the predefined taxonomy tree into the database.
    Safe to call multiple times (uses INSERT OR IGNORE).
    """
    for node_id, name, parent_id, depth, description, sort_order in ALL_TAXONOMY:
        db.execute(
            """INSERT OR IGNORE INTO taxonomy_nodes
               (id, name, parent_id, depth, description, sort_order)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (node_id, name, parent_id, depth, description, sort_order),
        )
    db.commit()


def get_all_leaf_ids() -> list[str]:
    """Return IDs of all leaf taxonomy nodes (no children)."""
    rows = db.fetchall(
        """SELECT t.id FROM taxonomy_nodes t
           LEFT JOIN taxonomy_nodes c ON c.parent_id = t.id
           WHERE c.id IS NULL
             AND (t.id = ? OR t.id LIKE ? || '.%')
           ORDER BY t.id"""
        ,
        (ROOT_NODE_ID, ROOT_NODE_ID),
    )
    return [r["id"] for r in rows]


def get_taxonomy_flat() -> list[dict]:
    """Return the full taxonomy as a flat list."""
    return db.fetchall(
        """SELECT * FROM taxonomy_nodes
           WHERE id = ? OR id LIKE ? || '.%'
           ORDER BY depth, sort_order, id""",
        (ROOT_NODE_ID, ROOT_NODE_ID),
    )


def get_node(node_id: str) -> dict | None:
    """Get a single taxonomy node."""
    return db.fetchone("SELECT * FROM taxonomy_nodes WHERE id=?", (node_id,))


def get_children(node_id: str) -> list[dict]:
    """Get direct children of a node, with aggregated counts."""
    children = db.fetchall(
        """SELECT t.*,
              (SELECT COUNT(DISTINCT pt.paper_id)
               FROM paper_taxonomy pt
               JOIN taxonomy_nodes sub ON pt.node_id = sub.id
               WHERE sub.id = t.id
                  OR sub.id LIKE t.id || '.%') AS paper_count,
              (SELECT COUNT(DISTINCT r.method_name)
               FROM results r
               JOIN result_taxonomy rt ON rt.result_id = r.id
               JOIN taxonomy_nodes sub ON rt.node_id = sub.id
               WHERE sub.id = t.id
                  OR sub.id LIKE t.id || '.%') AS method_count,
              (SELECT COUNT(*)
               FROM matrix_gaps mg
               JOIN taxonomy_nodes sub ON mg.node_id = sub.id
               WHERE sub.id = t.id
                  OR sub.id LIKE t.id || '.%') AS gap_count
           FROM taxonomy_nodes t
           WHERE t.parent_id=?
           ORDER BY t.sort_order""",
        (node_id,),
    )
    for child in children:
        summary = db.get_node_summary(child["id"])
        if summary:
            child["gap_count"] = max(child.get("gap_count", 0), len(summary.get("current_gaps", [])))
    return children


def get_root_children() -> list[dict]:
    """Get the root node's children (top-level categories)."""
    return get_children(ROOT_NODE_ID)


def get_breadcrumb(node_id: str) -> list[dict]:
    """Return the path from root to node_id as a list of {id, name}."""
    crumbs = []
    current = node_id
    while current:
        node = db.fetchone(
            "SELECT id, name, parent_id FROM taxonomy_nodes WHERE id=?",
            (current,),
        )
        if not node:
            break
        crumbs.insert(0, {"id": node["id"], "name": node["name"]})
        current = node["parent_id"]
    return crumbs


def get_ancestor_ids(node_id: str) -> list[str]:
    """Return ancestor IDs including the node itself and root."""
    ancestors = []
    current = node_id
    while current:
        ancestors.append(current)
        row = db.fetchone("SELECT parent_id FROM taxonomy_nodes WHERE id=?", (current,))
        current = row["parent_id"] if row else None
    return ancestors


def get_node_papers(node_id: str, limit: int = 50) -> list[dict]:
    """Get papers classified under a node (including descendant nodes)."""
    rows = db.fetchall(
        """SELECT p.id, p.title, p.authors, p.published_date, p.status,
                  MAX(pt.confidence) as confidence,
                  pi.plain_summary, pi.problem_statement, pi.approach_summary,
                  pi.work_type, pi.key_findings, pi.limitations, pi.open_questions
           FROM papers p
           JOIN paper_taxonomy pt ON p.id = pt.paper_id
           JOIN taxonomy_nodes t ON pt.node_id = t.id
           LEFT JOIN paper_insights pi ON p.id = pi.paper_id
           WHERE t.id = ? OR t.id LIKE ? || '.%'
           GROUP BY p.id
           ORDER BY p.published_date DESC
           LIMIT ?""",
        (node_id, node_id, limit),
    )
    for row in rows:
        row["key_findings"] = _loads_list(row.get("key_findings"))
        row["limitations"] = _loads_list(row.get("limitations"))
        row["open_questions"] = _loads_list(row.get("open_questions"))
    return rows


def get_node_paper_clusters(node_id: str, min_papers_to_cluster: int = PAPER_CLUSTER_MIN_PAPERS) -> list[dict]:
    """Cluster papers inside a node using shared entity signals."""
    papers = get_node_papers(node_id, limit=200)
    if len(papers) < min_papers_to_cluster:
        return []

    paper_ids = [paper["id"] for paper in papers]
    placeholders = ",".join("?" for _ in paper_ids)
    mention_rows = db.fetchall(
        f"""SELECT pem.paper_id, er.canonical_entity_id, ge.canonical_name
            FROM paper_entity_mentions pem
            JOIN entity_resolutions er ON er.entity_id = pem.entity_id
            JOIN graph_entities ge ON ge.id = er.canonical_entity_id
            WHERE pem.paper_id IN ({placeholders})
              AND ge.entity_type != 'metric'""",
        tuple(paper_ids),
    )

    paper_entities: dict[str, set[str]] = defaultdict(set)
    entity_names: dict[str, dict[str, str]] = defaultdict(dict)
    for row in mention_rows:
        paper_entities[row["paper_id"]].add(row["canonical_entity_id"])
        entity_names[row["paper_id"]][row["canonical_entity_id"]] = row["canonical_name"]

    work_types = {paper["id"]: paper.get("work_type", "") or "" for paper in papers}
    return cluster_papers_from_signals(
        papers=papers,
        paper_entities=paper_entities,
        work_types=work_types,
        entity_names=entity_names,
        min_papers_to_cluster=min_papers_to_cluster,
    )


def _intersection_strength(paper_overlap: int, entity_overlap: int) -> float:
    """Combine paper overlap and shared entities into one heatmap strength."""
    return round(paper_overlap * 2.0 + entity_overlap * 0.35, 2)


def cluster_papers_from_signals(
    papers: list[dict],
    paper_entities: dict[str, set[str]],
    work_types: dict[str, str],
    entity_names: dict[str, dict[str, str]],
    min_shared_entities: int = 2,
    min_papers_to_cluster: int = PAPER_CLUSTER_MIN_PAPERS,
) -> list[dict]:
    """Cluster papers by shared entities and work type signals."""
    if len(papers) < min_papers_to_cluster:
        return []

    paper_ids = [paper["id"] for paper in papers]
    adjacency: dict[str, set[str]] = defaultdict(set)
    for i, left_id in enumerate(paper_ids):
        left_entities = paper_entities.get(left_id, set())
        for right_id in paper_ids[i + 1:]:
            right_entities = paper_entities.get(right_id, set())
            shared = left_entities & right_entities
            same_work_type = bool(work_types.get(left_id) and work_types.get(left_id) == work_types.get(right_id))
            if len(shared) >= min_shared_entities or (same_work_type and len(shared) >= 1):
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)

    visited = set()
    components = []
    for paper_id in paper_ids:
        if paper_id in visited:
            continue
        stack = [paper_id]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(adjacency.get(current, set()) - visited))
        if len(component) >= 2:
            components.append(sorted(component))

    paper_map = {paper["id"]: paper for paper in papers}
    clusters = []
    for component in sorted(components, key=lambda ids: (-len(ids), ids[0])):
        entity_counter = Counter()
        work_type_counter = Counter()
        for paper_id in component:
            work_type = work_types.get(paper_id)
            if work_type:
                work_type_counter[work_type] += 1
            for entity_id in paper_entities.get(paper_id, set()):
                entity_counter[entity_id] += 1

        shared_entities = [
            entity_names.get(component[0], {}).get(entity_id) or entity_id
            for entity_id, count in entity_counter.most_common(5)
            if count >= 2
        ]
        dominant_work_type = work_type_counter.most_common(1)[0][0].replace("_", " ").title() if work_type_counter else ""
        if shared_entities:
            label = " / ".join(shared_entities[:2])
        elif dominant_work_type:
            label = dominant_work_type
        else:
            label = f"Cluster of {len(component)} related papers"

        clusters.append({
            "label": label,
            "paper_count": len(component),
            "paper_ids": component,
            "shared_entities": shared_entities[:5],
            "work_types": [name for name, _count in work_type_counter.most_common(3)],
            "sample_papers": [
                {"id": paper_id, "title": paper_map[paper_id]["title"]}
                for paper_id in component[:4]
            ],
        })

    return clusters[:6]


def get_leaf_descendants(node_id: str, limit: int = 12, min_papers: int = 1) -> list[dict]:
    """Get leaf descendants with paper counts for a node."""
    rows = db.fetchall(
        """SELECT t.id, t.name, t.parent_id, t.depth, t.description, t.sort_order,
                  COUNT(DISTINCT pt.paper_id) AS paper_count
           FROM taxonomy_nodes t
           LEFT JOIN taxonomy_nodes c ON c.parent_id = t.id
           LEFT JOIN paper_taxonomy pt ON pt.node_id = t.id
           WHERE c.id IS NULL
             AND (t.id = ? OR t.id LIKE ? || '.%')
           GROUP BY t.id, t.name, t.parent_id, t.depth, t.description, t.sort_order
           HAVING paper_count >= ?
           ORDER BY paper_count DESC, t.depth DESC, t.sort_order, t.name
           LIMIT ?""",
        (node_id, node_id, min_papers, limit),
    )
    return rows


def get_subfield_intersection_matrix(node_id: str, limit: int = 12) -> dict:
    """Build a subfield × subfield matrix using paper overlap and shared entities."""
    subfields = get_leaf_descendants(node_id, limit=limit, min_papers=1)
    if len(subfields) < 2:
        return {"subfields": subfields, "cells": {}, "max_strength": 0, "has_signal": False}

    subfield_ids = [row["id"] for row in subfields]
    placeholders = ",".join("?" for _ in subfield_ids)

    paper_rows = db.fetchall(
        f"""SELECT node_id, paper_id
            FROM paper_taxonomy
            WHERE node_id IN ({placeholders})""",
        tuple(subfield_ids),
    )
    paper_map: dict[str, set[str]] = defaultdict(set)
    for row in paper_rows:
        paper_map[row["node_id"]].add(row["paper_id"])

    entity_rows = db.fetchall(
        f"""SELECT pem.node_id, er.canonical_entity_id, ge.canonical_name
            FROM paper_entity_mentions pem
            JOIN entity_resolutions er ON er.entity_id = pem.entity_id
            JOIN graph_entities ge ON ge.id = er.canonical_entity_id
            WHERE pem.node_id IN ({placeholders})""",
        tuple(subfield_ids),
    )
    entity_map: dict[str, set[str]] = defaultdict(set)
    entity_name_map: dict[str, dict[str, str]] = defaultdict(dict)
    for row in entity_rows:
        entity_map[row["node_id"]].add(row["canonical_entity_id"])
        entity_name_map[row["node_id"]][row["canonical_entity_id"]] = row["canonical_name"]

    paper_title_rows = db.fetchall(
        f"""SELECT DISTINCT p.id, p.title
            FROM papers p
            JOIN paper_taxonomy pt ON pt.paper_id = p.id
            WHERE pt.node_id IN ({placeholders})""",
        tuple(subfield_ids),
    )
    paper_titles = {row["id"]: row["title"] for row in paper_title_rows}

    cells = {}
    max_strength = 0.0
    has_signal = False
    for row_a in subfields:
        for row_b in subfields:
            id_a = row_a["id"]
            id_b = row_b["id"]
            paper_overlap_ids = sorted(paper_map[id_a] & paper_map[id_b], reverse=True)
            entity_overlap_ids = sorted(entity_map[id_a] & entity_map[id_b])
            paper_overlap = len(paper_overlap_ids)
            entity_overlap = len(entity_overlap_ids)

            if id_a == id_b:
                strength = _intersection_strength(row_a["paper_count"], entity_overlap)
            else:
                strength = _intersection_strength(paper_overlap, entity_overlap)

            if id_a != id_b and (paper_overlap > 0 or entity_overlap > 0):
                has_signal = True
            max_strength = max(max_strength, strength)

            key = f"{id_a}|||{id_b}"
            cells[key] = {
                "paper_overlap": paper_overlap if id_a != id_b else row_a["paper_count"],
                "shared_entity_count": entity_overlap,
                "strength": strength,
                "sample_papers": [
                    {"id": pid, "title": paper_titles.get(pid, pid)}
                    for pid in paper_overlap_ids[:3]
                ],
                "shared_entities": [
                    entity_name_map[id_a].get(entity_id) or entity_name_map[id_b].get(entity_id) or entity_id
                    for entity_id in entity_overlap_ids[:5]
                ],
            }

    return {
        "subfields": subfields,
        "cells": cells,
        "max_strength": max_strength,
        "has_signal": has_signal,
    }


def get_method_dataset_matrix(node_id: str) -> dict:
    """Build the method x dataset matrix for a taxonomy node (and descendants).

    Returns:
        {
            "methods": ["m1", "m2", ...],
            "datasets": ["d1", "d2", ...],
            "metrics": ["acc", "f1", ...],
            "cells": {
                "m1|||d1|||acc": {"value": 95.2, "paper_id": "...", "is_sota": 0, ...},
                ...
            }
        }
    """
    rows = db.fetchall(
        """SELECT DISTINCT r.id, r.method_name, r.dataset_name, r.metric_name,
                  r.metric_value, r.metric_unit, r.paper_id,
                  r.is_sota, r.evidence_location
           FROM results r
           JOIN result_taxonomy rt ON rt.result_id = r.id
           JOIN taxonomy_nodes t ON rt.node_id = t.id
           WHERE t.id = ? OR t.id LIKE ? || '.%'
           ORDER BY r.method_name, r.dataset_name, r.metric_name""",
        (node_id, node_id),
    )

    methods = sorted(set(r["method_name"] for r in rows))
    datasets = sorted(set(r["dataset_name"] for r in rows))
    metrics = sorted(set(r["metric_name"] for r in rows if r["metric_name"]))

    cells = {}
    for r in rows:
        key = f"{r['method_name']}|||{r['dataset_name']}|||{r['metric_name'] or ''}"
        existing = cells.get(key)
        if _is_better_result(r, existing):
            cells[key] = {
                "value": r["metric_value"],
                "unit": r["metric_unit"],
                "paper_id": r["paper_id"],
                "is_sota": r["is_sota"],
                "evidence": r["evidence_location"],
            }

    return {
        "methods": methods,
        "datasets": datasets,
        "metrics": metrics,
        "cells": cells,
    }


def get_node_gaps(node_id: str) -> list[dict]:
    """Get matrix gaps for a node and descendants."""
    return db.fetchall(
        """SELECT mg.*
           FROM matrix_gaps mg
           JOIN taxonomy_nodes t ON mg.node_id = t.id
           WHERE t.id = ? OR t.id LIKE ? || '.%'
           ORDER BY mg.value_score DESC""",
        (node_id, node_id),
    )


def get_node_summary(node_id: str) -> dict | None:
    """Get the cached plain-language node summary."""
    return db.get_node_summary(node_id)


def get_node_signal_snapshot(node_id: str, paper_limit: int = 15) -> dict:
    """Collect summary signals for a node."""
    papers = get_node_papers(node_id, limit=paper_limit)
    graph_summary = graph.ensure_node_graph_summary(node_id)
    paper_clusters = get_node_paper_clusters(node_id)
    from db import opportunity_engine as opp
    opportunities = opp.ensure_node_opportunities(node_id)

    work_types = db.fetchall(
        """SELECT pi.work_type, COUNT(DISTINCT p.id) as count
           FROM papers p
           JOIN paper_taxonomy pt ON p.id = pt.paper_id
           JOIN taxonomy_nodes t ON pt.node_id = t.id
           JOIN paper_insights pi ON p.id = pi.paper_id
           WHERE (t.id = ? OR t.id LIKE ? || '.%')
             AND pi.work_type IS NOT NULL
             AND pi.work_type != ''
           GROUP BY pi.work_type
           ORDER BY count DESC, pi.work_type""",
        (node_id, node_id),
    )

    methods = db.fetchall(
        """SELECT r.method_name as name, COUNT(DISTINCT r.paper_id) as paper_count, COUNT(*) as result_count
           FROM results r
           JOIN result_taxonomy rt ON rt.result_id = r.id
           JOIN taxonomy_nodes t ON rt.node_id = t.id
           WHERE t.id = ? OR t.id LIKE ? || '.%'
           GROUP BY r.method_name
           ORDER BY paper_count DESC, result_count DESC, r.method_name
           LIMIT 12""",
        (node_id, node_id),
    )

    datasets = db.fetchall(
        """SELECT r.dataset_name as name, COUNT(DISTINCT r.paper_id) as paper_count, COUNT(*) as result_count
           FROM results r
           JOIN result_taxonomy rt ON rt.result_id = r.id
           JOIN taxonomy_nodes t ON rt.node_id = t.id
           WHERE t.id = ? OR t.id LIKE ? || '.%'
           GROUP BY r.dataset_name
           ORDER BY paper_count DESC, result_count DESC, r.dataset_name
           LIMIT 12""",
        (node_id, node_id),
    )

    limitations = Counter()
    open_questions = Counter()
    for paper in papers:
        for item in paper.get("limitations", []):
            limitations[item.strip()] += 1
        for item in paper.get("open_questions", []):
            open_questions[item.strip()] += 1

    return {
        "paper_count": _count_subtree_papers(node_id),
        "result_count": _count_subtree_results(node_id),
        "children": get_children(node_id),
        "papers": papers,
        "work_types": work_types,
        "methods": methods,
        "datasets": datasets,
        "limitations": [text for text, _ in limitations.most_common(12)],
        "open_questions": [text for text, _ in open_questions.most_common(12)],
        "matrix_gaps": get_node_gaps(node_id),
        "graph_summary": graph_summary or {},
        "paper_clusters": paper_clusters,
        "opportunities": opportunities,
    }


def ensure_node_summary(node_id: str, force: bool = False) -> dict | None:
    """Return a cached node summary, generating it when missing or stale."""
    node = get_node(node_id)
    if not node:
        return None

    snapshot = get_node_signal_snapshot(node_id)
    existing = db.get_node_summary(node_id)
    if existing and not force:
        if (
            existing.get("paper_count") == snapshot["paper_count"]
            and existing.get("result_count") == snapshot["result_count"]
        ):
            return existing

    if snapshot["paper_count"] == 0 and not snapshot["children"]:
        return None

    from agents.domain_summary_agent import generate_domain_summary, fallback_domain_summary

    try:
        summary, _tokens = generate_domain_summary(node, snapshot)
    except Exception as e:
        print(f"[TAXONOMY] Domain summary generation failed for {node_id}: {e}", flush=True)
        summary = fallback_domain_summary(node, snapshot)

    summary["node_id"] = node_id
    summary["audience"] = "general"
    summary["generated_from_papers"] = [paper["id"] for paper in snapshot["papers"]]
    summary["paper_count"] = snapshot["paper_count"]
    summary["result_count"] = snapshot["result_count"]

    db.upsert_node_summary(summary)
    return db.get_node_summary(node_id)


def assign_paper_to_node(paper_id: str, node_id: str, confidence: float = 1.0):
    """Assign a paper to a taxonomy node."""
    db.execute(
        """INSERT OR REPLACE INTO paper_taxonomy (paper_id, node_id, confidence)
           VALUES (?, ?, ?)""",
        (paper_id, node_id, confidence),
    )
    db.commit()


def insert_result(result: dict) -> int:
    """Insert a (method, dataset, metric, value) result row."""
    cur = db.execute(
        """INSERT INTO results
           (paper_id, node_id, method_name, dataset_name, metric_name,
            metric_value, metric_unit, is_sota, evidence_location, conditions)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (result["paper_id"], result.get("node_id"), result["method_name"],
         result["dataset_name"], result["metric_name"],
         result.get("metric_value"), result.get("metric_unit"),
         result.get("is_sota", 0), result.get("evidence_location"),
         json.dumps(result.get("conditions", {}))),
    )
    db.commit()
    return cur.lastrowid


def assign_result_to_node(result_id: int, node_id: str, commit: bool = True):
    """Assign a result row to a taxonomy node."""
    db.execute(
        """INSERT OR IGNORE INTO result_taxonomy (result_id, node_id)
           VALUES (?, ?)""",
        (result_id, node_id),
    )
    if commit:
        db.commit()


def backfill_result_taxonomy():
    """Ensure existing result rows have taxonomy links."""
    rows = db.fetchall(
        """SELECT r.id, r.paper_id, r.node_id
           FROM results r
           LEFT JOIN result_taxonomy rt ON rt.result_id = r.id
           WHERE rt.result_id IS NULL"""
    )

    for row in rows:
        node_rows = db.fetchall(
            "SELECT node_id FROM paper_taxonomy WHERE paper_id=? ORDER BY node_id",
            (row["paper_id"],),
        )
        node_ids = [item["node_id"] for item in node_rows]
        if not node_ids and row.get("node_id"):
            node_ids = [row["node_id"]]
        for node_id in node_ids:
            assign_result_to_node(row["id"], node_id, commit=False)
    if rows:
        db.commit()


def insert_matrix_gap(gap: dict) -> int:
    """Insert a gap found in the method x dataset matrix."""
    cur = db.execute(
        """INSERT INTO matrix_gaps
           (node_id, method_name, dataset_name, metric_name,
            gap_description, research_proposal, value_score, evidence_paper_ids)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (gap["node_id"], gap["method_name"], gap["dataset_name"],
         gap.get("metric_name"), gap["gap_description"],
         gap.get("research_proposal"), gap.get("value_score"),
         json.dumps(gap.get("evidence_paper_ids", []))),
    )
    db.commit()
    return cur.lastrowid


# ── Dynamic taxonomy node creation ─────────────────────────────────

def create_dynamic_node(
    node_id: str,
    name: str,
    parent_id: str,
    description: str = "",
    sort_order: int = 0,
) -> dict | None:
    """Create a new taxonomy node dynamically (not from the predefined list).

    Returns the created node dict, or None if it already exists.
    """
    existing = get_node(node_id)
    if existing:
        return None

    parent = get_node(parent_id)
    depth = (parent["depth"] + 1) if parent else 0

    db.execute(
        """INSERT OR IGNORE INTO taxonomy_nodes
           (id, name, parent_id, depth, description, sort_order)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (node_id, name, parent_id, depth, description, sort_order),
    )
    db.commit()
    return get_node(node_id)


def get_direct_paper_count(node_id: str) -> int:
    """Return the number of papers directly assigned to a node (not subtree)."""
    row = db.fetchone(
        "SELECT COUNT(*) as c FROM paper_taxonomy WHERE node_id = ?",
        (node_id,),
    )
    return row["c"] if row else 0


def is_leaf_node(node_id: str) -> bool:
    """Check whether a node is a leaf (has no children)."""
    row = db.fetchone(
        "SELECT id FROM taxonomy_nodes WHERE parent_id = ? LIMIT 1",
        (node_id,),
    )
    return row is None


def get_recently_created_nodes(limit: int = 20) -> list[dict]:
    """Get the most recently created dynamic taxonomy nodes (by created_at)."""
    return db.fetchall(
        """SELECT t.*,
                  COUNT(DISTINCT pt.paper_id) AS paper_count
           FROM taxonomy_nodes t
           LEFT JOIN paper_taxonomy pt ON pt.node_id = t.id
           GROUP BY t.id
           ORDER BY t.created_at DESC
           LIMIT ?""",
        (limit,),
    )
