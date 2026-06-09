"""Binding manuscript-writing standards for PaperOrchestra outputs."""

from __future__ import annotations


WRITING_STANDARD_VERSION = "paperorchestra_manuscript_writing_standard_v1_2026_06_02"


ABSTRACT_STANDARD = """Abstract standard:
- Follow this narrative order: background trend -> concrete problem -> limitation of existing methods -> proposed method -> core mechanism -> main results.
- The first one or two sentences must enter the concrete problem directly; avoid long generic preambles.
- When the method name first appears, write the full name and abbreviation.
- Do not over-explain method details in the abstract.
- Do not make unsupported broad generalization claims.
- Do not expose non-core details that weaken presentation, such as tiny sample counts, temporary protocols, or unfinished experiments.
- State results naturally using standard effect language such as percentage points, absolute improvement, relative reduction, or cost reduction.
- If evidence is from a subset, controlled setting, materialized-trace benchmark, or case study, the abstract must state that boundary and must not imply a complete benchmark.
- Avoid excessive colons, semicolons, rhetorical questions, and multi-clause sentences.
- Do not duplicate the abstract, title, or section heading; exactly one abstract environment is allowed.
- Forbidden unless explicitly supported by strong evidence: "This is the first", "Comprehensive experiments show", "Extensive experiments demonstrate", "Universal", "General"."""


INTRODUCTION_STANDARD = """Introduction standard:
- The Introduction must do four jobs: explain why the problem matters, explain why existing methods are insufficient, state what this paper solves, and summarize how the method solves it with supported results.
- Preferred structure: background trend; concrete challenge; two or three core weaknesses of existing methods; proposed method and corresponding mechanism; short result paragraph; contribution summary.
- Do not add standalone Question/Motivation/Answer/Result mini-headings or rhetorical-question paragraphs.
- If using Problem I / II / III framing, each problem must correspond to a real failure mode, the total number should normally be at most three, and every problem must have a matching design in the Method.
- Do not split problems mechanically just to create a numbered structure, and keep problem paragraphs compact.
- After introducing multiple problems, respond to them explicitly with sentences such as "For Problem I, ..." and state the mechanism and why it mitigates the problem.
- The result paragraph should be short, typically two or three sentences, and must not repeat the abstract verbatim."""


CONTRIBUTIONS_STANDARD = """Contribution standard:
- Use three or four contribution bullets.
- Preferred order: identify or define the problem; formulate the task or analyze the failure mode; propose the method; evaluate with completed evidence.
- Each contribution should be a plain sentence beginning with We identify/formulate/propose/construct/evaluate; do not use bold or italic mini labels inside bullets.
- Only claim work that is actually completed by this paper.
- Avoid vague claims, inflated claims, or repackaging ordinary experiments as major contributions.
- Do not use the phrase "training-free"; say "inference-time" or "without model-weight updates" only when that concrete scope is necessary.
- Avoid "We conduct extensive experiments", "We are the first to", and "We comprehensively study" unless the evidence truly supports them."""


RELATED_WORK_STANDARD = """Related Work standard:
- Related Work should position the paper, not dump citations.
- Each subsection should follow: area introduction -> representative method categories -> relationship or difference from this paper.
- Subsection titles should be Title Case noun phrases, one to three words, short enough for two-column layout, and grammatically consistent.
- Organize citations by category. Do not write large undifferentiated citation clusters such as \\cite{a,b,c,d,e,f,g,h}.
- Preferred citation pattern: describe one category and cite one or two papers, then describe another category and cite another one or two papers.
- Each Related Work subsection should end with a gap sentence: what this paper does differently, what it adds, and what it is not trying to solve.
- By default, keep each Related Work subsection to one dense paragraph unless the paper is a survey or has unusually large space."""


METHOD_STANDARD = """Method standard:
- The Method section should make the method understandable and reproducible.
- Use a structure appropriate to the paper, usually three to five subsections such as Problem Formulation, Framework Design, Core Algorithm, Inference/Optimization, and Implementation Details.
- Avoid too many subsections, avoid moving experimental protocol into Method, and avoid baseline evaluation inside Method.
- Every important equation must be followed by an explanation of the input, output, variables, purpose of each term, and which problem the equation addresses.
- Prefer intuitive notation and readable piecewise definitions over scattered indicator functions when possible.
- If the method has a clear procedure, include an algorithm block with explicit inputs and outputs, at most 15--20 lines, consistent with the equations.
- Algorithm blocks must not contain experiment results and must not be an empty generic subsection.
- The Method section must not contain experiment numbers, significance analysis, baseline criticism, TODOs, undefined variables, or assumptions inconsistent with experiments."""


MANUSCRIPT_WRITING_STANDARD_TEXT = "\n\n".join(
    [
        f"Binding manuscript writing standard: {WRITING_STANDARD_VERSION}.",
        ABSTRACT_STANDARD,
        INTRODUCTION_STANDARD,
        CONTRIBUTIONS_STANDARD,
        RELATED_WORK_STANDARD,
        METHOD_STANDARD,
    ]
)


def section_style_rules(section_title: str) -> str:
    """Return section-specific style rules for compact section-writing prompts."""
    title = (section_title or "").lower()
    if "intro" in title or "related" in title:
        return INTRODUCTION_STANDARD + "\n\n" + CONTRIBUTIONS_STANDARD + "\n\n" + RELATED_WORK_STANDARD
    if "method" in title:
        return METHOD_STANDARD
    if "discussion" in title or "conclusion" in title:
        return (
            "Discussion and Conclusion standard:\n"
            "- Interpret only completed evidence and avoid unsupported extrapolation.\n"
            "- State limitations compactly without apology-like internal workflow language.\n"
            "- Do not introduce new methods, new baselines, or new results."
        )
    return MANUSCRIPT_WRITING_STANDARD_TEXT
