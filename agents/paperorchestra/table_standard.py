"""Binding table policy for PaperOrchestra manuscripts."""

from __future__ import annotations

TABLE_STANDARD_VERSION = "paperorchestra_table_standard_v1_2026_05_29"

TABLE_RULES = {
    "style": "top_venue_compact_booktabs",
    "numeric_first": True,
    "header_background": "gray!18",
    "own_method_background": "red!7",
    "continuous_highlight": True,
    "booktabs_required": True,
    "recommended_packages": ["booktabs", "xcolor[table]", "array", "tabularx"],
    "structure": {
        "top_rule": "\\toprule",
        "header_rule": "\\midrule",
        "bottom_rule": "\\bottomrule",
        "avoid_vertical_lines_by_default": True,
        "allow_sparse_group_separators": True,
    },
    "layout": {
        "wide_tables": r"Use table* + tabularx with \textwidth to fill text width; avoid @{\extracolsep{\fill}} because it breaks continuous row shading.",
        "single_column_tables": r"Use tabularx with \linewidth when the table is narrow enough.",
        "arraystretch": "1.04-1.08",
        "headers": r"Short symbolic headers such as Acc., Std., Tok., Lat., Route, $\Delta$.",
    },
    "highlighting": {
        "header": r"Use \rowcolor{gray!18} for the header row.",
        "own_method": r"Use \rowcolor{red!7} across the entire proposed-method row; do not color only scattered cells.",
        "bold": "Use bold sparingly, usually only for the proposed method or primary best value.",
        "no_cell_only_highlight_for_main_method": True,
    },
    "content": {
        "no_long_text_columns": True,
        "move_interpretation_to_prose": True,
        "prefer_derived_numeric_columns": ["Std.", r"95\% CI", r"$\Delta$", "Rel.", "Range"],
        "boolean_values": r"Use \checkmark and -- rather than Yes/No.",
    },
}


def table_policy_manifest() -> dict:
    return {
        "standard_version": TABLE_STANDARD_VERSION,
        "table_rules": TABLE_RULES,
    }
