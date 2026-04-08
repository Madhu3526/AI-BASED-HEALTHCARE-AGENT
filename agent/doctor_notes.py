"""
Doctor's Clinical Notes Integration
Parses free-text notes entered by the doctor and merges them
with AI model predictions to produce a richer combined report.

Notes can contain:
  - Patient symptoms and history
  - Vital signs
  - Physical examination findings
  - Pre-existing conditions
  - Current medications

The combiner re-weights AI predictions based on note keywords,
flags contradictions, and adds note context to the final report.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Keyword → disease relevance map ──────────────────────────────────────────
# Maps clinical note keywords to diseases they increase/decrease confidence for.
_KEYWORD_BOOST: Dict[str, Dict[str, float]] = {
    # Respiratory symptoms
    "cough":            {"Pneumonia": +0.15, "Infiltration": +0.10, "Atelectasis": +0.05},
    "fever":            {"Pneumonia": +0.20, "Consolidation": +0.15, "Infiltration": +0.10},
    "dyspnea":          {"Effusion": +0.10, "Edema": +0.15, "Pneumothorax": +0.10, "Emphysema": +0.10},
    "shortness of breath": {"Edema": +0.15, "Effusion": +0.10, "Pneumothorax": +0.10},
    "wheezing":         {"Emphysema": +0.15, "Atelectasis": +0.10},
    "haemoptysis":      {"Mass": +0.20, "Pneumonia": +0.10},
    "hemoptysis":       {"Mass": +0.20, "Pneumonia": +0.10},
    "chest pain":       {"Pneumothorax": +0.15, "Effusion": +0.10, "Pleural_Thickening": +0.10},
    "pleuritic":        {"Effusion": +0.15, "Pneumothorax": +0.10},

    # Cardiac symptoms
    "edema":            {"Cardiomegaly": +0.15, "Edema": +0.20},
    "palpitations":     {"Cardiomegaly": +0.10},
    "orthopnea":        {"Edema": +0.20, "Cardiomegaly": +0.15},

    # Signs
    "clubbing":         {"Fibrosis": +0.15, "Mass": +0.10},
    "barrel chest":     {"Emphysema": +0.20},
    "crackles":         {"Fibrosis": +0.15, "Edema": +0.10, "Pneumonia": +0.10},

    # History
    "smoking":          {"Emphysema": +0.15, "Mass": +0.15, "Nodule": +0.10},
    "smoker":           {"Emphysema": +0.15, "Mass": +0.15, "Nodule": +0.10},
    "copd":             {"Emphysema": +0.20, "Atelectasis": +0.10},
    "asbestos":         {"Pleural_Thickening": +0.25, "Mass": +0.10},
    "cancer":           {"Mass": +0.25, "Nodule": +0.15},
    "malignancy":       {"Mass": +0.25, "Nodule": +0.15},
    "heart failure":    {"Cardiomegaly": +0.20, "Edema": +0.20, "Effusion": +0.15},
    "hypertension":     {"Cardiomegaly": +0.10},
    "diabetes":         {"Pneumonia": +0.10, "Infiltration": +0.05},
    "immunocompromised": {"Pneumonia": +0.20, "Infiltration": +0.15},
    "hiv":              {"Pneumonia": +0.20, "Infiltration": +0.15},
    "tb":               {"Infiltration": +0.20, "Nodule": +0.10},
    "tuberculosis":     {"Infiltration": +0.20, "Nodule": +0.10},
    "trauma":           {"Pneumothorax": +0.25},
    "pneumothorax":     {"Pneumothorax": +0.30},
    "reflux":           {"Hernia": +0.20},
    "heartburn":        {"Hernia": +0.20},
    "dysphagia":        {"Hernia": +0.15},
}

# Keywords that reduce confidence for a disease
_KEYWORD_SUPPRESS: Dict[str, Dict[str, float]] = {
    "no fever":         {"Pneumonia": -0.10, "Infiltration": -0.05},
    "no cough":         {"Pneumonia": -0.10},
    "no chest pain":    {"Pneumothorax": -0.10, "Effusion": -0.05},
    "non-smoker":       {"Emphysema": -0.15, "Mass": -0.10},
    "no history":       {"Mass": -0.05},
    "resolved":         {"Pneumonia": -0.10, "Consolidation": -0.10},
}


# ── Structured note parser ─────────────────────────────────────────────────────
@dataclass
class ParsedNotes:
    """Structured representation of parsed doctor's notes."""
    raw_text:           str
    symptoms:           List[str] = field(default_factory=list)
    history:            List[str] = field(default_factory=list)
    vitals:             Dict[str, str] = field(default_factory=dict)
    medications:        List[str] = field(default_factory=list)
    examination:        List[str] = field(default_factory=list)
    matched_keywords:   List[str] = field(default_factory=list)
    boost_map:          Dict[str, float] = field(default_factory=dict)  # disease → delta


def parse_doctor_notes(notes_text: str) -> ParsedNotes:
    """
    Parse free-text clinical notes into a structured object.

    Extracts:
      - Symptoms and signs mentioned
      - Clinical history keywords
      - Vital signs (BP, HR, SpO2, Temp)
      - Drug names (basic heuristic)

    Returns:
      ParsedNotes with matched_keywords and boost_map populated.
    """
    if not notes_text or not notes_text.strip():
        return ParsedNotes(raw_text="")

    text_lower = notes_text.lower()

    # ── Vital signs (regex) ────────────────────────────────────────────────
    vitals = {}
    bp_match  = re.search(r"bp\s*[:\-]?\s*(\d{2,3}/\d{2,3})", text_lower)
    hr_match  = re.search(r"hr\s*[:\-]?\s*(\d{2,3})\s*bpm", text_lower)
    spo2_match= re.search(r"spo2\s*[:\-]?\s*(\d{2,3})\s*%", text_lower)
    temp_match= re.search(r"temp\s*[:\-]?\s*(\d{2,3}(?:\.\d)?)\s*[°c°f]?", text_lower)
    rr_match  = re.search(r"rr\s*[:\-]?\s*(\d{1,2})\s*/min", text_lower)
    if bp_match:   vitals["BP"]    = bp_match.group(1)
    if hr_match:   vitals["HR"]    = hr_match.group(1) + " bpm"
    if spo2_match: vitals["SpO2"]  = spo2_match.group(1) + "%"
    if temp_match: vitals["Temp"]  = temp_match.group(1) + "°"
    if rr_match:   vitals["RR"]    = rr_match.group(1) + "/min"

    # ── Keyword matching → disease boost map ──────────────────────────────
    matched  = []
    boost    = {}

    for kw, boosts in _KEYWORD_BOOST.items():
        if kw in text_lower:
            matched.append(kw)
            for disease, delta in boosts.items():
                boost[disease] = boost.get(disease, 0.0) + delta

    for kw, suppresses in _KEYWORD_SUPPRESS.items():
        if kw in text_lower:
            matched.append(f"[suppress] {kw}")
            for disease, delta in suppresses.items():
                boost[disease] = boost.get(disease, 0.0) + delta  # delta is negative

    # Clamp boosts to [-0.5, +0.5] so notes don't completely override the model
    boost = {d: max(-0.5, min(0.5, v)) for d, v in boost.items()}

    # ── Heuristic section splitter ────────────────────────────────────────
    symptoms    = _extract_section(notes_text, ["symptoms", "complaint", "presenting"])
    history     = _extract_section(notes_text, ["history", "background", "hx"])
    medications = _extract_section(notes_text, ["medications", "drugs", "meds", "rx"])
    examination = _extract_section(notes_text, ["examination", "findings", "exam"])

    return ParsedNotes(
        raw_text        = notes_text,
        symptoms        = symptoms,
        history         = history,
        vitals          = vitals,
        medications     = medications,
        examination     = examination,
        matched_keywords= matched,
        boost_map       = boost,
    )


def _extract_section(text: str, keywords: List[str]) -> List[str]:
    """
    Naively extract bullet points or comma-separated items
    that appear after a section heading keyword.
    """
    text_lower = text.lower()
    for kw in keywords:
        idx = text_lower.find(kw)
        if idx == -1:
            continue
        # Grab up to 300 chars after the keyword
        snippet = text[idx + len(kw): idx + len(kw) + 300]
        # Split on newlines or commas
        items = [
            item.strip(" -•:,\n\r")
            for item in re.split(r"[\n,]", snippet)
            if item.strip(" -•:,\n\r")
        ]
        return [i for i in items if len(i) > 2][:10]
    return []


# ── Note + prediction combiner ─────────────────────────────────────────────────
def combine_notes_with_predictions(
    raw_probabilities: Dict[str, float],
    parsed_notes: ParsedNotes,
    note_weight: float = 0.3,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Merge model probabilities with note-derived boosts.

    Combined probability = model_prob + note_weight * boost
    Clamped to [0, 1].

    Args:
        raw_probabilities: {disease: probability} from model
        parsed_notes:      output of parse_doctor_notes()
        note_weight:       how much to trust the notes (0 = ignore, 1 = full)

    Returns:
        (adjusted_probabilities, explanation_lines)
    """
    adjusted = {}
    explanations = []

    for disease, prob in raw_probabilities.items():
        boost    = parsed_notes.boost_map.get(disease, 0.0)
        delta    = note_weight * boost
        new_prob = max(0.0, min(1.0, prob + delta))
        adjusted[disease] = new_prob

        if abs(delta) >= 0.03:   # only explain meaningful changes
            direction = "increased" if delta > 0 else "decreased"
            explanations.append(
                f"{disease}: {prob*100:.1f}% -> {new_prob*100:.1f}% "
                f"({direction} based on clinical notes)"
            )

    return adjusted, explanations


# ── Vitals risk flags ──────────────────────────────────────────────────────────
def flag_vitals(vitals: Dict[str, str]) -> List[str]:
    """Return list of clinical alert strings for abnormal vitals."""
    flags = []

    spo2_str = vitals.get("SpO2", "")
    spo2_match = re.search(r"(\d+)", spo2_str)
    if spo2_match:
        spo2 = int(spo2_match.group(1))
        if spo2 < 90:
            flags.append(f"CRITICAL: SpO2 {spo2}% — severe hypoxia, immediate intervention")
        elif spo2 < 94:
            flags.append(f"WARNING: SpO2 {spo2}% — supplemental oxygen required")

    hr_str = vitals.get("HR", "")
    hr_match = re.search(r"(\d+)", hr_str)
    if hr_match:
        hr = int(hr_match.group(1))
        if hr > 120:
            flags.append(f"WARNING: HR {hr} bpm — tachycardia, investigate cause")
        elif hr < 50:
            flags.append(f"WARNING: HR {hr} bpm — bradycardia")

    temp_str = vitals.get("Temp", "")
    temp_match = re.search(r"(\d+(?:\.\d)?)", temp_str)
    if temp_match:
        temp = float(temp_match.group(1))
        if temp >= 38.5:
            flags.append(f"WARNING: Temp {temp}° — fever, consider infectious aetiology")
        elif temp <= 35.5:
            flags.append(f"WARNING: Temp {temp}° — hypothermia")

    return flags
