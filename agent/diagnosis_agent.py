"""
AI Diagnosis Agent
Combines model prediction probabilities, doctor's clinical notes,
and RAG-retrieved medical knowledge to produce a unified report.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from data.dataset import DISEASE_LABELS
from rag.knowledge_base import MedicalKnowledgeBase
from agent.doctor_notes import (
    ParsedNotes,
    parse_doctor_notes,
    combine_notes_with_predictions,
    flag_vitals,
)


# ── Data structures ────────────────────────────────────────────────────────────
@dataclass
class DiseaseResult:
    """Single disease prediction with clinical context."""
    name:           str
    probability:    float
    model_prob:     float             # raw model probability before note adjustment
    severity:       str               # "low" | "moderate" | "high" | "critical"
    note_adjusted:  bool = False      # True if notes changed the probability
    symptoms:       List[str]     = field(default_factory=list)
    risk_factors:   List[str]     = field(default_factory=list)
    treatment:      List[str]     = field(default_factory=list)
    follow_up:      str           = ""
    overview:       str           = ""


@dataclass
class DiagnosisReport:
    """Complete AI-generated diagnosis report for a doctor."""
    patient_id:          str
    model_used:          str
    predicted_diseases:  List[DiseaseResult]
    primary_impression:  str
    urgency_level:       str           # "routine" | "urgent" | "emergency"
    recommended_actions: List[str]
    disclaimer:          str
    raw_probabilities:   Dict[str, float] = field(default_factory=dict)
    # Note-specific fields
    doctor_notes:        str            = ""
    parsed_notes:        Optional[ParsedNotes] = None
    note_adjustments:    List[str]      = field(default_factory=list)  # explanation strings
    vitals_flags:        List[str]      = field(default_factory=list)  # abnormal vital alerts


# ── Severity / urgency logic ───────────────────────────────────────────────────
_CRITICAL_CONDITIONS = {"Pneumothorax", "Edema"}
_HIGH_CONDITIONS     = {"Mass", "Pneumonia", "Effusion", "Consolidation"}
_MODERATE_CONDITIONS = {"Cardiomegaly", "Emphysema", "Fibrosis", "Pleural_Thickening"}

def _classify_severity(disease: str, probability: float) -> str:
    if disease in _CRITICAL_CONDITIONS and probability > 0.6:
        return "critical"
    if disease in _HIGH_CONDITIONS and probability > 0.5:
        return "high"
    if disease in _MODERATE_CONDITIONS and probability > 0.4:
        return "moderate"
    return "low"

def _overall_urgency(diseases: List[DiseaseResult]) -> str:
    severities = [d.severity for d in diseases]
    if "critical" in severities:
        return "emergency"
    if "high" in severities:
        return "urgent"
    return "routine"


# ── Diagnosis Agent ────────────────────────────────────────────────────────────
class DiagnosisAgent:
    """
    AI Diagnosis Agent pipeline:
      1. Run model on X-ray image tensor → raw probabilities
      2. Parse doctor's clinical notes → keyword boosts
      3. Combine: adjust probabilities based on clinical notes
      4. Filter detections above threshold
      5. Retrieve disease context from RAG knowledge base
      6. Build DiagnosisReport

    Usage::

        agent = DiagnosisAgent(model, knowledge_base)

        # Without notes:
        report = agent.diagnose(image_tensor, patient_id="PT-001")

        # With doctor's notes:
        notes = \"\"\"
            55M, smoker. Presenting with 3 weeks cough and haemoptysis.
            No fever. SpO2: 96%. HR: 88 bpm.
            History: COPD, ex-smoker 10 pack-years.
        \"\"\"
        report = agent.diagnose(image_tensor, patient_id="PT-001", doctor_notes=notes)
        print(agent.format_report(report))
    """

    def __init__(
        self,
        model: nn.Module,
        knowledge_base: MedicalKnowledgeBase,
        threshold: float = 0.45,
        top_k: int = 5,
        model_name: str = "ResNet50",
        note_weight: float = 0.3,
        device: Optional[str] = None,
    ):
        self.model        = model
        self.kb           = knowledge_base
        self.threshold    = threshold
        self.top_k        = top_k
        self.model_name   = model_name
        self.note_weight  = note_weight   # how much notes influence probabilities (0–1)
        self.device       = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    # ── Inference ──────────────────────────────────────────────────────────────
    def predict(self, image_tensor: torch.Tensor) -> Dict[str, float]:
        """Run model and return {disease: probability} dict."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            probs = torch.sigmoid(self.model(image_tensor)).squeeze().cpu().numpy()
        return {label: float(p) for label, p in zip(DISEASE_LABELS, probs)}

    # ── Full pipeline ──────────────────────────────────────────────────────────
    def diagnose(
        self,
        image_tensor: torch.Tensor,
        patient_id: str = "UNKNOWN",
        doctor_notes: str = "",
    ) -> DiagnosisReport:
        """
        Full diagnostic pipeline.

        Args:
            image_tensor:  preprocessed chest X-ray (1, 3, 224, 224)
            patient_id:    patient identifier
            doctor_notes:  free-text clinical notes from the doctor (optional)

        Returns:
            DiagnosisReport with AI predictions + note context combined
        """
        # Step 1 — Model inference
        raw_probs = self.predict(image_tensor)

        # Step 2 — Parse doctor's notes
        parsed     = parse_doctor_notes(doctor_notes) if doctor_notes.strip() else ParsedNotes(raw_text="")
        vitals_flags = flag_vitals(parsed.vitals)

        # Step 3 — Combine model probabilities with note boosts
        if doctor_notes.strip():
            adjusted_probs, note_adjustments = combine_notes_with_predictions(
                raw_probs, parsed, self.note_weight
            )
        else:
            adjusted_probs   = raw_probs
            note_adjustments = []

        # Step 4 — Filter and rank detections using adjusted probabilities
        detected = [
            (name, adjusted_probs[name], raw_probs[name])
            for name in sorted(adjusted_probs, key=adjusted_probs.get, reverse=True)
            if adjusted_probs[name] >= self.threshold
        ][:self.top_k]

        if not detected:
            return self._no_finding_report(
                patient_id, raw_probs, adjusted_probs,
                doctor_notes, parsed, vitals_flags, note_adjustments,
            )

        # Step 5 — Retrieve RAG context
        disease_results: List[DiseaseResult] = []
        for disease_name, adj_prob, model_prob in detected:
            info     = self.kb.get_disease_info(disease_name) or {}
            severity = _classify_severity(disease_name, adj_prob)
            disease_results.append(DiseaseResult(
                name          = disease_name,
                probability   = round(adj_prob, 4),
                model_prob    = round(model_prob, 4),
                severity      = severity,
                note_adjusted = abs(adj_prob - model_prob) >= 0.01,
                symptoms      = info.get("symptoms",     []),
                risk_factors  = info.get("risk_factors", []),
                treatment     = info.get("treatment",    []),
                follow_up     = info.get("follow_up",    "Follow up with physician."),
                overview      = info.get("overview",     ""),
            ))

        # Step 6 — Compose report
        urgency    = _overall_urgency(disease_results)
        # Bump urgency if vitals show critical values
        if any("CRITICAL" in f for f in vitals_flags):
            urgency = "emergency"
        elif any("WARNING" in f for f in vitals_flags) and urgency == "routine":
            urgency = "urgent"

        actions    = self._generate_actions(disease_results, urgency, parsed)
        impression = self._generate_impression(disease_results, urgency, bool(doctor_notes.strip()))

        return DiagnosisReport(
            patient_id          = patient_id,
            model_used          = self.model_name,
            predicted_diseases  = disease_results,
            primary_impression  = impression,
            urgency_level       = urgency,
            recommended_actions = actions,
            disclaimer          = _DISCLAIMER,
            raw_probabilities   = adjusted_probs,
            doctor_notes        = doctor_notes,
            parsed_notes        = parsed,
            note_adjustments    = note_adjustments,
            vitals_flags        = vitals_flags,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _generate_impression(
        self,
        diseases: List[DiseaseResult],
        urgency: str,
        notes_provided: bool,
    ) -> str:
        primary = diseases[0]
        urgency_str = {
            "emergency": "— URGENT ATTENTION REQUIRED",
            "urgent":    "— Requires prompt evaluation",
            "routine":   "— Routine clinical evaluation suggested",
        }.get(urgency, "")

        note_str = " (probability adjusted using clinical notes)" if primary.note_adjusted else ""
        impression = (
            f"AI model ({self.model_name}) indicates probable "
            f"**{primary.name}** ({primary.probability*100:.1f}%){note_str}. "
        )
        if len(diseases) > 1:
            others = ", ".join(f"{d.name} ({d.probability*100:.1f}%)" for d in diseases[1:])
            impression += f"Additional findings: {others}. "
        impression += urgency_str
        return impression

    def _generate_actions(
        self,
        diseases: List[DiseaseResult],
        urgency: str,
        parsed_notes: ParsedNotes,
    ) -> List[str]:
        actions = []

        if urgency == "emergency":
            actions.append("IMMEDIATE clinical evaluation by attending physician")

        seen = set()
        for d in diseases:
            for step in d.treatment[:3]:
                if step not in seen:
                    actions.append(step)
                    seen.add(step)

        follow_ups = list({d.follow_up for d in diseases if d.follow_up})
        actions.extend(follow_ups[:2])

        # Note-specific additions
        if parsed_notes.vitals:
            actions.append("Monitor vital signs closely — abnormal vitals documented in notes")
        if parsed_notes.medications:
            actions.append("Review current medications for interactions / pulmonary side effects")

        actions.append("Review with a radiologist for final X-ray interpretation")
        actions.append("Correlate AI findings with clinical history and examination")
        return actions

    def _no_finding_report(
        self,
        patient_id: str,
        raw_probs: Dict[str, float],
        adjusted_probs: Dict[str, float],
        doctor_notes: str,
        parsed_notes: ParsedNotes,
        vitals_flags: List[str],
        note_adjustments: List[str],
    ) -> DiagnosisReport:
        top = max(adjusted_probs.items(), key=lambda x: x[1])
        return DiagnosisReport(
            patient_id          = patient_id,
            model_used          = self.model_name,
            predicted_diseases  = [],
            primary_impression  = (
                f"No significant pathology detected above threshold "
                f"({self.threshold*100:.0f}%). "
                f"Highest: {top[0]} ({top[1]*100:.1f}%). "
                "Clinical correlation recommended."
            ),
            urgency_level       = "emergency" if any("CRITICAL" in f for f in vitals_flags) else "routine",
            recommended_actions = [
                "Clinical correlation with patient history and notes",
                "Repeat imaging if symptoms persist or worsen",
                "Consult radiologist if clinically indicated",
            ],
            disclaimer          = _DISCLAIMER,
            raw_probabilities   = adjusted_probs,
            doctor_notes        = doctor_notes,
            parsed_notes        = parsed_notes,
            note_adjustments    = note_adjustments,
            vitals_flags        = vitals_flags,
        )

    # ── Formatted text output ──────────────────────────────────────────────────
    def format_report(self, report: DiagnosisReport) -> str:
        sep   = "─" * 60
        lines = [
            sep,
            f"  AI DIAGNOSTIC REPORT — Patient: {report.patient_id}",
            sep,
            f"  Model        : {report.model_used}",
            f"  Urgency      : {report.urgency_level.upper()}",
            "",
            "  IMPRESSION",
            f"  {report.primary_impression}",
            "",
        ]

        # Vitals flags
        if report.vitals_flags:
            lines.append("  VITALS ALERTS")
            for flag in report.vitals_flags:
                lines.append(f"  ⚠  {flag}")
            lines.append("")

        # Note adjustments
        if report.note_adjustments:
            lines.append("  NOTE-BASED PROBABILITY ADJUSTMENTS")
            for adj in report.note_adjustments:
                lines.append(f"  → {adj}")
            lines.append("")

        if report.predicted_diseases:
            lines.append("  DETECTED CONDITIONS")
            for i, d in enumerate(report.predicted_diseases, 1):
                adj_marker = f" [note-adjusted from {d.model_prob*100:.1f}%]" if d.note_adjusted else ""
                lines.append(
                    f"  {i}. {d.name:<25} "
                    f"Prob: {d.probability*100:5.1f}%{adj_marker}  |  Severity: {d.severity}"
                )
            lines.append("")

        lines.append("  RECOMMENDED ACTIONS")
        for j, action in enumerate(report.recommended_actions, 1):
            lines.append(f"  {j:2}. {action}")

        if report.doctor_notes.strip():
            lines.extend(["", "  DOCTOR'S NOTES SUMMARY"])
            if report.parsed_notes and report.parsed_notes.vitals:
                lines.append(f"  Vitals: {report.parsed_notes.vitals}")
            if report.parsed_notes and report.parsed_notes.matched_keywords:
                kws = [k for k in report.parsed_notes.matched_keywords if not k.startswith("[suppress]")]
                if kws:
                    lines.append(f"  Key clinical terms: {', '.join(kws[:8])}")

        lines.extend(["", "  DISCLAIMER", f"  {report.disclaimer}", sep])
        return "\n".join(lines)

    def get_probability_table(
        self, report: DiagnosisReport
    ) -> List[Tuple[str, float, str]]:
        return [
            (disease, round(prob * 100, 2), _classify_severity(disease, prob))
            for disease, prob in sorted(
                report.raw_probabilities.items(), key=lambda x: x[1], reverse=True
            )
        ]


# ── Disclaimer ─────────────────────────────────────────────────────────────────
_DISCLAIMER = (
    "This AI-generated analysis is intended as a decision-support tool only. "
    "It does NOT replace clinical judgement, radiological review, or physician "
    "consultation. All findings must be verified by a qualified medical professional "
    "before clinical action is taken. The system is not certified for autonomous "
    "medical diagnosis."
)
