"""
Patient Longitudinal Tracking
Persists per-patient diagnosis history to a JSON store so the system
can show disease progression across multiple X-ray visits.

Each record stores the timestamp, model predictions, urgency, and
doctor notes summary for one visit.

Usage:
    tracker = PatientTracker()
    tracker.save_visit(patient_id="PT-001", report=report)
    history = tracker.get_history("PT-001")
    trend   = tracker.disease_trend("PT-001", "Pneumonia")
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from data.dataset import DISEASE_LABELS

# Default storage path — can be overridden via PatientTracker(store_dir=...)
DEFAULT_STORE_DIR = Path(__file__).resolve().parent.parent / "patient" / "records"


class PatientTracker:
    """
    File-backed patient record store.

    Each patient gets a single JSON file: records/<patient_id>.json
    containing an ordered list of visit records.
    """

    def __init__(self, store_dir: Optional[str] = None):
        self.store_dir = Path(store_dir or DEFAULT_STORE_DIR)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ── Storage helpers ────────────────────────────────────────────────────────
    def _path(self, patient_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in patient_id)
        return self.store_dir / f"{safe_id}.json"

    def _load(self, patient_id: str) -> List[Dict]:
        p = self._path(patient_id)
        if not p.exists():
            return []
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, patient_id: str, records: List[Dict]) -> None:
        with open(self._path(patient_id), "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    # ── Public API ─────────────────────────────────────────────────────────────
    def save_visit(
        self,
        patient_id: str,
        report,                          # DiagnosisReport instance
        hospital_id: Optional[str] = None,
        notes_summary: Optional[str] = None,
    ) -> Dict:
        """
        Persist a diagnosis report as a patient visit record.

        Returns the saved record dict.
        """
        record = {
            "timestamp":   datetime.utcnow().isoformat() + "Z",
            "hospital_id": hospital_id or "unknown",
            "model_used":  report.model_used,
            "urgency":     report.urgency_level,
            "probabilities": {
                d: round(p, 4)
                for d, p in report.raw_probabilities.items()
            },
            "detected_diseases": [
                {
                    "name":        dr.name,
                    "probability": round(dr.probability, 4),
                    "severity":    dr.severity,
                }
                for dr in report.predicted_diseases
            ],
            "notes_summary": notes_summary or (
                report.doctor_notes[:200] if report.doctor_notes.strip() else ""
            ),
        }

        records = self._load(patient_id)
        records.append(record)
        self._save(patient_id, records)
        return record

    def get_history(self, patient_id: str) -> List[Dict]:
        """Return all visit records for a patient, oldest first."""
        return self._load(patient_id)

    def get_latest(self, patient_id: str) -> Optional[Dict]:
        """Return the most recent visit record, or None if no history."""
        records = self._load(patient_id)
        return records[-1] if records else None

    def disease_trend(self, patient_id: str, disease: str) -> List[Dict]:
        """
        Return timestamped probability series for a single disease.

        Example return:
            [
              {"timestamp": "2026-01-10T...", "probability": 0.41},
              {"timestamp": "2026-02-15T...", "probability": 0.58},
            ]
        """
        records = self._load(patient_id)
        return [
            {
                "timestamp":   r["timestamp"],
                "probability": r["probabilities"].get(disease, 0.0),
                "urgency":     r["urgency"],
            }
            for r in records
        ]

    def progression_flags(self, patient_id: str, delta_threshold: float = 0.10) -> List[str]:
        """
        Compare the two most recent visits and flag diseases where probability
        changed by more than delta_threshold since last visit.
        """
        records = self._load(patient_id)
        if len(records) < 2:
            return []

        prev = records[-2]["probabilities"]
        curr = records[-1]["probabilities"]
        flags: List[str] = []

        for disease in DISEASE_LABELS:
            p_prev = prev.get(disease, 0.0)
            p_curr = curr.get(disease, 0.0)
            delta  = p_curr - p_prev
            if abs(delta) >= delta_threshold:
                direction = "increased" if delta > 0 else "decreased"
                flags.append(
                    f"{disease}: {direction} {abs(delta)*100:.1f}pp "
                    f"({p_prev*100:.1f}% → {p_curr*100:.1f}%)"
                )
        return flags

    def list_patients(self) -> List[str]:
        """Return all patient IDs that have at least one record."""
        return [p.stem for p in self.store_dir.glob("*.json")]

    def delete_patient(self, patient_id: str) -> bool:
        """Remove all records for a patient. Returns True if file existed."""
        p = self._path(patient_id)
        if p.exists():
            os.remove(p)
            return True
        return False
