from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HospitalConfig:
    hospital_id: str
    title: str
    subtitle: str
    accent: str
    secondary: str
    model_type: str
    checkpoint_path: str
    threshold: float
    top_k: int
    heatmaps: int
    note_weight: float
    welcome: str


ROOT = Path(__file__).resolve().parent.parent


HOSPITAL_CONFIGS = {
    "Hospital_A": HospitalConfig(
        hospital_id="Hospital_A",
        title="KMHC",
        subtitle="Emergency and respiratory triage workspace",
        accent="#1e88e5",
        secondary="#90caf9",
        model_type="densenet",
        checkpoint_path=str(ROOT / "models" / "federated" / "global_model_final.pth"),
        threshold=0.45,
        top_k=5,
        heatmaps=3,
        note_weight=0.30,
        welcome="Use this view for rapid review of urgent respiratory presentations.",
    ),
    "Hospital_B": HospitalConfig(
        hospital_id="Hospital_B",
        title="City Heart and Lung Centre",
        subtitle="Pulmonology-focused review for follow-up and specialist interpretation",
        accent="#00897b",
        secondary="#80cbc4",
        model_type="densenet",
        checkpoint_path=str(ROOT / "models" / "federated" / "global_model_final.pth"),
        threshold=0.40,
        top_k=6,
        heatmaps=2,
        note_weight=0.25,
        welcome="This interface is tuned for specialist review and lower-threshold follow-up.",
    ),
    "Hospital_C": HospitalConfig(
        hospital_id="Hospital_C",
        title="Starlight Teaching Hospital",
        subtitle="Teaching and explainability workspace with expanded model output review",
        accent="#fb8c00",
        secondary="#ffcc80",
        model_type="densenet",
        checkpoint_path=str(ROOT / "models" / "federated" / "global_model_final.pth"),
        threshold=0.45,
        top_k=5,
        heatmaps=4,
        note_weight=0.20,
        welcome="Use this page for case review, GradCAM inspection, and teaching rounds.",
    ),
}
