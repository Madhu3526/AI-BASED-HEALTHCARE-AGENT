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
    dataset_partition: str
    dataset_seed: int
    subset_fraction: float


ROOT = Path(__file__).resolve().parent.parent


HOSPITAL_CONFIGS = {
    "Hospital_A": HospitalConfig(
        hospital_id="Hospital_A",
        title="KMCH",
        subtitle="Emergency and respiratory triage workspace",
        accent="#1e88e5",
        secondary="#90caf9",
        model_type="densenet",
        checkpoint_path=str(ROOT / "models" / "federated" / "global_model_final.pth"),
        threshold=0.45,
        top_k=5,
        heatmaps=3,
        note_weight=0.30,
        dataset_partition="NIH ChestXray14",
        dataset_seed=0,
        subset_fraction=1 / 3,
    ),
    "Hospital_B": HospitalConfig(
        hospital_id="Hospital_B",
        title="SUMITH",
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
        dataset_partition="Hospital B local subset from NIH ChestXray14",
        dataset_seed=100,
        subset_fraction=1 / 3,
    ),
    "Hospital_C": HospitalConfig(
        hospital_id="Hospital_C",
        title="GH",
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
        dataset_partition="Hospital C local subset from NIH ChestXray14",
        dataset_seed=200,
        subset_fraction=1 / 3,
    ),
}
