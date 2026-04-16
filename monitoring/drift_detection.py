"""
Data Drift Detection
Detects distribution shift in model prediction probabilities between
a reference cohort (e.g. training set) and a current inference window.

Uses the Kolmogorov-Smirnov (KS) two-sample test per disease class.
A significant KS p-value indicates the current input distribution has
drifted from the reference — the model may silently degrade.

Usage:
    detector = DriftDetector()

    # Build reference from training outputs:
    detector.set_reference(reference_probs_array)   # shape (N, 14)

    # After a batch of new predictions:
    report = detector.check(current_probs_array)    # shape (M, 14)
    detector.print_report(report)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.dataset import DISEASE_LABELS

DEFAULT_REF_PATH = Path(__file__).resolve().parent / "reference_distribution.json"


class DriftDetector:
    """
    Per-disease KS-test drift detector.

    Maintains a reference distribution of model output probabilities and
    tests incoming batches for statistical divergence.
    """

    def __init__(
        self,
        p_value_threshold: float = 0.05,
        ref_path: Optional[str]  = None,
    ):
        self.p_value_threshold = p_value_threshold
        self.ref_path          = Path(ref_path or DEFAULT_REF_PATH)
        self._reference: Optional[np.ndarray] = None   # shape (N, 14)

        if self.ref_path.exists():
            self._load_reference()

    # ── Reference management ───────────────────────────────────────────────────
    def set_reference(self, probs: np.ndarray) -> None:
        """
        Store a reference distribution of shape (N, 14) where N is the
        number of reference samples (typically from validation set outputs).
        """
        assert probs.ndim == 2 and probs.shape[1] == len(DISEASE_LABELS), \
            f"Expected shape (N, {len(DISEASE_LABELS)}), got {probs.shape}"
        self._reference = probs
        self._save_reference()

    def _save_reference(self) -> None:
        self.ref_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "n_samples": int(self._reference.shape[0]),
            "distributions": {
                label: self._reference[:, i].tolist()
                for i, label in enumerate(DISEASE_LABELS)
            },
        }
        with open(self.ref_path, "w") as f:
            json.dump(data, f)

    def _load_reference(self) -> None:
        with open(self.ref_path, "r") as f:
            data = json.load(f)
        distributions = data["distributions"]
        arrays = [distributions[label] for label in DISEASE_LABELS]
        self._reference = np.array(arrays).T   # (N, 14)

    # ── Drift check ────────────────────────────────────────────────────────────
    def check(
        self,
        current_probs: np.ndarray,
    ) -> Dict:
        """
        Run a KS test per disease class against the stored reference.

        Args:
            current_probs: shape (M, 14) — batch of recent predictions

        Returns:
            report dict with per-disease results and overall drift status
        """
        from scipy.stats import ks_2samp

        if self._reference is None:
            raise RuntimeError(
                "No reference distribution set. Call set_reference() first."
            )

        assert current_probs.ndim == 2 and current_probs.shape[1] == len(DISEASE_LABELS), \
            f"Expected shape (M, {len(DISEASE_LABELS)}), got {current_probs.shape}"

        results: Dict[str, Dict] = {}
        drifted_diseases: List[str] = []

        for i, disease in enumerate(DISEASE_LABELS):
            ref     = self._reference[:, i]
            curr    = current_probs[:, i]
            stat, p = ks_2samp(ref, curr)

            drifted = p < self.p_value_threshold
            if drifted:
                drifted_diseases.append(disease)

            results[disease] = {
                "ks_statistic": round(float(stat), 4),
                "p_value":      round(float(p), 4),
                "drifted":      drifted,
                "ref_mean":     round(float(ref.mean()), 4),
                "curr_mean":    round(float(curr.mean()), 4),
                "mean_delta":   round(float(curr.mean() - ref.mean()), 4),
            }

        return {
            "timestamp":        datetime.utcnow().isoformat() + "Z",
            "n_reference":      int(self._reference.shape[0]),
            "n_current":        int(current_probs.shape[0]),
            "p_value_threshold": self.p_value_threshold,
            "overall_drift":    len(drifted_diseases) > 0,
            "n_drifted":        len(drifted_diseases),
            "drifted_diseases": drifted_diseases,
            "per_disease":      results,
        }

    # ── Single-image incremental buffer ───────────────────────────────────────
    def update_buffer(
        self,
        probs_dict: Dict[str, float],
        buffer_key: str = "default",
        max_buffer: int = 500,
    ) -> None:
        """
        Accumulate single-image prediction dicts into an in-memory buffer.
        Useful when you want drift detection over a rolling window of patients
        without batching manually.

        Call check_buffer() once buffer is large enough.
        """
        if not hasattr(self, "_buffers"):
            self._buffers: Dict[str, List[List[float]]] = {}
        buf = self._buffers.setdefault(buffer_key, [])
        row = [probs_dict.get(d, 0.0) for d in DISEASE_LABELS]
        buf.append(row)
        if len(buf) > max_buffer:
            buf.pop(0)   # sliding window

    def check_buffer(self, buffer_key: str = "default") -> Optional[Dict]:
        """Run drift check on the accumulated buffer. Returns None if empty."""
        if not hasattr(self, "_buffers") or buffer_key not in self._buffers:
            return None
        buf = self._buffers[buffer_key]
        if not buf:
            return None
        return self.check(np.array(buf))

    # ── Reporting ──────────────────────────────────────────────────────────────
    @staticmethod
    def print_report(report: Dict) -> None:
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  Data Drift Report  —  {report['timestamp'][:19]}")
        print(sep)
        status = "DRIFT DETECTED" if report["overall_drift"] else "No significant drift"
        print(f"  Status: {status}  ({report['n_drifted']} / {len(DISEASE_LABELS)} diseases)")
        print(f"  Reference samples: {report['n_reference']}  |  Current samples: {report['n_current']}")
        print()
        if report["drifted_diseases"]:
            print("  Drifted diseases:")
            for disease in report["drifted_diseases"]:
                d = report["per_disease"][disease]
                print(
                    f"    {disease:<22} KS={d['ks_statistic']:.3f}  "
                    f"p={d['p_value']:.4f}  "
                    f"mean: {d['ref_mean']:.3f} → {d['curr_mean']:.3f}  "
                    f"(Δ{d['mean_delta']:+.3f})"
                )
        print(sep)

    @staticmethod
    def summary_flags(report: Dict) -> List[str]:
        """Return a list of alert strings suitable for dashboard display."""
        flags: List[str] = []
        if not report["overall_drift"]:
            return flags
        for disease in report["drifted_diseases"]:
            d = report["per_disease"][disease]
            flags.append(
                f"{disease}: mean shifted {d['mean_delta']:+.3f} "
                f"(KS p={d['p_value']:.3f})"
            )
        return flags
