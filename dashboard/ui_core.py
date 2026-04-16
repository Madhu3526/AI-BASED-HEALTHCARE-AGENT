import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.auth import logout, render_login
from agent.diagnosis_agent import DiagnosisAgent
from agent.doctor_notes import parse_doctor_notes
from data.dataset import DISEASE_LABELS, get_transforms
from data.dicom_loader import load_xray, dicom_metadata
from explainability.gradcam import generate_multi_class_heatmaps
from models.densenet_model import build_densenet
from models.resnet_model import build_resnet
from models.uncertainty import mc_dropout_predict, flag_uncertain_predictions, build_uncertainty_summary
from patient.longitudinal import PatientTracker
from monitoring.drift_detection import DriftDetector
from rag.chatbot import MedicalChatbot
from rag.knowledge_base import MedicalKnowledgeBase


EXAMPLE_DOCTOR_NOTES = (
    "55M, ex-smoker (20 pack-years). Presenting with 3-week cough and haemoptysis.\n"
    "Vitals: HR: 92 bpm, SpO2: 95%, Temp: 37.2C, BP: 138/88.\n"
    "Hx: COPD, hypertension. No fever. No chest pain.\n"
    "Medications: Salbutamol inhaler, amlodipine."
)


def apply_page_style(accent: str, secondary: str) -> None:
    st.markdown(
        f"""
<style>
    .stApp {{ background-color: #0e1117; }}
    .main-header {{
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, {accent}, {secondary});
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }}
    .sub-header {{ color: #90a4ae; font-size: 0.98rem; margin-bottom: 1rem; }}
    .welcome-card {{
        background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 4px solid {accent};
        padding: 0.9rem 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: #d7e3ea;
    }}
    .urgent-banner {{
        background: #b71c1c; color: white; padding: 0.7rem 1rem;
        border-radius: 8px; font-weight: 600; margin: 0.5rem 0;
    }}
    .routine-banner {{
        background: #1b5e20; color: white; padding: 0.7rem 1rem;
        border-radius: 8px; font-weight: 600; margin: 0.5rem 0;
    }}
    .warning-banner {{
        background: #e65100; color: white; padding: 0.7rem 1rem;
        border-radius: 8px; font-weight: 600; margin: 0.5rem 0;
    }}
    .chat-user {{
        background: {accent}; color: white;
        padding: 0.6rem 1rem; border-radius: 12px 12px 2px 12px;
        margin: 0.3rem 0; max-width: 80%; margin-left: auto;
    }}
    .chat-assistant {{
        background: #1e2329; color: #cfd8dc;
        padding: 0.6rem 1rem; border-radius: 12px 12px 12px 2px;
        margin: 0.3rem 0; max-width: 85%;
        border-left: 3px solid {accent};
    }}
    .note-adjusted {{ color: #ffa726; font-style: italic; }}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading AI model...")
def load_model(model_type: str = "resnet", checkpoint_path: Optional[str] = None):
    if model_type == "densenet":
        model = build_densenet(num_classes=14, pretrained=True)
    elif model_type == "vit":
        from models.vit_model import build_vit

        model = build_vit(num_classes=14, pretrained=True, unfreeze_last_n=4)
    else:
        model = build_resnet(num_classes=14, pretrained=True)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        current_state = model.state_dict()
        compatible_state = {}
        skipped_keys = []

        for key, value in state.items():
            if key not in current_state:
                skipped_keys.append(key)
                continue
            if current_state[key].shape != value.shape:
                skipped_keys.append(key)
                continue
            compatible_state[key] = value

        current_state.update(compatible_state)
        model.load_state_dict(current_state, strict=False)

        if skipped_keys:
            st.warning(
                "Checkpoint loaded with partial compatibility. "
                f"Skipped {len(skipped_keys)} layer(s) because they do not match the selected model architecture."
            )

    model.eval()
    return model


class _EnsembleWrapper(torch.nn.Module):
    """
    Thin nn.Module shim that wraps EnsembleModel so it is compatible with
    DiagnosisAgent (which calls torch.sigmoid(model(x))).

    forward() returns logit-space values so that sigmoid(logit(p)) = p.
    Per-model individual predictions are stored in _last_individual for
    display in the Predictions tab.
    """

    def __init__(self, ensemble):
        super().__init__()
        self._ensemble = ensemble
        # DenseNet is the primary component — used for GradCAM & MC Dropout
        self._primary_model = ensemble.models[1] if len(ensemble.models) > 1 else ensemble.models[0]
        self._last_individual = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs_dict, self._last_individual = self._ensemble.predict(x)
        probs = torch.tensor(
            [probs_dict[d] for d in __import__("data.dataset", fromlist=["DISEASE_LABELS"]).DISEASE_LABELS],
            dtype=torch.float32,
        )
        # Invert sigmoid so DiagnosisAgent's sigmoid(model(x)) gives back the same probs
        logits = torch.logit(probs.clamp(1e-6, 1.0 - 1e-6))
        return logits.unsqueeze(0)

    def eval(self):
        for m in self._ensemble.models:
            m.eval()
        return super().eval()

    def train(self, mode: bool = True):
        # Ensemble is inference-only; never switch component models to train mode
        return self


@st.cache_resource(show_spinner="Loading Ensemble (ResNet-50 + DenseNet-121 + ViT-Base)...")
def load_ensemble_model(checkpoint_path: Optional[str] = None):
    from models.ensemble import EnsembleModel
    from models.vit_model import build_vit

    component_models = [
        ("ResNet-50",    build_resnet(num_classes=14, pretrained=True)),
        ("DenseNet-121", build_densenet(num_classes=14, pretrained=True)),
        ("ViT-Base/16",  build_vit(num_classes=14, pretrained=True, unfreeze_last_n=4)),
    ]

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        for _name, m in component_models:
            cur = m.state_dict()
            compatible = {k: v for k, v in state.items() if k in cur and cur[k].shape == v.shape}
            cur.update(compatible)
            m.load_state_dict(cur, strict=False)

    names, models = zip(*component_models)
    ensemble = EnsembleModel(models=list(models), names=list(names))
    return _EnsembleWrapper(ensemble)


@st.cache_resource(show_spinner="Building knowledge base...")
def load_knowledge_base():
    return MedicalKnowledgeBase(use_faiss=True)


def _get_chatbot(state_key: str, kb: MedicalKnowledgeBase) -> MedicalChatbot:
    if state_key not in st.session_state:
        st.session_state[state_key] = MedicalChatbot(kb)
    return st.session_state[state_key]


def _ensure_state(prefix: str) -> None:
    defaults = {
        f"{prefix}_doctor_notes_input": "",
        f"{prefix}_current_report": None,
        f"{prefix}_chat_input": "",
        f"{prefix}_standalone_chat": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _load_example_notes(prefix: str) -> None:
    st.session_state[f"{prefix}_doctor_notes_input"] = EXAMPLE_DOCTOR_NOTES


@st.cache_resource(show_spinner=False)
def _get_tracker() -> PatientTracker:
    return PatientTracker()


@st.cache_resource(show_spinner=False)
def _get_drift_detector() -> DriftDetector:
    return DriftDetector()


def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    transform = get_transforms(split="test", img_size=224)
    return transform(pil_img.convert("RGB")).unsqueeze(0)


def make_agent(model, kb, threshold: float, model_type: str, note_weight: float):
    model_label = {
        "resnet":   "ResNet-50",
        "densenet": "DenseNet-121 + SE",
        "vit":      "ViT-Base/16",
        "ensemble": "Ensemble (ResNet-50 + DenseNet-121 + ViT)",
    }.get(model_type, model_type)
    return DiagnosisAgent(
        model=model,
        knowledge_base=kb,
        threshold=threshold,
        model_name=f"{model_label} (Federated)",
        note_weight=note_weight,
    )


def prob_colour(p: float) -> str:
    if p >= 0.7:
        return "#f44336"
    if p >= 0.5:
        return "#ff7043"
    if p >= 0.35:
        return "#ffa726"
    return "#66bb6a"


def render_sidebar(config) -> dict:
    st.sidebar.markdown(f"## {config.title}")
    st.sidebar.caption(config.subtitle)
    _arch_options = ["resnet", "densenet", "vit", "ensemble"]
    _default_idx  = _arch_options.index(config.model_type) if config.model_type in _arch_options else 1
    model_type = st.sidebar.selectbox(
        "Model architecture",
        options=_arch_options,
        index=_default_idx,
        help=(
            "resnet / densenet / vit — single federated model.\n"
            "ensemble — averages ResNet-50 + DenseNet-121 + ViT predictions for higher accuracy."
        ),
    )
    checkpoint = st.sidebar.text_input(
        "Model checkpoint path",
        value=config.checkpoint_path,
        help="Use the hospital default or override with a different checkpoint.",
    )
    threshold = st.sidebar.slider("Prediction threshold", 0.1, 0.9, config.threshold, 0.05)
    top_k = st.sidebar.slider("Max diseases to show", 1, 14, config.top_k)
    show_all = st.sidebar.checkbox("Show all 14 disease scores", True)
    n_heatmaps = st.sidebar.slider("GradCAM heatmaps", 1, 5, config.heatmaps)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Hospital Node**  \n"
        f"`{config.hospital_id}`  \n"
        f"Default model: `{config.model_type}`  \n"
        f"Note weight: `{config.note_weight:.2f}`  \n"
        f"Dataset seed: `{config.dataset_seed}`"
    )
    if st.sidebar.button("Logout", key=f"{config.hospital_id}_logout"):
        logout(config.hospital_id)
        st.rerun()
    return {
        "model_type": model_type,
        "checkpoint": checkpoint or None,
        "threshold": threshold,
        "top_k": top_k,
        "show_all": show_all,
        "n_heatmaps": n_heatmaps,
    }


def render_hospital_page(config) -> None:
    prefix = config.hospital_id.lower()
    _ensure_state(prefix)
    apply_page_style(config.accent, config.secondary)

    if not render_login(config.hospital_id, config.title, config.accent):
        return

    st.markdown(f'<div class="main-header">{config.title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{config.subtitle}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="welcome-card">{config.welcome}</div>', unsafe_allow_html=True)

    settings = render_sidebar(config)
    is_ensemble = settings["model_type"] == "ensemble"
    if is_ensemble:
        model = load_ensemble_model(settings["checkpoint"])
        primary_model = model._primary_model   # DenseNet — used for GradCAM & MC Dropout
    else:
        model = load_model(settings["model_type"], settings["checkpoint"])
        primary_model = model
    kb = load_knowledge_base()
    agent = make_agent(model, kb, settings["threshold"], settings["model_type"], config.note_weight)
    agent.top_k = settings["top_k"]
    chatbot = _get_chatbot(f"{prefix}_chatbot", kb)

    st.markdown("### Dataset Scope")
    d1, d2, d3 = st.columns(3)
    d1.metric("Dataset slice", config.dataset_partition)
    d2.metric("Subset fraction", f"{config.subset_fraction:.2f}")
    d3.metric("Partition seed", str(config.dataset_seed))
    st.caption(
        "This hospital page represents a local partition of the shared NIH ChestXray14 dataset. "
        "In the federated trainer, partitions are simulated by giving each hospital its own seed and subset fraction."
    )

    col_id, col_upload = st.columns([1, 3])
    with col_id:
        patient_id = st.text_input("Patient ID", value=f"{config.hospital_id}-PT-001", max_chars=32)
    with col_upload:
        uploaded = st.file_uploader(
            "Upload Chest X-ray (PNG / JPG / DICOM)",
            type=["png", "jpg", "jpeg", "dcm"],
            label_visibility="collapsed",
            key=f"{prefix}_uploader",
        )

    with st.expander("Doctor's Clinical Notes", expanded=False):
        st.caption("Add local clinical context to complement the image model.")
        col_notes, col_example = st.columns([3, 1])
        with col_notes:
            doctor_notes = st.text_area(
                "Clinical notes",
                key=f"{prefix}_doctor_notes_input",
                placeholder=f"Example:\n{EXAMPLE_DOCTOR_NOTES}",
                height=130,
                label_visibility="collapsed",
            )
        with col_example:
            st.button(
                "Load example notes",
                on_click=_load_example_notes,
                args=(prefix,),
                key=f"{prefix}_load_notes",
            )
            if doctor_notes.strip():
                parsed = parse_doctor_notes(doctor_notes)
                if parsed.vitals:
                    st.caption(f"Parsed vitals: {parsed.vitals}")
                if parsed.matched_keywords:
                    kws = [k for k in parsed.matched_keywords if not k.startswith("[")]
                    if kws:
                        st.caption(f"Clinical keywords: {', '.join(kws[:6])}")

    if uploaded is None:
        st.info("Upload a chest X-ray image to begin analysis for this hospital.")
        _render_chatbot_standalone(chatbot, prefix)
        _render_patient_history(patient_id, prefix)
        return

    # ── Load image — DICOM or raster ──────────────────────────────────────
    is_dicom = uploaded.name.lower().endswith(".dcm")
    if is_dicom:
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        pil_img = load_xray(tmp_path)
        dcm_meta = dicom_metadata(tmp_path)
        os.unlink(tmp_path)
    else:
        pil_img  = Image.open(uploaded).convert("RGB")
        dcm_meta = {}

    image_tensor = preprocess_image(pil_img)

    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.markdown("**Original X-ray**")
        st.image(pil_img, use_container_width=True)
        if dcm_meta:
            with st.expander("DICOM Metadata"):
                for k, v in dcm_meta.items():
                    st.caption(f"**{k}:** {v}")

    with st.spinner("Running AI analysis..."):
        t0 = time.time()
        report = agent.diagnose(image_tensor, patient_id=patient_id, doctor_notes=doctor_notes)
        elapsed = time.time() - t0
        st.session_state[f"{prefix}_current_report"] = report
        # Capture per-model predictions when ensemble is active
        individual_probs = list(getattr(model, "_last_individual", [])) if is_ensemble else []

    # ── Save to patient history ───────────────────────────────────────────
    tracker = _get_tracker()
    tracker.save_visit(report=report, patient_id=patient_id, hospital_id=config.hospital_id)

    # ── Update drift detector buffer ──────────────────────────────────────
    drift_detector = _get_drift_detector()
    drift_detector.update_buffer(report.raw_probabilities, buffer_key=config.hospital_id)

    chatbot.set_report_context(report.raw_probabilities, threshold=settings["threshold"])

    with col_info:
        _render_urgency_banner(report.urgency_level)
        for flag in report.vitals_flags:
            icon = "CRITICAL" if "CRITICAL" in flag else "WARNING"
            st.warning(f"{icon}: {flag}")
        st.markdown("**Primary Impression**")
        st.info(report.primary_impression)
        st.caption(
            f"Analysis: {elapsed:.2f}s | Model: {report.model_used}"
            + (f" | Notes: {len(report.note_adjustments)} adjustments" if report.note_adjustments else "")
        )

    tab_pred, tab_uncertainty, tab_gradcam, tab_report, tab_history, tab_chat, tab_raw = st.tabs(
        ["Predictions", "Uncertainty", "GradCAM", "Clinical Report", "Patient History", "AI Chatbot", "Raw Data"]
    )
    with tab_pred:
        _render_predictions(report, individual_probs=individual_probs)
    with tab_uncertainty:
        _render_uncertainty(primary_model, image_tensor, report, settings["threshold"], prefix,
                            is_ensemble=is_ensemble)
    with tab_gradcam:
        _render_gradcam(primary_model, image_tensor, settings["n_heatmaps"],
                        is_ensemble=is_ensemble)
    with tab_report:
        _render_clinical_report(report, prefix)
    with tab_history:
        _render_patient_history(patient_id, prefix)
    with tab_chat:
        _render_chatbot(chatbot, prefix)
    with tab_raw:
        _render_raw_data(report, settings["threshold"])


def _render_urgency_banner(urgency: str) -> None:
    banners = {
        "emergency": '<div class="urgent-banner">EMERGENCY - Immediate clinical evaluation required</div>',
        "urgent": '<div class="warning-banner">URGENT - Prompt physician review recommended</div>',
        "routine": '<div class="routine-banner">ROUTINE - Standard clinical follow-up suggested</div>',
    }
    st.markdown(banners.get(urgency, banners["routine"]), unsafe_allow_html=True)


def _render_predictions(report, individual_probs: list = None) -> None:
    import plotly.graph_objects as go

    st.markdown("### Disease Probability Scores")
    if report.note_adjustments:
        with st.expander(f"{len(report.note_adjustments)} probabilities adjusted from notes"):
            for adj in report.note_adjustments:
                st.markdown(f"- {adj}")

    probs = report.raw_probabilities
    labels = list(probs.keys())
    values = [probs[l] * 100 for l in labels]
    colours = [prob_colour(probs[l]) for l in labels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colours,
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Predicted Disease Probabilities",
        xaxis_title="Probability (%)",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,35,41,1)",
        font=dict(color="#cfd8dc"),
        xaxis=dict(range=[0, 115]),
    )
    st.plotly_chart(fig, use_container_width=True)

    if report.predicted_diseases:
        st.markdown("### Detected Conditions")
        for disease in report.predicted_diseases:
            c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
            c1.markdown(f"**{disease.name}**")
            c2.markdown(
                f'<span style="color:{prob_colour(disease.probability)};font-weight:600;">'
                f"{disease.probability*100:.1f}%</span>",
                unsafe_allow_html=True,
            )
            if disease.note_adjusted:
                c3.markdown(
                    f'<span class="note-adjusted">model: {disease.model_prob*100:.1f}%</span>',
                    unsafe_allow_html=True,
                )
            c4.caption(f"Severity: {disease.severity}")
    else:
        st.success("No significant pathology detected above confidence threshold.")

    # ── Ensemble: per-model comparison ────────────────────────────────────────
    if individual_probs:
        import plotly.graph_objects as go
        from data.dataset import DISEASE_LABELS as _DL

        model_names = [p.get("_model_name", f"Model {i+1}") if isinstance(p, dict) and "_model_name" in p
                       else f"Model {i+1}" for i, p in enumerate(individual_probs)]
        # individual_probs is a list of {disease: prob} dicts from EnsembleModel.predict
        # EnsembleModel returns list without model names, so use fixed names
        model_names = ["ResNet-50", "DenseNet-121", "ViT-Base/16"][:len(individual_probs)]

        st.divider()
        st.markdown("### Ensemble — Per-Model Breakdown")
        st.caption("Bars show each component model's raw probability. Disagreement between models flags cases for review.")

        # ── Agreement / disagreement flags ────────────────────────────────
        threshold = 0.45
        disagreements = []
        for disease in _DL:
            decisions = [p.get(disease, 0.0) >= threshold for p in individual_probs]
            if 0 < sum(decisions) < len(decisions):
                votes_str = " | ".join(
                    f"{name}: {'✓' if dec else '✗'} ({p.get(disease,0)*100:.0f}%)"
                    for name, dec, p in zip(model_names, decisions, individual_probs)
                )
                disagreements.append(f"**{disease}** — models disagree: {votes_str}")

        if disagreements:
            with st.expander(f"Model disagreement on {len(disagreements)} disease(s) — flag for review", expanded=True):
                for d in disagreements:
                    st.warning(d)
        else:
            st.success("All component models agree on binary decisions for every disease class.")

        # ── Grouped bar chart — top 8 by ensemble prob ────────────────────
        sorted_diseases = sorted(_DL, key=lambda d: report.raw_probabilities.get(d, 0), reverse=True)[:8]
        colours = ["#4fc3f7", "#81d4fa", "#b3e5fc"]
        fig_ens = go.Figure()
        for model_name, probs_dict, colour in zip(model_names, individual_probs, colours):
            fig_ens.add_trace(go.Bar(
                name=model_name,
                x=sorted_diseases,
                y=[probs_dict.get(d, 0) * 100 for d in sorted_diseases],
                marker_color=colour,
                text=[f"{probs_dict.get(d,0)*100:.1f}%" for d in sorted_diseases],
                textposition="outside",
            ))
        fig_ens.update_layout(
            barmode="group",
            title="Top-8 Diseases — Per-Model Probabilities",
            yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 115], gridcolor="#2a3039"),
            xaxis=dict(gridcolor="#2a3039"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,35,41,1)",
            font=dict(color="#cfd8dc"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ens, use_container_width=True)


def _render_gradcam(model, image_tensor, n_heatmaps: int, is_ensemble: bool = False) -> None:
    st.markdown("### GradCAM")
    if is_ensemble:
        st.caption("Ensemble mode: GradCAM runs on the DenseNet-121 component (the primary backbone).")
    with st.spinner("Generating GradCAM..."):
        try:
            results = generate_multi_class_heatmaps(
                model=model,
                image_tensor=image_tensor,
                disease_labels=DISEASE_LABELS,
                top_k=n_heatmaps,
            )
        except Exception as exc:
            st.error(f"GradCAM error: {exc}")
            return

    cols = st.columns(min(n_heatmaps, 3))
    for i, (disease, prob, _, overlay) in enumerate(results):
        with cols[i % len(cols)]:
            st.image(overlay, caption=f"{disease}: {prob*100:.1f}%", use_container_width=True)


def _render_clinical_report(report, prefix: str = "") -> None:
    st.markdown("### Recommended Actions")
    for i, action in enumerate(report.recommended_actions, 1):
        st.markdown(f"{i}. {action}")

    if report.doctor_notes.strip() and report.parsed_notes:
        st.markdown("### Clinical Notes Summary")
        p = report.parsed_notes
        col1, col2 = st.columns(2)
        with col1:
            if p.vitals:
                st.markdown("**Parsed Vitals**")
                for k, v in p.vitals.items():
                    st.markdown(f"- {k}: {v}")
            if p.symptoms:
                st.markdown("**Symptoms (from notes)**")
                for s in p.symptoms[:5]:
                    st.markdown(f"- {s}")
        with col2:
            if p.history:
                st.markdown("**History (from notes)**")
                for h in p.history[:5]:
                    st.markdown(f"- {h}")
            if p.medications:
                st.markdown("**Medications (from notes)**")
                for m in p.medications[:5]:
                    st.markdown(f"- {m}")

    if report.predicted_diseases:
        st.markdown("### Disease Details")
        for disease in report.predicted_diseases:
            adj_str = f" *(note-adjusted from {disease.model_prob*100:.1f}%)*" if disease.note_adjusted else ""
            with st.expander(f"{disease.name} ({disease.probability*100:.1f}%){adj_str}"):
                if disease.overview:
                    st.markdown(f"**Overview:** {disease.overview}")
                c1, c2 = st.columns(2)
                with c1:
                    if disease.symptoms:
                        st.markdown("**Symptoms**")
                        for s in disease.symptoms:
                            st.markdown(f"- {s}")
                    if disease.risk_factors:
                        st.markdown("**Risk Factors**")
                        for r in disease.risk_factors:
                            st.markdown(f"- {r}")
                with c2:
                    if disease.treatment:
                        st.markdown("**Treatment**")
                        for t in disease.treatment:
                            st.markdown(f"- {t}")
                    if disease.follow_up:
                        st.info(f"**Follow-up:** {disease.follow_up}")

    st.divider()
    st.caption(report.disclaimer)

    # ── PDF export ────────────────────────────────────────────────────────
    st.markdown("### Export")
    try:
        from reports.pdf_export import export_report_pdf
        pdf_bytes = export_report_pdf(report, hospital_name="Federated Medical AI")
        st.download_button(
            label     = "Download PDF Report",
            data      = pdf_bytes,
            file_name = f"report_{report.patient_id}_{report.model_used.replace(' ', '_')}.pdf",
            mime      = "application/pdf",
            key       = f"{prefix}_pdf_download",
        )
    except ImportError:
        st.caption("Install `reportlab` to enable PDF export: `pip install reportlab`")


def _submit_chat(chatbot: MedicalChatbot, message: str) -> None:
    chatbot.chat(message)
    st.rerun()


def _render_chatbot(chatbot: MedicalChatbot, prefix: str) -> None:
    st.markdown("### Medical AI Chatbot")
    suggestions = chatbot.get_suggested_questions()
    if suggestions:
        st.markdown("**Quick questions:**")
        btn_cols = st.columns(3)
        for i, q in enumerate(suggestions[:6]):
            if btn_cols[i % 3].button(q, key=f"{prefix}_sugg_{i}"):
                _submit_chat(chatbot, q)

    for msg in chatbot.history:
        css_class = "chat-user" if msg.role == "user" else "chat-assistant"
        content = msg.content.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)

    col_input, col_send, col_clear = st.columns([5, 1, 1])
    with col_input:
        user_input = st.text_input(
            "Your question",
            key=f"{prefix}_chat_input",
            placeholder="Ask about findings, treatment, or follow-up.",
            label_visibility="collapsed",
        )
    with col_send:
        send = st.button("Send", type="primary", use_container_width=True, key=f"{prefix}_chat_send")
    with col_clear:
        if st.button("Clear", use_container_width=True, key=f"{prefix}_chat_clear"):
            chatbot.history = []
            st.rerun()

    if send and user_input.strip():
        _submit_chat(chatbot, user_input)


def _render_chatbot_standalone(chatbot: MedicalChatbot, prefix: str) -> None:
    st.divider()
    st.markdown("### Medical Knowledge Chatbot")
    col_input, col_send = st.columns([5, 1])
    with col_input:
        q = st.text_input(
            "Ask a medical question",
            key=f"{prefix}_standalone_chat",
            placeholder="e.g. What are the symptoms of Pneumonia?",
            label_visibility="collapsed",
        )
    with col_send:
        if st.button("Ask", key=f"{prefix}_standalone_send", type="primary"):
            if q.strip():
                chatbot.chat(q)

    for msg in chatbot.history[-6:]:
        css = "chat-user" if msg.role == "user" else "chat-assistant"
        st.markdown(f'<div class="{css}">{msg.content}</div>', unsafe_allow_html=True)


def _render_uncertainty(model, image_tensor, report, threshold: float, prefix: str,
                        is_ensemble: bool = False) -> None:
    st.markdown("### Uncertainty Quantification (Monte Carlo Dropout)")
    if is_ensemble:
        st.caption("Ensemble mode: MC Dropout runs on the DenseNet-121 component model.")
    st.caption(
        "Runs 30 stochastic forward passes with dropout active. "
        "High std deviation means the model is uncertain — flag for radiologist review."
    )
    with st.spinner("Running MC Dropout inference (30 passes)..."):
        mean_probs, std_probs = mc_dropout_predict(model, image_tensor, n_passes=30)

    uncertain_flags = flag_uncertain_predictions(mean_probs, std_probs, threshold=threshold)
    if uncertain_flags:
        st.warning("Uncertain predictions near the decision boundary:")
        for flag in uncertain_flags:
            st.markdown(f"- {flag}")
    else:
        st.success("No uncertain predictions near the decision boundary.")

    import pandas as pd
    rows = build_uncertainty_summary(mean_probs, std_probs)
    df   = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_patient_history(patient_id: str, prefix: str) -> None:
    import pandas as pd
    import plotly.express as px

    tracker = _get_tracker()
    history = tracker.get_history(patient_id)

    if not history:
        st.info(f"No prior visits found for patient **{patient_id}**.")
        return

    st.markdown(f"### Patient History — {patient_id}")
    st.caption(f"{len(history)} visit(s) on record")

    # ── Progression flags ─────────────────────────────────────────────────
    flags = tracker.progression_flags(patient_id)
    if flags:
        st.warning("Changes since last visit:")
        for f in flags:
            st.markdown(f"- {f}")

    # ── Visit summary table ────────────────────────────────────────────────
    summary_rows = [
        {
            "Visit":    i + 1,
            "Date":     r["timestamp"][:10],
            "Model":    r["model_used"],
            "Urgency":  r["urgency"].upper(),
            "Detected": ", ".join(d["name"] for d in r["detected_diseases"]) or "None",
        }
        for i, r in enumerate(history)
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── Disease trend chart ───────────────────────────────────────────────
    st.markdown("#### Disease Probability Trends")
    detected_diseases = {
        d["name"]
        for r in history
        for d in r["detected_diseases"]
    }
    if detected_diseases:
        selected = st.multiselect(
            "Select diseases to plot",
            options=sorted(detected_diseases),
            default=sorted(detected_diseases)[:3],
            key=f"{prefix}_trend_select",
        )
        if selected:
            trend_data = []
            for disease in selected:
                for point in tracker.disease_trend(patient_id, disease):
                    trend_data.append({
                        "Date":        point["timestamp"][:10],
                        "Probability": point["probability"] * 100,
                        "Disease":     disease,
                    })
            if trend_data:
                fig = px.line(
                    pd.DataFrame(trend_data),
                    x="Date", y="Probability", color="Disease",
                    markers=True,
                    title=f"Disease Probability Over Time — {patient_id}",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(30,35,41,1)",
                    font=dict(color="#cfd8dc"),
                    yaxis_title="Probability (%)",
                )
                st.plotly_chart(fig, use_container_width=True)


def _render_raw_data(report, threshold: float) -> None:
    import pandas as pd

    rows = [
        {
            "Disease": d,
            "AI Prob (%)": f"{p*100:.2f}",
            "Detected": "Yes" if p >= threshold else "No",
        }
        for d, p in sorted(report.raw_probabilities.items(), key=lambda x: x[1], reverse=True)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if report.doctor_notes.strip():
        st.markdown("### Doctor's Notes (submitted)")
        st.code(report.doctor_notes, language="text")
