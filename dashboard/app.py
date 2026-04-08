"""
Doctor Dashboard — Streamlit Web Application
Provides an interactive interface for uploading chest X-rays,
entering clinical notes, viewing AI predictions, GradCAM heatmaps,
clinical recommendations, and an interactive RAG chatbot.
"""

import io
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch
from PIL import Image

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset import DISEASE_LABELS, get_transforms
from models.resnet_model import build_resnet
from explainability.gradcam import generate_gradcam_heatmap, generate_multi_class_heatmaps
from rag.knowledge_base import MedicalKnowledgeBase
from rag.chatbot import MedicalChatbot
from agent.diagnosis_agent import DiagnosisAgent
from agent.doctor_notes import parse_doctor_notes

# ── Streamlit page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Federated Medical AI — Chest X-ray Diagnosis",
    page_icon   = "🫁",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #4fc3f7, #81d4fa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #78909c; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .urgent-banner {
        background: #b71c1c; color: white; padding: 0.7rem 1rem;
        border-radius: 8px; font-weight: 600; margin: 0.5rem 0;
    }
    .routine-banner {
        background: #1b5e20; color: white; padding: 0.7rem 1rem;
        border-radius: 8px; font-weight: 600; margin: 0.5rem 0;
    }
    .warning-banner {
        background: #e65100; color: white; padding: 0.7rem 1rem;
        border-radius: 8px; font-weight: 600; margin: 0.5rem 0;
    }
    /* Chat bubbles */
    .chat-user {
        background: #1565c0; color: white;
        padding: 0.6rem 1rem; border-radius: 12px 12px 2px 12px;
        margin: 0.3rem 0; max-width: 80%; margin-left: auto;
    }
    .chat-assistant {
        background: #1e2329; color: #cfd8dc;
        padding: 0.6rem 1rem; border-radius: 12px 12px 12px 2px;
        margin: 0.3rem 0; max-width: 85%;
        border-left: 3px solid #4fc3f7;
    }
    .note-adjusted { color: #ffa726; font-style: italic; }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loaders ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_model(checkpoint_path: Optional[str] = None):
    model = build_resnet(num_classes=14, pretrained=True)
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt  = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        st.sidebar.success(f"Loaded: {Path(checkpoint_path).name}")
    model.eval()
    return model

@st.cache_resource(show_spinner="Building knowledge base...")
def load_knowledge_base():
    return MedicalKnowledgeBase(use_faiss=True)

def make_agent(model, kb, threshold):
    return DiagnosisAgent(
        model          = model,
        knowledge_base = kb,
        threshold      = threshold,
        model_name     = "ResNet-50 (Federated)",
        note_weight    = 0.3,
    )


# ── Session state helpers ──────────────────────────────────────────────────────
def _init_session():
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = None
    if "chat_initialized" not in st.session_state:
        st.session_state["chat_initialized"] = False
    if "current_report" not in st.session_state:
        st.session_state["current_report"] = None


def _get_chatbot(kb: MedicalKnowledgeBase) -> MedicalChatbot:
    if st.session_state["chatbot"] is None:
        st.session_state["chatbot"] = MedicalChatbot(kb)
    return st.session_state["chatbot"]


# ── Image preprocessing ────────────────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> torch.Tensor:
    transform = get_transforms(split="test", img_size=224)
    return transform(pil_img.convert("RGB")).unsqueeze(0)


# ── Colour helpers ─────────────────────────────────────────────────────────────
def severity_colour(s: str) -> str:
    return {"critical": "#f44336", "high": "#ff7043",
            "moderate": "#ffa726", "low": "#66bb6a"}.get(s, "#78909c")

def prob_colour(p: float) -> str:
    if p >= 0.7:  return "#f44336"
    if p >= 0.5:  return "#ff7043"
    if p >= 0.35: return "#ffa726"
    return "#66bb6a"


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("## Settings")
    checkpoint = st.sidebar.text_input(
        "Model checkpoint path",
        placeholder="models/federated/global_final.pth",
        help="Leave empty to use ImageNet pretrained weights.",
    )
    threshold  = st.sidebar.slider("Prediction threshold", 0.1, 0.9, 0.45, 0.05)
    top_k      = st.sidebar.slider("Max diseases to show", 1, 14, 5)
    show_all   = st.sidebar.checkbox("Show all 14 disease scores", True)
    n_heatmaps = st.sidebar.slider("GradCAM heatmaps", 1, 5, 3)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**About**  \n"
        "Federated Medical AI — NIH ChestX-ray14  \n"
        "ResNet-50 · FedAvg · Differential Privacy  \n"
        "GradCAM · RAG · Chatbot"
    )
    return dict(
        checkpoint = checkpoint or None,
        threshold  = threshold,
        top_k      = top_k,
        show_all   = show_all,
        n_heatmaps = n_heatmaps,
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    _init_session()

    st.markdown('<div class="main-header">🫁 Federated Medical AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Chest X-ray Multi-disease Detection · Federated Learning · '
        'Differential Privacy · Explainable AI · RAG Chatbot'
        '</div>',
        unsafe_allow_html=True,
    )

    settings = render_sidebar()
    model    = load_model(settings["checkpoint"])
    kb       = load_knowledge_base()
    agent    = make_agent(model, kb, settings["threshold"])
    agent.top_k = settings["top_k"]
    chatbot  = _get_chatbot(kb)

    # ── Patient info row ────────────────────────────────────────────────────
    col_id, col_upload = st.columns([1, 3])
    with col_id:
        patient_id = st.text_input("Patient ID", value="PT-001", max_chars=20)

    with col_upload:
        uploaded = st.file_uploader(
            "Upload Chest X-ray (PNG / JPG)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )

    # ── Doctor's Notes ──────────────────────────────────────────────────────
    with st.expander("📝 Doctor's Clinical Notes  (optional — improves AI accuracy)", expanded=False):
        st.caption(
            "Enter patient symptoms, history, vitals, medications. "
            "The AI will adjust its predictions based on your clinical context."
        )
        col_notes, col_example = st.columns([3, 1])
        with col_notes:
            doctor_notes = st.text_area(
                "Clinical notes",
                placeholder=(
                    "Example:\n"
                    "55M, ex-smoker (20 pack-years). Presenting with 3-week cough and haemoptysis.\n"
                    "Vitals: HR: 92 bpm, SpO2: 95%, Temp: 37.2°C, BP: 138/88.\n"
                    "Hx: COPD, hypertension. No fever. No chest pain.\n"
                    "Medications: Salbutamol inhaler, amlodipine."
                ),
                height=130,
                label_visibility="collapsed",
            )
        with col_example:
            if st.button("Load example notes"):
                st.session_state["example_notes"] = (
                    "55M, ex-smoker (20 pack-years). Presenting with 3-week cough and haemoptysis.\n"
                    "Vitals: HR: 92 bpm, SpO2: 95%, Temp: 37.2°C, BP: 138/88.\n"
                    "Hx: COPD, hypertension. No fever. No chest pain.\n"
                    "Medications: Salbutamol inhaler, amlodipine."
                )
            # Show parsed preview
            if doctor_notes.strip():
                parsed = parse_doctor_notes(doctor_notes)
                if parsed.vitals:
                    st.caption(f"Parsed vitals: {parsed.vitals}")
                if parsed.matched_keywords:
                    kws = [k for k in parsed.matched_keywords if not k.startswith("[")]
                    if kws:
                        st.caption(f"Clinical keywords detected: {', '.join(kws[:6])}")

    # Use example notes if loaded
    if "example_notes" in st.session_state and not doctor_notes.strip():
        doctor_notes = st.session_state.get("example_notes", "")

    # ── No image uploaded ───────────────────────────────────────────────────
    if uploaded is None:
        _render_demo_info()
        _render_chatbot_standalone(chatbot)
        return

    # ── Process image ───────────────────────────────────────────────────────
    pil_img      = Image.open(uploaded)
    image_tensor = preprocess_image(pil_img)

    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.markdown("**Original X-ray**")
        st.image(pil_img, use_container_width=True)

    with st.spinner("Running AI analysis..."):
        t0     = time.time()
        report = agent.diagnose(image_tensor, patient_id=patient_id, doctor_notes=doctor_notes)
        elapsed = time.time() - t0
        st.session_state["current_report"] = report

    # Bind chatbot to this report
    chatbot.set_report_context(report.raw_probabilities, threshold=settings["threshold"])

    with col_info:
        _render_urgency_banner(report.urgency_level)

        # Vitals flags from notes
        for flag in report.vitals_flags:
            icon = "🚨" if "CRITICAL" in flag else "⚠️"
            st.warning(f"{icon} {flag}")

        st.markdown("**Primary Impression**")
        st.info(report.primary_impression)
        st.caption(
            f"Analysis: {elapsed:.2f}s | Model: {report.model_used}"
            + (f" | Notes: {len(report.note_adjustments)} probability adjustments"
               if report.note_adjustments else "")
        )

    st.divider()

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab_pred, tab_gradcam, tab_report, tab_chat, tab_raw = st.tabs([
        "📊 Predictions",
        "🔥 GradCAM",
        "📋 Clinical Report",
        "💬 AI Chatbot",
        "🔬 Raw Data",
    ])

    with tab_pred:
        _render_predictions(report, settings["show_all"])

    with tab_gradcam:
        _render_gradcam(model, image_tensor, report, settings["n_heatmaps"])

    with tab_report:
        _render_clinical_report(report)

    with tab_chat:
        _render_chatbot(chatbot)

    with tab_raw:
        _render_raw_data(report)


# ── Tab renderers ──────────────────────────────────────────────────────────────
def _render_demo_info():
    st.info(
        "Upload a chest X-ray image to begin analysis.\n\n"
        "**New:** Add doctor's notes above to improve prediction accuracy — "
        "the AI adjusts probabilities based on your clinical context."
    )
    st.markdown("#### Supported Disease Labels")
    cols = st.columns(4)
    for i, label in enumerate(DISEASE_LABELS):
        cols[i % 4].markdown(f"- {label}")


def _render_urgency_banner(urgency: str):
    banners = {
        "emergency": ('<div class="urgent-banner">🚨 EMERGENCY — Immediate clinical evaluation required</div>', True),
        "urgent":    ('<div class="warning-banner">⚠️ URGENT — Prompt physician review recommended</div>', True),
        "routine":   ('<div class="routine-banner">✅ ROUTINE — Standard clinical follow-up suggested</div>', False),
    }
    html, _ = banners.get(urgency, banners["routine"])
    st.markdown(html, unsafe_allow_html=True)


def _render_predictions(report, show_all: bool):
    import plotly.graph_objects as go

    st.markdown("### Disease Probability Scores")

    # Note adjustments callout
    if report.note_adjustments:
        with st.expander(f"📝 {len(report.note_adjustments)} probabilities adjusted from clinical notes"):
            for adj in report.note_adjustments:
                st.markdown(f"- {adj}")

    probs   = report.raw_probabilities
    labels  = list(probs.keys())
    values  = [probs[l] * 100 for l in labels]
    colours = [prob_colour(probs[l]) for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colours,
        text=[f"{v:.1f}%" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title="Predicted Disease Probabilities",
        xaxis_title="Probability (%)", height=500,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(30,35,41,1)",
        font=dict(color="#cfd8dc"), xaxis=dict(range=[0, 115]),
    )
    st.plotly_chart(fig, use_container_width=True)

    if report.predicted_diseases:
        st.markdown("### Detected Conditions")
        for d in report.predicted_diseases:
            c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
            c1.markdown(f"**{d.name}**")
            c2.markdown(
                f'<span style="color:{prob_colour(d.probability)};font-weight:600;">'
                f'{d.probability*100:.1f}%</span>', unsafe_allow_html=True
            )
            if d.note_adjusted:
                c3.markdown(
                    f'<span class="note-adjusted">model: {d.model_prob*100:.1f}%</span>',
                    unsafe_allow_html=True,
                )
            c4.caption(f"Severity: {d.severity}")
    else:
        st.success("No significant pathology detected above confidence threshold.")


def _render_gradcam(model, image_tensor, report, n_heatmaps: int):
    st.markdown("### GradCAM — Explainability Heatmaps")
    st.caption("Red/yellow = high activation (model focused here), blue = low activation.")

    with st.spinner("Generating GradCAM..."):
        try:
            results = generate_multi_class_heatmaps(
                model=model, image_tensor=image_tensor,
                disease_labels=DISEASE_LABELS, top_k=n_heatmaps,
            )
        except Exception as e:
            st.error(f"GradCAM error: {e}")
            return

    cols = st.columns(min(n_heatmaps, 3))
    for i, (disease, prob, heatmap, overlay) in enumerate(results):
        with cols[i % len(cols)]:
            st.image(overlay, caption=f"{disease}: {prob*100:.1f}%", use_container_width=True)


def _render_clinical_report(report):
    st.markdown("### Recommended Actions")
    for i, action in enumerate(report.recommended_actions, 1):
        st.markdown(f"{i}. {action}")

    # Doctor notes summary
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

        if report.note_adjustments:
            st.markdown("**Note-based Adjustments**")
            for adj in report.note_adjustments:
                st.markdown(f"- {adj}")

    # Per-disease details
    if report.predicted_diseases:
        st.markdown("### Disease Details")
        for d in report.predicted_diseases:
            adj_str = f" *(note-adjusted from {d.model_prob*100:.1f}%)*" if d.note_adjusted else ""
            with st.expander(f"🔍 {d.name}  ({d.probability*100:.1f}%){adj_str}"):
                if d.overview:
                    st.markdown(f"**Overview:** {d.overview}")
                c1, c2 = st.columns(2)
                with c1:
                    if d.symptoms:
                        st.markdown("**Symptoms**")
                        for s in d.symptoms: st.markdown(f"- {s}")
                    if d.risk_factors:
                        st.markdown("**Risk Factors**")
                        for r in d.risk_factors: st.markdown(f"- {r}")
                with c2:
                    if d.treatment:
                        st.markdown("**Treatment**")
                        for t in d.treatment: st.markdown(f"- {t}")
                    if d.follow_up:
                        st.info(f"**Follow-up:** {d.follow_up}")

    st.divider()
    st.caption(f"⚠️ {report.disclaimer}")


def _render_chatbot(chatbot: MedicalChatbot):
    """RAG chatbot tab — grounded in the current patient's report."""
    st.markdown("### 💬 Medical AI Chatbot")
    st.caption(
        "Ask questions about the detected conditions, treatments, follow-up steps, "
        "or any medical topic. The chatbot is grounded in this patient's findings."
    )

    # Suggested question buttons
    suggestions = chatbot.get_suggested_questions()
    if suggestions:
        st.markdown("**Quick questions:**")
        btn_cols = st.columns(3)
        for i, q in enumerate(suggestions[:6]):
            if btn_cols[i % 3].button(q, key=f"sugg_{i}"):
                _submit_chat(chatbot, q)

    st.markdown("---")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        if not chatbot.history:
            st.markdown(
                '<div class="chat-assistant">Hello! I\'m your Medical AI Assistant. '
                'Ask me about the findings in this X-ray, treatment options, '
                'or any clinical questions.</div>',
                unsafe_allow_html=True,
            )
        for msg in chatbot.history:
            css_class = "chat-user" if msg.role == "user" else "chat-assistant"
            # Escape HTML in content but preserve markdown-like bold
            content = msg.content.replace("<", "&lt;").replace(">", "&gt;")
            st.markdown(
                f'<div class="{css_class}">{content}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Input row
    col_input, col_send, col_clear = st.columns([5, 1, 1])
    with col_input:
        user_input = st.text_input(
            "Your question",
            key="chat_input",
            placeholder="e.g. How is Pneumonia treated? What does SpO2 of 95% mean?",
            label_visibility="collapsed",
        )
    with col_send:
        send = st.button("Send", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            chatbot.history = []
            st.rerun()

    if send and user_input.strip():
        _submit_chat(chatbot, user_input)


def _submit_chat(chatbot: MedicalChatbot, message: str):
    """Send a message and trigger rerun to display the response."""
    chatbot.chat(message)
    st.rerun()


def _render_chatbot_standalone(chatbot: MedicalChatbot):
    """Chatbot shown even without an uploaded image."""
    st.divider()
    st.markdown("### 💬 Medical Knowledge Chatbot")
    st.caption("You can use the chatbot without uploading an image to ask general medical questions.")

    col_input, col_send = st.columns([5, 1])
    with col_input:
        q = st.text_input("Ask a medical question", key="standalone_chat",
                          placeholder="e.g. What are the symptoms of Pneumonia?",
                          label_visibility="collapsed")
    with col_send:
        if st.button("Ask", key="standalone_send", type="primary"):
            if q.strip():
                chatbot.chat(q)

    for msg in chatbot.history[-6:]:   # show last 3 exchanges
        css = "chat-user" if msg.role == "user" else "chat-assistant"
        st.markdown(f'<div class="{css}">{msg.content}</div>', unsafe_allow_html=True)


def _render_raw_data(report):
    import pandas as pd

    st.markdown("### Raw Probability Data")
    rows = [
        {
            "Disease":       d,
            "AI Prob (%)":   f"{p*100:.2f}",
            "Detected":      "Yes" if p >= 0.45 else "No",
        }
        for d, p in sorted(report.raw_probabilities.items(), key=lambda x: x[1], reverse=True)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if report.doctor_notes.strip():
        st.markdown("### Doctor's Notes (submitted)")
        st.code(report.doctor_notes, language="text")

    st.markdown("### Full Report Text")
    agent_fmt = DiagnosisAgent.__new__(DiagnosisAgent)
    lines = [
        f"Patient ID : {report.patient_id}",
        f"Model      : {report.model_used}",
        f"Urgency    : {report.urgency_level.upper()}",
        "",
        "IMPRESSION:",
        report.primary_impression,
    ]
    if report.vitals_flags:
        lines += ["", "VITALS ALERTS:"] + [f"  {f}" for f in report.vitals_flags]
    if report.note_adjustments:
        lines += ["", "NOTE ADJUSTMENTS:"] + [f"  {a}" for a in report.note_adjustments]
    lines += [
        "",
        "DETECTED CONDITIONS:",
        *[f"  {d.name}: {d.probability*100:.1f}% (severity: {d.severity})"
          + (f" [note-adjusted from {d.model_prob*100:.1f}%]" if d.note_adjusted else "")
          for d in report.predicted_diseases],
        "",
        "RECOMMENDED ACTIONS:",
        *[f"  {i+1}. {a}" for i, a in enumerate(report.recommended_actions)],
        "",
        f"DISCLAIMER: {report.disclaimer}",
    ]
    st.code("\n".join(lines), language="text")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Need this import here to avoid circular issues in format_report raw data tab
    from agent.diagnosis_agent import DiagnosisAgent
    main()
else:
    from agent.diagnosis_agent import DiagnosisAgent
    main()
