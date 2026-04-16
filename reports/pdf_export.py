"""
PDF Report Export
Generates a clinical-grade PDF from a DiagnosisReport, optionally
embedding the GradCAM heatmap overlay image.

Requires: reportlab  (pip install reportlab)

Usage:
    from reports.pdf_export import export_report_pdf

    pdf_bytes = export_report_pdf(
        report        = report,
        heatmap_path  = "/tmp/gradcam_pneumonia.png",   # optional
        output_path   = "reports/output/PT-001.pdf",    # optional; returns bytes if None
    )
"""

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── ReportLab imports ──────────────────────────────────────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        Image as RLImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    _REPORTLAB_AVAILABLE = True

    # ── Colour palette (defined here so `colors` is guaranteed in scope) ──────
    _RED    = colors.HexColor("#c62828")
    _ORANGE = colors.HexColor("#e65100")
    _GREEN  = colors.HexColor("#2e7d32")
    _BLUE   = colors.HexColor("#1565c0")
    _GREY   = colors.HexColor("#455a64")
    _LIGHT  = colors.HexColor("#eceff1")

except ImportError:
    _REPORTLAB_AVAILABLE = False
    colors = None  # type: ignore[assignment]
    _RED = _ORANGE = _GREEN = _BLUE = _GREY = _LIGHT = None


def _require_reportlab() -> None:
    if not _REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is required for PDF export. Install with: pip install reportlab"
        )


def _urgency_colour(urgency: str):
    return {"emergency": _RED, "urgent": _ORANGE, "routine": _GREEN}.get(urgency, _GREY)


def _severity_colour(severity: str):
    return {"critical": _RED, "high": _ORANGE, "moderate": _ORANGE, "low": _GREEN}.get(severity, _GREY)


# ── Main export function ───────────────────────────────────────────────────────
def export_report_pdf(
    report,
    heatmap_path: Optional[str]  = None,
    output_path:  Optional[str]  = None,
    hospital_name: str           = "Federated Medical AI System",
) -> bytes:
    """
    Build a clinical PDF from a DiagnosisReport.

    Args:
        report:       DiagnosisReport instance from DiagnosisAgent
        heatmap_path: optional path to a GradCAM overlay PNG to embed
        output_path:  if provided, also write the PDF to disk
        hospital_name: header institution name

    Returns:
        PDF content as bytes (always, regardless of output_path)
    """
    _require_reportlab()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize     = A4,
        rightMargin  = 2 * cm,
        leftMargin   = 2 * cm,
        topMargin    = 2 * cm,
        bottomMargin = 2 * cm,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Helper styles ──────────────────────────────────────────────────────
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, textColor=_BLUE,
                         spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, textColor=_GREY,
                         spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, leading=13)
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=7.5,
                            textColor=_GREY, leading=11)
    bold_body = ParagraphStyle("Bold", parent=body, fontName="Helvetica-Bold")
    centered = ParagraphStyle("Centered", parent=body, alignment=TA_CENTER)
    right    = ParagraphStyle("Right",    parent=small, alignment=TA_RIGHT)

    def hr():
        return HRFlowable(width="100%", thickness=0.5, color=_GREY, spaceAfter=6)

    # ── Header ─────────────────────────────────────────────────────────────
    story.append(Paragraph(hospital_name, h1))
    story.append(Paragraph("AI-Assisted Chest X-ray Diagnostic Report", h2))
    story.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  |  "
        f"Patient: <b>{report.patient_id}</b>  |  Model: <b>{report.model_used}</b>",
        small,
    ))
    story.append(hr())

    # ── Urgency banner ─────────────────────────────────────────────────────
    urgency_col = _urgency_colour(report.urgency_level)
    urgency_tbl = Table(
        [[Paragraph(f"URGENCY: {report.urgency_level.upper()}", bold_body)]],
        colWidths=["100%"],
    )
    urgency_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), urgency_col),
        ("TEXTCOLOR",  (0, 0), (-1, -1), colors.white),
        ("PADDING",    (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(urgency_tbl)
    story.append(Spacer(1, 0.3 * cm))

    # ── Primary impression ─────────────────────────────────────────────────
    story.append(Paragraph("Primary Impression", h2))
    story.append(Paragraph(report.primary_impression, body))

    # ── Vitals alerts ──────────────────────────────────────────────────────
    if report.vitals_flags:
        story.append(Paragraph("Vitals Alerts", h2))
        for flag in report.vitals_flags:
            story.append(Paragraph(f"• {flag}", body))

    # ── Detected conditions table ──────────────────────────────────────────
    if report.predicted_diseases:
        story.append(Paragraph("Detected Conditions", h2))
        table_data = [["Disease", "Probability", "Severity", "Note-Adjusted"]]
        for d in report.predicted_diseases:
            adj = "Yes" if d.note_adjusted else "No"
            table_data.append([d.name, f"{d.probability*100:.1f}%", d.severity.upper(), adj])

        tbl = Table(table_data, colWidths=[5 * cm, 3 * cm, 3 * cm, 3 * cm])
        tbl_style = TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), _BLUE),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT]),
            ("GRID",        (0, 0), (-1, -1), 0.4, _GREY),
            ("PADDING",     (0, 0), (-1, -1), 5),
            ("ALIGN",       (1, 1), (-1, -1), "CENTER"),
        ])
        # Colour severity cells
        for row_idx, d in enumerate(report.predicted_diseases, start=1):
            tbl_style.add("TEXTCOLOR", (2, row_idx), (2, row_idx), _severity_colour(d.severity))
        tbl.setStyle(tbl_style)
        story.append(tbl)

    # ── Note adjustments ───────────────────────────────────────────────────
    if report.note_adjustments:
        story.append(Paragraph("Note-Based Probability Adjustments", h2))
        for adj in report.note_adjustments:
            story.append(Paragraph(f"→ {adj}", body))

    # ── Recommended actions ────────────────────────────────────────────────
    story.append(Paragraph("Recommended Actions", h2))
    for i, action in enumerate(report.recommended_actions, 1):
        story.append(Paragraph(f"{i}. {action}", body))

    # ── All disease probabilities (small table) ────────────────────────────
    story.append(Paragraph("Full Disease Probability Scores", h2))
    sorted_probs = sorted(report.raw_probabilities.items(), key=lambda x: x[1], reverse=True)
    prob_rows = [["Disease", "Probability"]] + [
        [d, f"{p*100:.2f}%"] for d, p in sorted_probs
    ]
    prob_tbl = Table(prob_rows, colWidths=[8 * cm, 4 * cm])
    prob_tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), _GREY),
        ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _LIGHT]),
        ("GRID",           (0, 0), (-1, -1), 0.3, _GREY),
        ("PADDING",        (0, 0), (-1, -1), 4),
        ("ALIGN",          (1, 1), (-1, -1), "RIGHT"),
    ]))
    story.append(prob_tbl)

    # ── GradCAM heatmap ────────────────────────────────────────────────────
    if heatmap_path and os.path.exists(heatmap_path):
        story.append(Paragraph("GradCAM Activation Map", h2))
        story.append(Paragraph(
            "Regions highlighted in red indicate areas that most influenced the model prediction.",
            small,
        ))
        story.append(Spacer(1, 0.2 * cm))
        img = RLImage(heatmap_path, width=8 * cm, height=8 * cm)
        story.append(img)

    # ── Doctor's notes ─────────────────────────────────────────────────────
    if report.doctor_notes.strip():
        story.append(Paragraph("Doctor's Clinical Notes", h2))
        story.append(Paragraph(report.doctor_notes.replace("\n", "<br/>"), body))

    # ── Disclaimer ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(hr())
    story.append(Paragraph("DISCLAIMER", bold_body))
    story.append(Paragraph(report.disclaimer, small))

    # ── Build PDF ──────────────────────────────────────────────────────────
    doc.build(story)
    pdf_bytes = buf.getvalue()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes
