# -*- coding: utf-8 -*-
"""
LAPE Enterprise PDF Report Generator
Generates reproducible, executive-ready financial triage reports with INR formatting,
audit trails, and compliance disclaimers using ReportLab.
"""
import io
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from src.currency import format_inr


def _register_fonts():
    """Register system TrueType font for Unicode INR symbol support."""
    font_paths = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    registered_name = "Helvetica"
    for p in font_paths:
        if os.path.exists(p):
            try:
                pdfmetrics.registerFont(TTFont("EnterpriseFont", p))
                registered_name = "EnterpriseFont"
                break
            except Exception:
                continue
    return registered_name


def generate_lape_pdf_report(
    summary_data: dict,
    top_alerts: pd.DataFrame,
    benchmarks: dict = None,
    scope: str = "All Domains",
    include_methodology: bool = True,
    include_quality: bool = True,
    include_evaluation: bool = True,
    date_format: str = "%d/%m/%Y",
) -> bytes:
    """
    Generates a professional executive PDF report in memory and returns bytes.
    """
    font_name = _register_fonts()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0B0B0C"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#5F6368"),
        spaceAfter=14,
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1A73E8"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#202124"),
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#5F6368"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#5F6368"),
        backColor=colors.HexColor("#F1F3F4"),
        borderPadding=6,
        spaceBefore=10,
    )

    story = []

    # Title & Header
    story.append(Paragraph("<b>LAPE: Leak Analysis &amp; Prediction Engine</b>", title_style))
    curr_date_str = datetime.now().strftime(date_format)
    story.append(
        Paragraph(
            f"<b>Summary Report: Finding and Fixing Lost Money</b> &nbsp;|&nbsp; "
            f"Date: <b>{curr_date_str}</b> &nbsp;|&nbsp; Scope: <b>{scope}</b> &nbsp;|&nbsp; "
            f"Currency: <b>INR (₹)</b> &nbsp;|&nbsp; Classification: <b>Internal Review</b>",
            subtitle_style,
        )
    )

    # 1. Summary of Checks
    if include_quality:
        story.append(Paragraph("1. Summary of Checks and Possible Money at Risk", h2_style))
        kpi_table_data = [
            [
                "Area Checked",
                "Total Records Checked",
                "Problems Found",
                "Possible Money at Risk (₹)",
            ],
            [
                "Orders Sold Below Cost",
                f"{summary_data.get('total_tx', 13962):,} orders",
                f"{summary_data.get('completed_neg', 5315):,} completed lines",
                format_inr(summary_data.get("tx_shortfall", 1214115.50)),
            ],
            [
                "Ad Campaigns with Broken Numbers",
                f"{summary_data.get('total_camp', 1200):,} campaigns",
                f"{summary_data.get('camp_viol', 171):,} irregular campaigns",
                format_inr(summary_data.get("camp_spend", 100105.00)),
            ],
            [
                "Warehouse Stock Count Mismatches",
                f"{summary_data.get('total_inv', 2000):,} stock records",
                f"{summary_data.get('inv_disc', 1253):,} missed stockout flags",
                "Operational (Risk of running out of stock)",
            ],
            [
                "Total Combined Workload",
                "27,450 records",
                f"{summary_data.get('total_alerts', 6739):,} warnings to review",
                format_inr(summary_data.get("total_exposure", 1314220.50)),
            ],
        ]
        t_kpi = Table(kpi_table_data, colWidths=[160, 100, 130, 150])
        t_kpi.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F3F4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#202124")),
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DADCE0")),
                ("FONTNAME", (0, 0), (-1, 0), font_name),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ])
        )
        story.append(t_kpi)
        story.append(Spacer(1, 10))

    # 2. Top Prioritized Warnings
    story.append(Paragraph("2. Most Urgent Warnings to Review", h2_style))
    if top_alerts is not None and not top_alerts.empty:
        sample_df = top_alerts.head(8).copy()
        alert_rows = [
            ["Warning ID", "Area", "Original Record ID", "Problem Found", "Proof Quality", "Urgency", "Money at Risk (₹)"]
        ]
        for _, row in sample_df.iterrows():
            alert_rows.append([
                str(row.get("alert_id", "")),
                str(row.get("domain", "")),
                str(row.get("source_record_id", "")),
                str(row.get("exposure_nature", ""))[:20],
                str(row.get("evidence_reliability", "")),
                str(row.get("action_urgency", "")),
                format_inr(row.get("potential_exposure_amount", 0.0)),
            ])
        t_alerts = Table(alert_rows, colWidths=[65, 75, 75, 115, 65, 65, 80])
        t_alerts.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F3F4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#202124")),
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DADCE0")),
                ("ALIGN", (-1, 0), (-1, -1), "RIGHT"),
            ])
        )
        story.append(t_alerts)
        story.append(Spacer(1, 10))

    # 3. Model Benchmark Evaluation
    if include_evaluation and benchmarks:
        story.append(Paragraph("3. Testing Model Accuracy on Artificial Test Data", h2_style))
        bench_rows = [
            ["Detection Method", "Accuracy When Flagging", "Problems Caught", "Overall Score", "How It Was Tested"]
        ]
        if "comparison_table" in benchmarks:
            for item in benchmarks["comparison_table"]:
                method_name = item.get("Method") or item.get("Algorithm", "")
                f1_val = item.get("F1-Score") if "F1-Score" in item else item.get("F1 Score", 0.0)
                bench_rows.append([
                    str(method_name),
                    f"{float(item.get('Precision', 0.0)):.3f}",
                    f"{float(item.get('Recall', 0.0)):.3f}",
                    f"{float(f1_val):.3f}",
                    "Controlled Artificial Scenarios",
                ])
        t_bench = Table(bench_rows, colWidths=[150, 70, 70, 70, 180])
        t_bench.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F3F4")),
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DADCE0")),
                ("ALIGN", (1, 0), (3, -1), "CENTER"),
            ])
        )
        story.append(t_bench)
        story.append(Spacer(1, 8))

    # 4. Methodology & Governance Notice
    if include_methodology:
        story.append(Paragraph("4. How These Calculations Work", h2_style))
        story.append(
            Paragraph(
                "• <b>How money at risk is calculated:</b> Number of Items × (Unit Cost - Selling Price). "
                "This estimates potential money at risk, not confirmed losses, until real receipts and invoices are checked.<br/>"
                "• <b>How reliable is the information:</b> Records with matching receipts and confirmed orders have HIGH reliability; "
                "orders still in progress have MEDIUM reliability; incomplete records have LOW reliability.<br/>"
                "• <b>How quickly to act:</b> CRITICAL items should be paused and checked immediately before closing the books; "
                "MEDIUM items can be reviewed during regular weekly checks.",
                body_style,
            )
        )
        story.append(Spacer(1, 6))

    # Disclaimer
    disclaimer_text = (
        "<b>NOTE &amp; COMPLIANCE NOTICE:</b> This report was generated automatically by LAPE for the "
        "DATATHON 2026 Track 4 demonstration. All values are presented in Indian Rupees (₹) using "
        "artificial demonstration data. Model accuracy scores reflect testing on controlled artificial test sets. "
        "Warnings highlight possible issues that should be verified with real receipts and invoices before taking financial action."
    )
    story.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(story)
    return buf.getvalue()
