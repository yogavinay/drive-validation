from __future__ import annotations

from io import BytesIO
from typing import Any, Dict

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def build_pdf(report: Dict[str, Any]) -> bytes:
    buff = BytesIO()
    doc = SimpleDocTemplate(buff, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Validation Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Report ID: {report.get('report_id', '')}", styles["Normal"]))
    elements.append(Paragraph(f"Drive URL: {report.get('drive_url', '')}", styles["Normal"]))
    summary = report.get("summary", {})
    elements.append(Paragraph(f"Summary: {summary}", styles["Normal"]))
    elements.append(Spacer(1, 12))
    for item in report.get("files", []):
        elements.append(Paragraph(f"File: {item.get('file_name')}", styles["Heading3"]))
        elements.append(Paragraph(f"Status: {item.get('status')}", styles["Normal"]))
        for rule in item.get("rule_evaluations", []):
            elements.append(
                Paragraph(
                    f"Rule: {rule.get('rule')} | {rule.get('status')} | Failed lines: {rule.get('failed_lines')}",
                    styles["Normal"],
                )
            )
        elements.append(Spacer(1, 8))
    doc.build(elements)
    return buff.getvalue()
