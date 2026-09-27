from __future__ import annotations

import html
import io
import re
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle


ROOT = Path(r"C:\projects\FabJudge-PHM2018-starter\FabJudge-PHM2018")
SOURCE = ROOT / "docs/FabJudge_portfolio_ko.md"
FIGURE = ROOT / "docs/figures/S1_alarm_preserving_pipeline.pdf"
OUTPUT = ROOT / "output/pdf/FabJudge_portfolio_ko.pdf"
TMP = ROOT / "tmp/pdfs"

pdfmetrics.registerFont(TTFont("NotoKR", r"C:\Windows\Fonts\NotoSansKR-Regular.ttf"))
pdfmetrics.registerFont(TTFont("NotoKR-Bold", r"C:\Windows\Fonts\NotoSansKR-Bold.ttf"))
pdfmetrics.registerFontFamily("NotoKR", normal="NotoKR", bold="NotoKR-Bold")

INK = colors.HexColor("#25364a")
TEAL = colors.HexColor("#117d73")
GRAY = colors.HexColor("#5c6a78")
LIGHT = colors.HexColor("#e9f3f1")
RULE = colors.HexColor("#d5e0e3")
PAPER_WIDTH = A4[0] - 100

ST = {
    "brand": ParagraphStyle("brand", fontName="NotoKR-Bold", fontSize=25, leading=31, textColor=TEAL, spaceAfter=5),
    "title": ParagraphStyle("title", fontName="NotoKR-Bold", fontSize=17, leading=25, textColor=INK, spaceAfter=10),
    "deck": ParagraphStyle("deck", fontName="NotoKR", fontSize=10, leading=16, textColor=GRAY, spaceAfter=10),
    "meta": ParagraphStyle("meta", fontName="NotoKR", fontSize=8, leading=12, textColor=GRAY, spaceAfter=17),
    "h1": ParagraphStyle("h1", fontName="NotoKR-Bold", fontSize=13.2, leading=20, textColor=INK, spaceBefore=18, spaceAfter=10, keepWithNext=True),
    "h2": ParagraphStyle("h2", fontName="NotoKR-Bold", fontSize=10.8, leading=17, textColor=TEAL, spaceBefore=14, spaceAfter=8, keepWithNext=True),
    "summary": ParagraphStyle("summary", fontName="NotoKR-Bold", fontSize=11.3, leading=18, textColor=TEAL, spaceBefore=3, spaceAfter=9, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="NotoKR", fontSize=9.05, leading=15.2, textColor=INK, spaceAfter=10.5, wordWrap="CJK", splitLongWords=1),
    "caption": ParagraphStyle("caption", fontName="NotoKR", fontSize=8.2, leading=13.5, textColor=GRAY, spaceBefore=5, spaceAfter=8, wordWrap="CJK"),
    "ref": ParagraphStyle("ref", fontName="NotoKR", fontSize=7.9, leading=12.8, textColor=INK, spaceAfter=8, wordWrap="CJK", splitLongWords=1),
    "table": ParagraphStyle("table", fontName="NotoKR", fontSize=7.4, leading=11.3, textColor=INK, wordWrap="CJK", splitLongWords=1),
    "tablehead": ParagraphStyle("tablehead", fontName="NotoKR-Bold", fontSize=7.4, leading=11.3, textColor=INK, wordWrap="CJK", splitLongWords=1),
}

LOCAL_REFS = {
    "A1": "artifacts/08_final_comparison/tables/T5_full_endpoint_comparison.csv",
    "A2": "artifacts/08_final_comparison/tables/T6_sequence_event_detection.csv",
    "A3": "artifacts/07b_rescue_lane_confirmatory/prereg_07_v2_1.json",
    "A4": "artifacts/07b_rescue_lane_confirmatory/hypothesis_tests.json",
    "A5": "artifacts/07b_rescue_lane_confirmatory/summary.json",
    "A6": "artifacts/07b_rescue_lane_confirmatory/llm_noRF_results.jsonl",
    "A7": "artifacts/08_final_comparison/tables/T4_pitfalls.csv",
    "A8": "artifacts/08_final_comparison/tables/F7_full_endpoint_curve.csv",
    "A9": "artifacts/08_final_comparison/tables/T3_experiment_history.csv",
    "A10": "artifacts/06d_lstm_jev_oof_edge/ERRATUM.md",
    "A11": "artifacts/06e_fix_and_weakness_diagnosis/conclusion.md",
    "A12": "artifacts/06c_lstm_jev_semantic_gate/summary.json",
    "A13": "artifacts/huang2018/sequence_metadata.csv",
    "A14": "artifacts/07b_rescue_lane_confirmatory/actual_request_body_example.json",
    "A15": "artifacts/08_final_comparison/tables/source_manifest.csv",
    "A16": "artifacts/07b_rescue_lane_confirmatory/trial_log.md",
}
WEB_REFS = {
    "R1": "https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/",
    "R2": "https://doi.org/10.36001/phmconf.2018.v10i1.590",
    "R3": "https://doi.org/10.36001/phmconf.2018.v10i1.591",
    "R4": "https://doi.org/10.36001/phmconf.2018.v10i1.589",
}


def rich(raw: str) -> str:
    escaped = html.escape(raw)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", r'<font color="#176d81">\1</font>', escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)

    def make_ref(match: re.Match[str]) -> str:
        ref = match.group(1)
        url = WEB_REFS.get(ref)
        if url is None and ref in LOCAL_REFS:
            url = "file:///" + (ROOT / LOCAL_REFS[ref]).as_posix()
        if url is None:
            return match.group(0)
        return f'<link href="{html.escape(url, quote=True)}" color="#137d87">[{ref}]</link>'

    escaped = re.sub(r"\[(A\d+|R\d+)\]", make_ref, escaped)
    return escaped


def footer(canv: canvas.Canvas, doc: BaseDocTemplate) -> None:
    canv.saveState()
    w, h = A4
    canv.setStrokeColor(RULE)
    canv.setLineWidth(0.5)
    canv.line(50, 39, w - 50, 39)
    canv.setFillColor(GRAY)
    canv.setFont("NotoKR", 7)
    canv.drawString(50, 25, "FabJudge  /  PHM2018 case study")
    canv.restoreState()


def table_from_lines(lines: list[str]) -> Table:
    rows = []
    for line in lines:
        cells = [s.strip() for s in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        rows.append([Paragraph(rich(s), ST["tablehead"] if not rows else ST["table"]) for s in cells])
    count = len(rows[0])
    if count == 7:
        widths = [114, 70, 35, 35, 73, 84, 84]
    elif count == 3:
        widths = [166, 104, 225]
    elif count == 2:
        widths = [40, 455]
    else:
        widths = [PAPER_WIDTH / count] * count
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafb")]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#a9c8c8")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build_chunk(lines: list[str], path: Path, first: bool) -> None:
    w, h = A4
    doc = BaseDocTemplate(str(path), pagesize=A4, leftMargin=50, rightMargin=50, topMargin=46, bottomMargin=54)
    frame = Frame(50, 54, w - 100, h - 100, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=footer))
    flow = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flow.append(Spacer(1, 3))
            i += 1
            continue
        if line.startswith("|"):
            group = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                group.append(lines[i])
                i += 1
            flow.extend([table_from_lines(group), Spacer(1, 11)])
            continue
        if line.startswith("# "):
            style = ST["brand"] if first and line == "# FabJudge" else ST["h1"]
            line = line[2:]
        elif line.startswith("## "):
            style = ST["title"] if first and line.startswith("## JEV") else ST["h2"]
            line = line[3:]
        elif line.startswith("### "):
            style = ST["summary"]
            line = line[4:]
        elif first and line.startswith("PHM2018 이온"):
            style = ST["deck"]
        elif first and line.startswith("**프로젝트"):
            style = ST["meta"]
        elif line.startswith("**그림 S1."):
            style = ST["caption"]
        elif line.startswith("**R") and line[3:4].isdigit():
            style = ST["ref"]
        else:
            style = ST["body"]
        flow.append(Paragraph(rich(line), style))
        i += 1
    doc.build(flow)


def overlay_number(page_number: int, width: float, height: float) -> PdfReader:
    stream = io.BytesIO()
    c = canvas.Canvas(stream, pagesize=(width, height))
    c.setFillColor(GRAY)
    c.setFont("NotoKR", 7)
    if width > height:
        c.drawRightString(width - 18, height - 19, str(page_number))
    else:
        c.drawRightString(width - 50, 25, str(page_number))
    c.save()
    stream.seek(0)
    return PdfReader(stream)


def main() -> None:
    content = SOURCE.read_text(encoding="utf-8").splitlines()
    marker = content.index("![그림 S1: 경보 보존 파이프라인](figures/S1_alarm_preserving_pipeline.png)")
    before_path = TMP / "portfolio_before_figure.pdf"
    after_path = TMP / "portfolio_after_figure.pdf"
    build_chunk(content[:marker], before_path, True)
    build_chunk(content[marker + 1 :], after_path, False)

    writer = PdfWriter()
    for part in (before_path, FIGURE, after_path):
        writer.append(PdfReader(str(part)))
    for n, page in enumerate(writer.pages, 1):
        page.merge_page(overlay_number(n, float(page.mediabox.width), float(page.mediabox.height)).pages[0])
    writer.add_metadata({
        "/Title": "FabJudge: JEV–LLM은 이상 탐지에서 Edge를 만들 수 있는가?",
        "/Subject": "PHM2018 RF–LSTM baseline and selective decision layer portfolio",
        "/Author": "FabJudge",
    })
    with OUTPUT.open("wb") as stream:
        writer.write(stream)
    print(f"OUTPUT={OUTPUT}")
    print(f"PAGES={len(writer.pages)} BEFORE={len(PdfReader(str(before_path)).pages)} FIGURE=1 AFTER={len(PdfReader(str(after_path)).pages)}")


if __name__ == "__main__":
    main()
