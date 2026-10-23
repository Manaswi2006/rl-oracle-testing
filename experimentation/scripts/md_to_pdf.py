#!/usr/bin/env python
"""Minimal Markdown -> PDF (reportlab) with page numbers, tables, images, code blocks and LaTeX-style equations (matplotlib mathtext).
Run with .venv_report/bin/python:  md_to_pdf.py in.md out.pdf
Supported: #/##/### headings, paragraphs, -/1. lists, pipe tables, ``` code, ![alt](img), $$display$$, $inline$, **b**, *i*, `code`, [text](url).
"""
import hashlib, os, re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle)

FD = matplotlib.get_data_path() + "/fonts/ttf/"   # DejaVu family shipped with matplotlib (incl. oblique)
for n, f in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"), ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(n, FD + f))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
INK, INK2 = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e")
ST = dict(
    body=ParagraphStyle("body", fontName="DV", fontSize=8.6, leading=12, textColor=INK, spaceAfter=5),
    h1=ParagraphStyle("h1", fontName="DV-B", fontSize=17, leading=21, spaceBefore=4, spaceAfter=8),
    h2=ParagraphStyle("h2", fontName="DV-B", fontSize=12.5, leading=16, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#1f4e8c")),
    h3=ParagraphStyle("h3", fontName="DV-B", fontSize=10, leading=13, spaceBefore=8, spaceAfter=3),
    li=ParagraphStyle("li", fontName="DV", fontSize=8.6, leading=12, leftIndent=14, bulletIndent=3, spaceAfter=2),
    cell=ParagraphStyle("cell", fontName="DV", fontSize=6.9, leading=8.6),
    cellh=ParagraphStyle("cellh", fontName="DV-B", fontSize=6.9, leading=8.6),
    code=ParagraphStyle("code", fontName="DVM", fontSize=6.8, leading=8.4, backColor=colors.HexColor("#f2f1ee"), leftIndent=4, rightIndent=4, spaceAfter=6),
    cap=ParagraphStyle("cap", fontName="DV-I", fontSize=7.5, leading=9.5, textColor=INK2, spaceAfter=6),
)
CACHE = Path(os.environ.get("MD2PDF_CACHE", "/tmp/md2pdf_cache")); CACHE.mkdir(parents=True, exist_ok=True)


def math_png(tex, size=9.5):
    h = hashlib.md5((tex + str(size)).encode()).hexdigest()
    p = CACHE / f"{h}.png"
    if not p.exists():
        fig = plt.figure(figsize=(0.1, 0.1)); fig.text(0, 0, f"${tex}$", fontsize=size, color="#0b0b0b")
        try:
            fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
        except Exception:
            plt.close(fig); fig = plt.figure(figsize=(0.1, 0.1)); fig.text(0, 0, tex, fontsize=size)
            fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.02, transparent=True)
        plt.close(fig)
    w, hh = PILImage.open(p).size
    return str(p), w * 72 / 300, hh * 72 / 300


def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", lambda m: f'<font face="DVM" size="7.4">{m.group(1)}</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#1f4e8c">\1</link>', t)

    def m_(m):
        p, w, h = math_png(m.group(1).replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"), 9)
        return f'<img src="{p}" width="{w:.1f}" height="{h:.1f}" valign="middle"/>'
    t = re.sub(r"(?<!\$)\$([^$\n]+)\$(?!\$)", m_, t)
    return t


def table(lines, avail):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\s*\|?\s*:?-{2,}", l)]
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    mx = [max(min(len(r[i]), 40) for r in rows) + 3 for i in range(n)]
    tot = sum(mx); widths = [avail * m / tot for m in mx]
    data = [[Paragraph(inline(c), ST["cellh" if i == 0 else "cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cfcdc7")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ecebe6")),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def build(md_path, out):
    base = Path(md_path).parent
    lines = open(md_path, encoding="utf-8").read().split("\n")
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.7 * cm, bottomMargin=1.7 * cm, title=Path(md_path).stem)
    avail = A4[0] - 3.6 * cm
    fl, i, para = [], 0, []

    def flush():
        nonlocal para
        if para:
            fl.append(Paragraph(inline(" ".join(para)), ST["body"])); para = []
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):
            flush(); j = i + 1; buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j][:130]); j += 1
            fl.append(Preformatted("\n".join(buf), ST["code"])); i = j + 1; continue
        if l.strip().startswith("$$"):
            flush(); txt = l.strip()
            if txt.count("$$") < 2:
                j = i + 1; buf = [txt[2:]]
                while j < len(lines) and "$$" not in lines[j]:
                    buf.append(lines[j]); j += 1
                buf.append(lines[j].replace("$$", "")); txt = "$$" + " ".join(buf) + "$$"; i = j
            p, w, h = math_png(txt.strip("$").strip(), 11)
            if w > avail: h *= avail / w; w = avail
            fl.append(Image(p, width=w, height=h)); fl.append(Spacer(1, 5)); i += 1; continue
        if l.startswith("|"):
            flush(); j = i; tb = []
            while j < len(lines) and lines[j].startswith("|"):
                tb.append(lines[j]); j += 1
            fl.append(table(tb, avail)); fl.append(Spacer(1, 6)); i = j; continue
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", l)
        if m:
            flush(); p = (base / m.group(2)).resolve()
            if p.exists():
                w, h = PILImage.open(p).size; sc = min(avail / w, 1.0) * avail / min(avail, w) if False else avail / w
                fl.append(KeepTogether([Image(str(p), width=avail, height=h * avail / w), Paragraph(inline(m.group(1)), ST["cap"])]))
            i += 1; continue
        if l.startswith("# "): flush(); fl.append(Paragraph(inline(l[2:]), ST["h1"]))
        elif l.startswith("## "): flush(); fl.append(Paragraph(inline(l[3:]), ST["h2"]))
        elif l.startswith("### "): flush(); fl.append(Paragraph(inline(l[4:]), ST["h3"]))
        elif re.match(r"^\s*[-*] ", l): flush(); fl.append(Paragraph(inline(re.sub(r"^\s*[-*] ", "", l)), ST["li"], bulletText="•"))
        elif re.match(r"^\s*\d+\. ", l):
            flush(); mm = re.match(r"^\s*(\d+)\. (.*)", l); fl.append(Paragraph(inline(mm.group(2)), ST["li"], bulletText=mm.group(1) + "."))
        elif l.strip() == "---": flush(); fl.append(Spacer(1, 4))
        elif l.strip() == "": flush()
        else: para.append(l.strip())
        i += 1
    flush()

    def footer(c, d):
        c.saveState(); c.setFont("DV", 7.5); c.setFillColor(INK2)
        c.drawString(1.8 * cm, 1.0 * cm, "CoRe extensions 2026 - P-SVM / FACR / RLF feasibility study"); c.drawRightString(A4[0] - 1.8 * cm, 1.0 * cm, f"Page {d.page}")
        c.restoreState()
    doc.build(fl, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2]); print("wrote", sys.argv[2])
