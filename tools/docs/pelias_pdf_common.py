# Copyright (c) 2026 John Lipka and the Pelias contributors. Apache-2.0.
# provenance: original
# Layout helpers for the Pelias overview PDF (tools/docs/make_project_overview.py). Needs reportlab and the DejaVu fonts.
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

D = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", D + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", D + "DejaVuSans-Oblique.ttf"))
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DV-B"); addMapping("DV", 0, 1, "DV-I"); addMapping("DV", 1, 1, "DV-B")
INK = colors.HexColor("#22262D"); ACC = colors.HexColor("#1F5F7A"); GRID = colors.HexColor("#A9AFB7"); ALT = colors.HexColor("#F2F5F7")
NOTE = colors.HexColor("#FFF5DE"); GOOD = colors.HexColor("#E4F4E2"); BAD = colors.HexColor("#FBE6E6")
ss = getSampleStyleSheet()


def style(name, size, lead, **kw):
    kw.setdefault("textColor", INK)
    return ParagraphStyle(name, parent=ss["Normal"], fontName="DV", fontSize=size, leading=lead, **kw)


TITLE = style("t", 21, 26, spaceAfter=4); TITLE.fontName = "DV-B"
SUB = style("sub", 11, 15, textColor=ACC, spaceAfter=10)
H1 = style("h1", 14, 18, spaceBefore=10, spaceAfter=4, textColor=ACC); H1.fontName = "DV-B"
H2 = style("h2", 11, 14, spaceBefore=6, spaceAfter=2, textColor=INK); H2.fontName = "DV-B"
B = style("b", 9.6, 13.4, spaceAfter=5)
BL = style("bl", 9.6, 13.2, leftIndent=14, bulletIndent=4, spaceAfter=2)
C = style("c", 7.9, 9.9)
SM = style("sm", 7.6, 9.6, textColor=colors.HexColor("#4A5058"))


def P(t, s=B):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(x, BL, bulletText="•") for x in items]


def note(text, color=NOTE, width=7.5):
    t = Table([[Paragraph(text, C)]], colWidths=[width * inch])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color), ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                           ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def table(header, rows, widths, shade=None):
    data = [[Paragraph('<font color="white"><b>%s</b></font>' % h, C) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), C) for c in r])
    cmd = [("GRID", (0, 0), (-1, -1), 0.4, GRID), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BACKGROUND", (0, 0), (-1, 0), INK),
           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, shade or ALT])]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1)
    t.setStyle(TableStyle(cmd))
    return t


def build(out, title, story, footer_text):
    doc = SimpleDocTemplate(out, pagesize=letter, leftMargin=0.5 * inch, rightMargin=0.5 * inch, topMargin=0.55 * inch,
                            bottomMargin=0.6 * inch, title=title, author="John Lipka and the Pelias contributors")

    def footer(c, d):
        c.saveState()
        c.setFont("DV", 7)
        c.setFillColor(GRID)
        c.drawString(0.5 * inch, 0.35 * inch, footer_text)
        c.drawRightString(letter[0] - 0.5 * inch, 0.35 * inch, "page %d" % d.page)
        c.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
