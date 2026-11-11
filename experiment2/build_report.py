"""从实测 results.json 同步生成中文 PDF 与可编辑 Markdown 报告。

算法与测试不依赖本脚本。重建 PDF 时需 reportlab，字体自动查找 Windows
宋体/黑体；其他平台可通过 --font 和 --bold-font 指定中文 TTF 文件。
"""

import argparse
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


from report_content import create_content

REPOSITORY_URL = "https://github.com/Mengziyang111/contemporary-ai-experiment1/tree/main/experiment2"


def write_markdown(pages, output):
    lines = []
    for page in pages:
        for kind, payload in page:
            if kind == "heading":
                lines += ["# " + payload, ""]
            elif kind == "repo_link":
                lines += [f"GitHub 代码仓库：[{payload}]({payload})", ""]
            elif kind == "subheading":
                lines += ["## " + payload, ""]
            elif kind == "table":
                headers, rows, _ = payload
                lines += ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                lines += ["| " + " | ".join(str(cell).replace("\n", "<br>") for cell in row) + " |" for row in rows]
                lines += [""]
            elif kind == "code":
                lines += ["```text", payload, "```", ""]
            elif kind == "figure":
                figure_type, details, caption = payload
                lines += [caption, ""]
                if figure_type == "grid":
                    lines += ["```text", *[" ".join(str(c).rjust(2) for c in row) for row in details["grid"]], "```", ""]
                elif figure_type == "tree":
                    lines += ["```text", "MAX 5", "  MIN 5: MAX(3,5)=5  MAX(6,9)=9", "  MIN 2: MAX(6,9)=9  MAX(1,2)=2", "```", ""]
                else:
                    lines += [f"- {label}: {value}" for label, value in details] + [""]
            else:
                lines += [payload, ""]
        lines += [""]
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_pdf(pages, output, font_path, bold_font_path):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, Preformatted, KeepTogether
    from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle

    pdfmetrics.registerFont(TTFont("BodyCN", str(font_path), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("HeadCN", str(bold_font_path), subfontIndex=0))
    blue = colors.HexColor("#235b83")
    gray = colors.HexColor("#d9d9d9")
    body = ParagraphStyle("body", fontName="BodyCN", fontSize=10.2, leading=16, spaceAfter=8, wordWrap="CJK", alignment=TA_JUSTIFY)
    heading = ParagraphStyle("heading", fontName="HeadCN", fontSize=18, leading=25, spaceAfter=14, textColor=colors.black, keepWithNext=True)
    sub = ParagraphStyle("sub", fontName="HeadCN", fontSize=11.6, leading=18, spaceBefore=6, spaceAfter=7, keepWithNext=True)
    identity = ParagraphStyle("identity", parent=body, fontName="HeadCN", fontSize=10.5, spaceAfter=18)
    repo_style = ParagraphStyle("repo", parent=body, fontSize=9, leading=13, spaceAfter=8, alignment=0)
    cell = ParagraphStyle("cell", fontName="BodyCN", fontSize=9.3, leading=13, wordWrap="CJK", alignment=TA_CENTER)
    headcell = ParagraphStyle("headcell", parent=cell, fontName="HeadCN")
    caption = ParagraphStyle("caption", parent=body, fontSize=9, leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#444444"), spaceAfter=9)
    code_style = ParagraphStyle("code", fontName="Courier", fontSize=8.25, leading=11.4, spaceAfter=9)

    def text(value):
        return escape(str(value)).replace("\n", "<br/>")

    def figure(kind, payload):
        if kind == "grid":
            grid, path, size = payload["grid"], payload["path"], payload["cell"]
            rows, cols = len(grid), len(grid[0])
            width, height = 487, (rows + 1) * size + 8
            draw = Drawing(width, height)
            left = (width - cols * size) / 2
            route = {tuple(pos): i for i, pos in enumerate(path)}
            for c in range(cols):
                draw.add(String(left + (c + .5) * size, height - 11, str(c), textAnchor="middle", fontName="BodyCN", fontSize=8))
            for r, row in enumerate(grid):
                y = height - (r + 2) * size + 2
                draw.add(String(left - 12, y + size / 2 - 3, str(r), textAnchor="middle", fontName="BodyCN", fontSize=8))
                for c, val in enumerate(row):
                    pos, x = (r, c), left + c * size
                    fill = colors.HexColor("#e0edf6") if pos in route else colors.white
                    if val == "X":
                        fill = colors.HexColor("#454d56")
                    draw.add(Rect(x, y, size, size, fillColor=fill, strokeColor=gray, strokeWidth=.6))
                    label = str(route[pos]) if pos in route and val not in ("S", "E") else str(val)
                    draw.add(String(x + size / 2, y + size / 2 - 3.5, label, textAnchor="middle", fontName="HeadCN" if val != "." else "BodyCN", fontSize=9, fillColor=colors.white if val == "X" else colors.black))
            return draw
        if kind == "tree":
            draw = Drawing(487, 166)
            levels = [[(244, 145, "MAX 5")], [(125, 102, "MIN 5"), (365, 102, "MIN 2")], [(65, 57, "MAX 5"), (185, 57, "MAX 9"), (305, 57, "MAX 9"), (425, 57, "MAX 2")]]
            for level, next_level in zip(levels, levels[1:]):
                for i, (x, y, _) in enumerate(level):
                    for cx, cy, _ in next_level[i * 2:i * 2 + 2]:
                        draw.add(Line(x, y - 7, cx, cy + 12, strokeColor=gray))
            leaves = [3, 5, 6, 9, 6, 9, 1, 2]
            for i, (x, y, _) in enumerate(levels[-1]):
                for j in range(2):
                    cx = x - 21 + j * 42
                    draw.add(Line(x, y - 7, cx, 21, strokeColor=gray))
                    draw.add(String(cx, 8, str(leaves[i * 2 + j]), textAnchor="middle", fontName="HeadCN", fontSize=10))
            for level in levels:
                for x, y, label in level:
                    draw.add(String(x, y, label, textAnchor="middle", fontName="HeadCN", fontSize=10))
            return draw
        draw = Drawing(487, 139)
        maxvalue = max(v for _, v in payload)
        for i, (label, value) in enumerate(payload):
            y = 109 - i * 29
            draw.add(String(88, y + 4, label, textAnchor="end", fontName="BodyCN", fontSize=9))
            length = 300 * value / maxvalue
            draw.add(Rect(99, y, length, 16, strokeColor=None, fillColor=blue if i == len(payload) - 1 else colors.HexColor("#98b5c9")))
            draw.add(String(108 + length, y + 4, str(value), fontName="HeadCN", fontSize=9))
        return draw

    story = []
    for index, page in enumerate(pages):
        if index:
            story.append(PageBreak())
        for kind, payload in page:
            if kind in {"paragraph", "heading", "subheading", "identity"}:
                story.append(Paragraph(text(payload), {"paragraph": body, "heading": heading, "subheading": sub, "identity": identity}[kind]))
            elif kind == "repo_link":
                url = escape(payload, quote=True)
                story.append(Paragraph(f'GitHub 代码仓库：<link href="{url}" color="#235b83"><u>{url}</u></link>', repo_style))
            elif kind == "code":
                story.append(Preformatted(payload, code_style))
            elif kind == "table":
                headers, rows, widths = payload
                values = [[Paragraph(text(c), headcell) for c in headers]] + [[Paragraph(text(c), cell) for c in row] for row in rows]
                t = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
                t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6edf3")), ("GRID", (0, 0), (-1, -1), .5, gray), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5)]))
                story.extend([t, Spacer(1, 10)])
            elif kind == "figure":
                figure_kind, details, title = payload
                story.append(KeepTogether([figure(figure_kind, details), Paragraph(text(title), caption)]))

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setFont("BodyCN", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(54, 815, "当代人工智能实验课  搜索算法")
        canvas.drawRightString(541, 28, str(document.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=54, leftMargin=54, topMargin=50, bottomMargin=46, title="实验二 搜索算法实验报告", author="孟子阳 10245501476")
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    if doc.page > 5:
        raise RuntimeError(f"报告实际为 {doc.page} 页，超过课程 5 页上限，请调整内容后重新生成。")
    return doc.page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/simsun.ttc"))
    parser.add_argument("--bold-font", type=Path, default=Path("C:/Windows/Fonts/simhei.ttf"))
    parser.add_argument("--repo-url", default=REPOSITORY_URL, help="报告首页的 GitHub 代码链接")
    args = parser.parse_args()
    data = json.loads((ROOT / "results/results.json").read_text(encoding="utf-8"))
    output = ROOT / "report"
    output.mkdir(exist_ok=True)
    pages = create_content(data, args.repo_url)
    write_markdown(pages, output / "实验二报告.md")
    page_count = write_pdf(pages, output / "实验二报告.pdf", args.font, args.bold_font)
    print(f"已生成 {output / '实验二报告.pdf'}，共 {page_count} 页。")


if __name__ == "__main__":
    main()
