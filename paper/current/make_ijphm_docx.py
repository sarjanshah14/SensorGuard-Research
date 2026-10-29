"""Convert paper2_revised.tex to an IJPHM Word manuscript.

Run: python paper/current/make_ijphm_docx.py   -> paper/current/Shah_IJPHM_Protocol_Sensitivity_Industrial_Anomaly_Detection.docx

1. Rewrite the LaTeX so pandoc can handle it: natbib citations become APA text
   from the \\bibitem labels; simple numeric maths becomes plain text.
2. pandoc with the IJPHM template as reference doc (its styles and numbering
   definitions are carried over).
3. Post-process word/document.xml: map pandoc's styles onto the template's
   phm_* styles (headings then auto-number from the template), rebuild the title
   block (title, authors, affiliation, e-mails, licence note), and make the body
   two-column after the one-column title/abstract section.
"""

import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
TEX = HERE / "paper2_revised.tex"
TEMPLATE = HERE.parent / "templates" / "ijphm" / "ijPHM-FullPaperTemplate.docx"
OUT = HERE / "Shah_IJPHM_Protocol_Sensitivity_Industrial_Anomaly_Detection.docx"
UNNUMBERED = {"References", "Data and Code Availability", "Disclosure of Interests"}
LICENCE = ("Sarjan Shah et al. This is an open-access article distributed under the terms of the Creative "
           "Commons Attribution 3.0 United States License, which permits unrestricted use, distribution, and "
           "reproduction in any medium, provided the original author and source are credited.")


def labels(src):
    out = {}
    for lab, key in re.findall(r"\\bibitem\[(.+?)\]\{(.+?)\}", src):
        m = re.match(r"(.*)\((\d{4}[a-z]?)\)$", lab)
        out[key] = (m.group(1).replace("~", " ").replace("\\&", "&").strip(), m.group(2))
    return out


# Inline maths becomes plain Unicode text: Word equation objects inside table
# cells render blank in some viewers (reviewer saw empty nu, k, lr and arrows).
MATH = [(r"\max_j |x_j-\mu_j|/\sigma_j", "maxⱼ |xⱼ − μⱼ| / σⱼ"), (r"\bar D", "D̄"), ("10^{-3}", "10⁻³"),
        ("{=}", " = "), ("{,}", ","), (r"\pm", " ± "), (r"\to", " → "), (r"\le", "≤ "), (r"\approx", "≈"),
        (r"\in", " ∈ "), (r"\{", "{"), (r"\}", "}"), (r"\,", " "), (r"\nu", "ν"), (r"\pi", "π"),
        (r"\tau", "τ"), ("-", "−")]


def plain_math(m):
    t = m.group(1).replace(r"\%", "PCT")
    for a, b in MATH:
        t = t.replace(a, b)
    if "\\" in t or "^" in t or "_" in t:
        return m.group(0)  # real formula: leave to pandoc
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"(?<![A-Za-z])([A-Za-zπντ])(?![A-Za-z])", r"\\emph{\1}", t).replace("PCT", r"\%")


def rewrite(src, lab):
    src = re.sub(r"\\citep\{(.+?)\}", lambda m: "(" + "; ".join(f"{lab[k][0]}, {lab[k][1]}" for k in m.group(1).split(",")) + ")", src)
    src = re.sub(r"\\citet\{(.+?)\}", lambda m: "; ".join(f"{lab[k][0]} ({lab[k][1]})" for k in m.group(1).split(",")), src)
    src = re.sub(r"\\bibitem\[.+?\]\{.+?\}\s*", r"\n\n", src)
    src = src.replace("\\begin{thebibliography}{99}", "\\section*{References}").replace("\\end{thebibliography}", "")
    src = re.sub(r"(?<!\\)\$([^$]+?)\$", plain_math, src)
    src = re.sub(r"\\let\\thefootnote.*?\n", "", src)           # licence note is rebuilt in the title block
    src = re.sub(r"\\author\{.*?\\date\{\}", r"\\author{AUTHORS}\n\\date{}", src, flags=re.S)
    # IJPHM allows tables and figures after the references in one column; wide
    # tables otherwise break across the two body columns
    floats = re.findall(r"\\begin\{(?:table|figure)\}.*?\\end\{(?:table|figure)\}", src, re.S)
    for f in floats:
        src = src.replace(f, "", 1)
    src = src.replace("\\end{document}", "\n\nFLOATSMARKER\n\n" + "\n\n".join(floats) + "\n\n\\end{document}")
    return src


def para(style, runs, extra=""):
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{extra}</w:pPr>{runs}</w:p>'


def run(text, sup=False, italic=False, size=None):
    rpr = ("<w:vertAlign w:val=\"superscript\"/>" if sup else "") + ("<w:i/>" if italic else "") + (f'<w:sz w:val="{size}"/>' if size else "")
    text = text.replace("&", "&amp;").replace("<", "&lt;")
    return f'<w:r>{"<w:rPr>" + rpr + "</w:rPr>" if rpr else ""}<w:t xml:space="preserve">{text}</w:t></w:r>'


def title_block(title_xml):
    return (title_xml.replace('w:val="Title"', 'w:val="phmtitle"')
            + para("phmauthorlist", run("Sarjan Shah") + run("1", sup=True) + run(" and Abhi Shah") + run("1", sup=True))
            + para("phmaffiliation", run("1", sup=True) + run("Department of Information Technology, Lok Jagruti University, Vadodara, Gujarat, India"))
            + para("phmemail", run("shahsarjan968@gmail.com"))
            + para("phmemail", run("abhi.shah@ljku.edu.in")))


def postprocess(xml, tpl_xml, refs=""):
    """refs: header/footer references (template's small-caps journal header from page 2,
    page-number footer, first-page journal/ISSN footer)."""
    xml = re.sub(r'"\s+/>', '"/>', xml)  # pandoc writes <w:x a="b" />
    # IJPHM: table captions go below the table
    xml = re.sub(r'(<w:p>(?:(?!<w:p>).)*?<w:pStyle w:val="TableCaption"/>.*?</w:p>)\s*(<w:tbl>.*?</w:tbl>)',
                 r"\2\1", xml, flags=re.S)
    # title block: replace pandoc's Title + Author paragraphs
    t = re.search(r'<w:p>(?:(?!<w:p>).)*?<w:pStyle w:val="Title"/>.*?</w:p>', xml, re.S).group(0)
    authors = re.findall(r'<w:p>(?:(?!<w:p>).)*?<w:pStyle w:val="Author"/>.*?</w:p>', xml, re.S)
    xml = xml.replace(t, title_block(t), 1)
    for a in authors:
        xml = xml.replace(a, "", 1)

    # unnumbered headings
    def head1(m):
        p = m.group(0)
        txt = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", p))
        return p.replace('w:val="Heading1"', 'w:val="phmheadingnonumber"' if txt.strip() in UNNUMBERED else 'w:val="phmheading1"')
    xml = re.sub(r'<w:p>(?:(?!<w:p>).)*?<w:pStyle w:val="Heading1"/>.*?</w:p>', head1, xml, flags=re.S)
    # strip pandoc's literal heading numbers if any remained
    for a, b in {"Heading2": "phmheading2", "Heading3": "phmheading3", "Heading4": "phmheading4",
                 "AbstractTitle": "phmheadingabstract", "Abstract": "phmbodytext",
                 "FirstParagraph": "phmbodytext", "BodyText": "phmbodytext",
                 "ImageCaption": "phmcaption", "TableCaption": "phmcaption", "CaptionedFigure": "phmfigures"}.items():
        xml = xml.replace(f'w:val="{a}"', f'w:val="{b}"')

    # references after the References heading -> phm_reference
    i = xml.find(">References<")
    if i > 0:
        head, tail = xml[:i], xml[i:]
        tail = re.sub(r'(<w:p>(?:(?!<w:p>).)*?<w:pStyle w:val=")phmbodytext(")', r"\1phmreference\2", tail, flags=re.S)
        xml = head + tail

    # section layout from the template: one-column title/abstract, then two columns
    pg = re.search(r"<w:pgSz [^>]*/>", tpl_xml).group(0) + re.search(r"<w:pgMar [^>]*/>", tpl_xml).group(0)
    one_col = f'<w:sectPr>{refs}<w:type w:val="continuous"/>{pg}<w:cols w:space="720"/><w:titlePg/></w:sectPr>'
    kw = re.search(r'<w:p>(?:(?!<w:p>).)*?Keywords:.*?</w:p>', xml, re.S).group(0)
    licence = para("phmnormal", run(LICENCE, italic=True, size=16), one_col)
    xml = xml.replace(kw, kw + licence, 1)
    two_col = f'<w:sectPr>{refs}<w:type w:val="continuous"/>{pg}<w:cols w:num="2" w:space="360"/><w:titlePg/></w:sectPr>'
    marker = re.search(r'<w:p>(?:(?!<w:p>).)*?FLOATSMARKER.*?</w:p>', xml, re.S).group(0)
    xml = xml.replace(marker, f'<w:p><w:pPr>{two_col}</w:pPr></w:p>', 1)   # body ends two-column
    j = xml.rfind("<w:sectPr")  # final section: tables and figures, one column
    xml = xml[:j] + f'<w:sectPr>{refs}<w:type w:val="nextPage"/>{pg}<w:cols w:space="720"/></w:sectPr></w:body></w:document>'

    # caption labels: pandoc numbers \ref targets but writes captions without "Table n."
    n = {"Table": 0, "Figure": 0}
    def label(m):
        kind = "Table" if m.group(1) else "Figure"
        n[kind] += 1
        return m.group(0).replace("</w:pPr>", "<w:keepLines/></w:pPr>" + run(f"{kind} {n[kind]}. "), 1)
    xml = re.sub(r'(</w:tbl>\s*)?<w:p><w:pPr><w:pStyle w:val="phmcaption"/>(?:(?!</w:pPr>).)*</w:pPr>', label, xml, flags=re.S)

    w = int(re.search(r'<w:pgSz [^>]*w:w="(\d+)"', tpl_xml).group(1))
    w -= sum(int(re.search(rf'<w:pgMar [^>]*w:{s}="(\d+)"', tpl_xml).group(1)) for s in ("left", "right"))
    return re.sub(r"<w:tbl>.*?</w:tbl>", lambda m: fix_table(m.group(0), w), xml, flags=re.S)


def add_missing_styles(styles, donor):
    """The IJPHM template lacks pandoc's styles (Table, Compact, ...); without
    them LibreOffice-based renderers dissolve every table into loose lines."""
    found = lambda s: {m.group(1): m.group(0) for m in re.finditer(r'<w:style [^>]*w:styleId="([^"]+)".*?</w:style>', s, re.S)}
    have, extra = found(styles), found(donor)
    return styles.replace("</w:styles>", "".join(v for k, v in extra.items() if k not in have) + "</w:styles>")


def _small(rpr_inner):
    # w:sz must precede w:u / w:vertAlign in the rPr schema order
    k = min([rpr_inner.find(t) for t in ("<w:u ", "<w:u/>", "<w:vertAlign") if t in rpr_inner] or [len(rpr_inner)])
    return rpr_inner[:k] + '<w:sz w:val="16"/><w:szCs w:val="16"/>' + rpr_inner[k:]


def fix_table(tbl, width):
    """pandoc leaves tables auto-width with equal columns, which some renderers
    squeeze into narrow strips. Give each a full-width fixed layout, columns
    weighted by their longest cell, 8 pt text, rules above and below, rows that
    do not split across pages."""
    rows = re.findall(r"<w:tr>.*?</w:tr>", tbl, re.S)
    lens = {}
    for r in rows:
        cells = re.findall(r"<w:tc>.*?</w:tc>", r, re.S)
        if len(cells) != tbl.count("<w:gridCol "):
            continue  # skip spanned header rows
        for i, c in enumerate(cells):
            lens[i] = max(lens.get(i, 0), len("".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", c))))
    weights = [min(max(lens.get(i, 9), 9), 45) for i in range(tbl.count("<w:gridCol "))]
    grid = "".join(f'<w:gridCol w:w="{round(width * x / sum(weights))}"/>' for x in weights)
    tbl = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", f"<w:tblGrid>{grid}</w:tblGrid>", tbl, flags=re.S)
    rule = '<w:{} w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
    tbl = re.sub(r'<w:tblW [^>]*/>', f'<w:tblW w:type="dxa" w:w="{width}"/><w:jc w:val="center"/><w:tblBorders>'
                 + rule.format("top") + rule.format("bottom") + rule.format("insideH").replace('w:sz="6"', 'w:sz="2"')
                 + '</w:tblBorders><w:tblLayout w:type="fixed"/>', tbl, count=1)
    tbl = re.sub(r"<w:tr>(?!<w:trPr>)", "<w:tr><w:trPr><w:cantSplit/></w:trPr>", tbl)
    tbl = tbl.replace("<w:trPr><w:tblHeader", "<w:trPr><w:cantSplit/><w:tblHeader")
    tbl = tbl.replace('<w:pStyle w:val="Compact"/>', '<w:pStyle w:val="Compact"/><w:keepNext/>')  # table stays with its caption
    tbl = re.sub(r"<w:rPr>(.*?)</w:rPr>", lambda m: f"<w:rPr>{_small(m.group(1))}</w:rPr>", tbl, flags=re.S)
    tbl = re.sub(r"<w:r>(?!<w:rPr>)", '<w:r><w:rPr><w:sz w:val="16"/><w:szCs w:val="16"/></w:rPr>', tbl)
    return tbl


def main():
    src = TEX.read_text()
    lab = labels(src)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "in.tex").write_text(rewrite(src, lab))
        raw = td / "raw.docx"
        subprocess.run(["pandoc", str(td / "in.tex"), "-o", str(raw), f"--reference-doc={TEMPLATE}",
                        f"--resource-path={HERE / 'overleaf_paper2' / 'figures'}:{HERE.parent.parent / 'results' / 'revised' / 'figures'}"],
                       check=True)
        with zipfile.ZipFile(TEMPLATE) as z:
            tpl_xml = z.read("word/document.xml").decode()
        (td / "ref.docx").write_bytes(subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                                                     capture_output=True, check=True).stdout)
        with zipfile.ZipFile(td / "ref.docx") as z:
            pandoc_styles = z.read("word/styles.xml").decode()
        with zipfile.ZipFile(raw) as z:
            rels = dict((t, i) for i, t in re.findall(r'Id="(\w+)" Target="((?:header|footer)\d\.xml)"', z.read("word/_rels/document.xml.rels").decode()))
        refs = (f'<w:headerReference w:type="default" r:id="{rels["header1.xml"]}"/>'
                f'<w:footerReference w:type="default" r:id="{rels["footer1.xml"]}"/>'
                f'<w:footerReference w:type="first" r:id="{rels["footer2.xml"]}"/>')
        with zipfile.ZipFile(raw) as zin, zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = postprocess(data.decode(), tpl_xml, refs).encode()
                if item.filename == "word/footer2.xml":
                    data = data.replace(b"2012", b"2026")  # year; "***" is the paper number, set after acceptance
                if item.filename == "word/styles.xml":
                    data = add_missing_styles(data.decode(), pandoc_styles).encode()
                zout.writestr(item, data)
    print("wrote", OUT, "| citations resolved:", len(lab))


if __name__ == "__main__":
    main()
