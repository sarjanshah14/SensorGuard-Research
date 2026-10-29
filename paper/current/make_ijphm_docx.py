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


def plain_math(m):
    t = m.group(1)
    if t == "z":
        return r"\emph{z}"
    if re.fullmatch(r"[0-9.,\-+{} ]|\\pm|\\,|\\%", t) or re.fullmatch(r"(?:[0-9.,\-+{} ]|\\pm|\\,|\\%)+", t):
        return t.replace("\\pm", " ± ").replace("{,}", ",").replace("\\,", " ").replace("{", "").replace("}", "").replace("-", "−")
    return m.group(0)


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


def postprocess(xml, tpl_xml):
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
    one_col = f'<w:sectPr><w:type w:val="continuous"/>{pg}<w:cols w:space="720"/></w:sectPr>'
    kw = re.search(r'<w:p>(?:(?!<w:p>).)*?Keywords:.*?</w:p>', xml, re.S).group(0)
    licence = para("phmnormal", run(LICENCE, italic=True, size=16), one_col)
    xml = xml.replace(kw, kw + licence, 1)
    two_col = f'<w:sectPr><w:type w:val="continuous"/>{pg}<w:cols w:num="2" w:space="360"/></w:sectPr>'
    marker = re.search(r'<w:p>(?:(?!<w:p>).)*?FLOATSMARKER.*?</w:p>', xml, re.S).group(0)
    xml = xml.replace(marker, f'<w:p><w:pPr>{two_col}</w:pPr></w:p>', 1)   # body ends two-column
    j = xml.rfind("<w:sectPr")  # final section: tables and figures, one column
    xml = xml[:j] + f'<w:sectPr><w:type w:val="nextPage"/>{pg}<w:cols w:space="720"/></w:sectPr></w:body></w:document>'

    return xml


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
        with zipfile.ZipFile(raw) as zin, zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = postprocess(data.decode(), tpl_xml).encode()
                zout.writestr(item, data)
    print("wrote", OUT, "| citations resolved:", len(lab))


if __name__ == "__main__":
    main()
