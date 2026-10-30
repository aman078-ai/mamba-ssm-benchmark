"""Build the paper from one Markdown source.

    python docs/paper/build.py            # paper.html + paper.tex (+ paper.pdf if Microsoft Edge is installed)

Source: docs/paper/paper.md, a small Markdown subset:
  front matter     ---\\n title: ...\\n author: ...\\n affiliation: ...\\n email: ...\\n date: ...\\n ---
  headings         # Title {#sec:label}   (## and ### for subsections; "# Abstract" and "# Appendix" are special)
  math             $inline$ and display blocks between lines "$$" and "$$ {#eq:label}"
  citations        [@key] or [@a; @b] (parenthetical), @key (textual) -- keys from refs.bib
  cross-refs       @sec:x, @fig:x, @tab:x, @eq:x
  figures          ![caption](relative/path.png){#fig:label}
  tables           "Table: caption {#tab:label}" line followed by a pipe table
  emphasis         **bold**, *italic*, `code`; lists with "- " or "1. "
Every number in the paper comes from results/ via the analysis scripts; this file only typesets.
"""

import html
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


# ---------------------------------------------------------------------------------------------------------- parsing
def parse_bib(path):
    entries = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", open(path, encoding="utf-8").read(), re.S):
        fields = dict((k.lower(), v.strip()) for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\},?\s*\n", m.group(3) + "\n", re.S))
        entries[m.group(2).strip()] = {"type": m.group(1).lower(), **fields}
    return entries


def bib_plain(s):
    s = re.sub(r"\\'\{?e\}?", "é", s)
    s = re.sub(r"\\url\{(.*?)\}", r"\1", s)
    return s.replace("{", "").replace("}", "").replace("--", "–")


def surnames(entry):
    names = [n.strip() for n in bib_plain(entry["author"]).split(" and ")]
    last = [n.split(",")[0] if "," in n else n.split()[-1] for n in names]
    if len(last) == 1:
        return last[0]
    if len(last) == 2:
        return f"{last[0]} and {last[1]}"
    return f"{last[0]} et al."


def parse(md):
    meta, body = {}, md
    if md.startswith("---"):
        head, body = md[3:].split("\n---", 1)
        for line in head.strip().splitlines():
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip().strip('"')
    blocks, lines, i = [], body.strip("\n").splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text, label = split_label(line[level:].strip())
            blocks.append(("h", level, text, label))
            i += 1
        elif line.strip() == "$$":
            j = i + 1
            while not lines[j].startswith("$$"):
                j += 1
            label = split_label(lines[j][2:].strip())[1] if lines[j][2:].strip() else None
            blocks.append(("eq", "\n".join(lines[i + 1:j]), label))
            i = j + 1
        elif line.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*?)\)\{#(fig:[\w-]+)\}", line)
            blocks.append(("fig", m.group(1), m.group(2), m.group(3)))
            i += 1
        elif line.startswith("Table:"):
            caption, label = split_label(line[6:].strip())
            j = i + 1
            rows = []
            while j < len(lines) and lines[j].startswith("|"):
                if not re.match(r"^\|[\s:|-]+\|$", lines[j]):
                    rows.append([c.strip() for c in lines[j].strip("|").split("|")])
                j += 1
            blocks.append(("table", caption, label, rows))
            i = j
        elif re.match(r"^(- |\d+\. )", line):
            ordered = bool(re.match(r"^\d+\. ", line))
            items = []
            while i < len(lines) and (re.match(r"^(- |\d+\. )", lines[i]) or lines[i].startswith("  ")):
                if re.match(r"^(- |\d+\. )", lines[i]):
                    items.append(re.sub(r"^(- |\d+\. )", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            blocks.append(("list", ordered, items))
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\$\$|!\[|Table:|\||- |\d+\. )", lines[i]):
                para.append(lines[i].strip())
                i += 1
            blocks.append(("p", " ".join(para)))
    return meta, blocks


def split_label(text):
    m = re.search(r"\s*\{#([\w:-]+)\}\s*$", text)
    return (text[:m.start()], m.group(1)) if m else (text, None)


def number(blocks):
    """Numbers for sections, equations, figures, tables (and appendix letters)."""
    nums, counters, appendix, sec = {}, {"eq": 0, "fig": 0, "tab": 0}, False, [0, 0, 0]
    for b in blocks:
        if b[0] == "h":
            if b[2] in ("Abstract", "References"):
                continue
            if b[2] == "Appendix":
                appendix, sec = True, [0, 0, 0]
                continue
            lvl = b[1] - 1 - (1 if appendix else 0)        # appendix: "##" headings are lettered A, B, ...
            sec[lvl] += 1
            sec[lvl + 1:] = [0] * (2 - lvl)
            top = chr(64 + sec[0]) if appendix else str(sec[0])
            n = ".".join([top] + [str(x) for x in sec[1:lvl + 1]])
            if b[3]:
                nums[b[3]] = n
        elif b[0] in ("eq", "fig", "table"):
            kind = {"eq": "eq", "fig": "fig", "table": "tab"}[b[0]]
            label = b[2] if b[0] == "eq" else b[3] if b[0] == "fig" else b[2]
            counters[kind] += 1
            if label:
                nums[label] = str(counters[kind])
    return nums


# ------------------------------------------------------------------------------------------------------------- HTML
REF_WORDS = {"sec": "Section", "fig": "Figure", "tab": "Table", "eq": "Eq."}


def inline_html(s, bib, nums, cited):
    maths = []
    s = re.sub(r"\$(.+?)\$", lambda m: maths.append(m.group(1)) or f"\x00{len(maths) - 1}\x00", s)
    s = html.escape(s, quote=False)

    def cite(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        cited.extend(k for k in keys if k not in cited)
        return "(" + "; ".join(f'<a href="#ref-{k}">{surnames(bib[k])}, {bib[k]["year"]}</a>' for k in keys) + ")"
    s = re.sub(r"\[(@[\w-]+(?:;\s*@[\w-]+)*)\]", cite, s)
    def xref(m):
        n = nums[m.group(1) + ":" + m.group(2)]
        word = "Appendix" if m.group(1) == "sec" and n[0].isalpha() else REF_WORDS[m.group(1)]
        n = f"({n})" if m.group(1) == "eq" else n
        return f'<a href="#{m.group(1)}:{m.group(2)}">{word}&nbsp;{n}</a>'
    s = re.sub(r"@(sec|fig|tab|eq):([\w-]+)", xref, s)

    def tcite(m):
        k = m.group(1)
        if k not in bib:
            return m.group(0)
        cited.append(k) if k not in cited else None
        return f'<a href="#ref-{k}">{surnames(bib[k])}</a> ({bib[k]["year"]})'
    s = re.sub(r"(?<![\w/])@([\w-]+)", tcite, s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: "\\(" + html.escape(maths[int(m.group(1))], quote=False) + "\\)", s)
    return s


def to_html(meta, blocks, bib, nums):
    cited, out, in_appendix = [], [], False
    for b in blocks:
        if b[0] == "h":
            if b[2] == "Appendix":
                in_appendix = True
                out.append('<h1 class="appendix-title">Appendix</h1>')
                continue
            n = nums.get(b[3], "") if b[3] else ""
            if not n and b[2] not in ("Abstract", "References"):
                n = ""
            anchor = f' id="{b[3]}"' if b[3] else ""
            cls = ' class="abstract-title"' if b[2] == "Abstract" else ""
            out.append(f"<h{b[1]}{anchor}{cls}>{(n + ' ') if n else ''}{inline_html(b[2], bib, nums, cited)}</h{b[1]}>")
        elif b[0] == "p":
            out.append(f"<p>{inline_html(b[1], bib, nums, cited)}</p>")
        elif b[0] == "list":
            tag = "ol" if b[1] else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline_html(x, bib, nums, cited)}</li>" for x in b[2]) + f"</{tag}>")
        elif b[0] == "eq":
            tag = f'<span class="eqno">({nums[b[2]]})</span>' if b[2] else ""
            anchor = f' id="{b[2]}"' if b[2] else ""
            out.append(f'<div class="eq"{anchor}>\\[{html.escape(b[1], quote=False)}\\]{tag}</div>')
        elif b[0] == "fig":
            out.append(f'<figure id="{b[3]}"><img src="{b[2]}" alt=""><figcaption><b>Figure {nums[b[3]]}.</b> '
                       f"{inline_html(b[1], bib, nums, cited)}</figcaption></figure>")
        elif b[0] == "table":
            head, *rows = b[3]
            t = "<thead><tr>" + "".join(f"<th>{inline_html(c, bib, nums, cited)}</th>" for c in head) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join(f"<td>{inline_html(c, bib, nums, cited)}</td>" for c in r) + "</tr>" for r in rows)
            out.append(f'<div class="table" id="{b[2]}"><div class="caption"><b>Table {nums[b[2]]}.</b> '
                       f'{inline_html(b[1], bib, nums, cited)}</div><table>{t}</tbody></table></div>')
    refs = sorted(cited, key=lambda k: (surnames(bib[k]).lower(), bib[k]["year"]))
    ref_html = "".join(f'<li id="ref-{k}">{html_ref(bib[k])}</li>' for k in refs)
    body = "\n".join(out)
    # references go before the appendix
    marker = '<h1 class="appendix-title">'
    refs_block = f'<h1 id="references">References</h1><ol class="refs">{ref_html}</ol>'
    body = body.replace(marker, refs_block + marker, 1) if marker in body else body + refs_block
    title = html.escape(meta["title"])
    authors = f'{html.escape(meta.get("author", ""))}<br><span class="aff">{html.escape(meta.get("affiliation", ""))}' \
              f'{" · " + html.escape(meta["email"]) if meta.get("email") else ""}</span>'
    return TEMPLATE.replace("$TITLE", title).replace("$AUTHORS", authors) \
        .replace("$DATE", html.escape(meta.get("date", ""))).replace("$BODY", body)


def html_ref(e):
    authors = bib_plain(e["author"]).split(" and ")
    authors = [a.split(",")[1].strip() + " " + a.split(",")[0].strip() if "," in a else a for a in authors]
    a = ", ".join(authors[:-1]) + (", and " if len(authors) > 2 else " and ") + authors[-1] if len(authors) > 1 else authors[0]
    venue = e.get("booktitle") or e.get("journal") or e.get("howpublished") or ""
    extra = "".join(f", {bib_plain(e[k])}" for k in ("volume", "pages") if k in e)
    note = f". {bib_plain(e['note'])}" if "note" in e else ""
    title = bib_plain(e["title"])
    title += "" if title[-1] in "?!." else "."
    return f"{html.escape(a)}. {html.escape(title)} <i>{html.escape(bib_plain(venue))}</i>{extra}, {e['year']}{html.escape(note)}."


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>$TITLE</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{delimiters:[{left:'\\[',right:'\\]',display:true},{left:'\\(',right:'\\)',display:false}]});document.body.dataset.ready=1"></script>
<style>
  @page { size: A4; margin: 22mm 20mm 22mm 20mm; }
  :root { --ink:#141414; --muted:#555; --rule:#bbb; }
  html { background:#fff; }
  body { font-family: "Cambria", "Georgia", serif; font-size: 10.5pt; line-height: 1.42; color: var(--ink);
         max-width: 170mm; margin: 0 auto; padding: 16px; background:#fff; }
  h1.title { font-size: 17pt; text-align: center; margin: 0 0 6pt; line-height: 1.25; }
  .authors { text-align: center; font-size: 11pt; margin-bottom: 2pt; }
  .aff { font-size: 9.5pt; color: var(--muted); }
  .date { text-align: center; font-size: 9pt; color: var(--muted); margin-bottom: 14pt; }
  h1 { font-size: 12.5pt; margin: 16pt 0 5pt; }
  h2 { font-size: 11pt; margin: 11pt 0 4pt; }
  h3 { font-size: 10.5pt; font-style: italic; margin: 9pt 0 3pt; }
  h1.abstract-title { text-align: center; font-size: 10.5pt; margin-top: 4pt; }
  h1.abstract-title + p { margin: 0 8mm 10pt; font-size: 9.8pt; text-align: justify; }
  p { margin: 0 0 6pt; text-align: justify; hyphens: auto; }
  ul, ol { margin: 0 0 6pt; padding-left: 18pt; }
  li { margin-bottom: 2pt; }
  a { color: #1d4f91; text-decoration: none; }
  code { font-family: Consolas, monospace; font-size: 9pt; }
  .eq { position: relative; margin: 6pt 0; text-align: center; }
  .eqno { position: absolute; right: 0; top: 50%; transform: translateY(-50%); }
  figure { margin: 10pt 0; break-inside: avoid; text-align: center; }
  figure img { max-width: 100%; }
  figcaption, .caption { font-size: 9.2pt; text-align: left; margin-top: 4pt; }
  .table { margin: 10pt 0; break-inside: avoid; }
  .table .caption { margin: 0 0 4pt; }
  table { border-collapse: collapse; width: 100%; font-size: 8.2pt; border-top: 1.2px solid #000; border-bottom: 1.2px solid #000; }
  th { border-bottom: 0.8px solid #000; font-weight: bold; }
  th, td { padding: 2pt 4pt; text-align: left; vertical-align: top; }
  td + td { white-space: nowrap; }
  ol.refs { font-size: 9pt; padding-left: 16pt; }
  ol.refs li { margin-bottom: 3pt; }
  h1.appendix-title { break-before: page; }
</style></head><body>
<h1 class="title">$TITLE</h1>
<div class="authors">$AUTHORS</div>
<div class="date">$DATE</div>
$BODY
</body></html>
"""


# ------------------------------------------------------------------------------------------------------------ LaTeX
TEX_CHARS = {"×": r"$\times$", "–": "--", "—": "---", "≈": r"$\approx$", "±": r"$\pm$", "≤": r"$\leq$",
             "≥": r"$\geq$", "−": r"$-$", "→": r"$\rightarrow$", "’": "'", "‘": "`", "“": "``", "”": "''",
             "Δ": r"$\Delta$", "∼": r"$\sim$", "·": r"$\cdot$", "é": r"\'e", "…": r"\ldots{}", " ": "~"}


def tex_escape(s):
    s = s.replace("\\", r"\textbackslash{}")
    for c in "&%#_{}":
        s = s.replace(c, "\\" + c)
    s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    for k, v in TEX_CHARS.items():
        s = s.replace(k, v)
    return s


def inline_tex(s):
    keep = []

    def stash(x):
        keep.append(x)
        return f"\x00{len(keep) - 1}\x00"
    s = re.sub(r"\$(.+?)\$", lambda m: stash(f"${m.group(1)}$"), s)
    s = re.sub(r"`(.+?)`", lambda m: stash(r"\texttt{" + tex_escape(m.group(1)) + "}"), s)
    s = re.sub(r"\[(@[\w-]+(?:;\s*@[\w-]+)*)\]",
               lambda m: stash(r"\citep{" + ",".join(k.strip().lstrip("@") for k in m.group(1).split(";")) + "}"), s)
    s = re.sub(r"@(sec|fig|tab|eq):([\w-]+)", lambda m: stash(
        {"sec": r"\S", "fig": "Figure~", "tab": "Table~", "eq": "Eq.~"}[m.group(1)]
        + (r"\eqref{" if m.group(1) == "eq" else r"\ref{") + f"{m.group(1)}:{m.group(2)}" + "}"), s)
    s = re.sub(r"(?<![\w/])@([\w-]+)", lambda m: stash(r"\citet{" + m.group(1) + "}"), s)
    s = re.sub(r"\*\*(.+?)\*\*", lambda m: stash(r"\textbf{" + inline_tex(m.group(1)) + "}"), s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", lambda m: stash(r"\emph{" + inline_tex(m.group(1)) + "}"), s)
    s = tex_escape(s)
    while "\x00" in s:
        s = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], s)
    return s


def to_tex(meta, blocks):
    out, appendix, in_abstract = [], False, False
    cmds = {1: "section", 2: "subsection", 3: "subsubsection"}
    for b in blocks:
        if b[0] == "h":
            if in_abstract:
                out.append(r"\end{abstract}")
                in_abstract = False
            if b[2] == "Abstract":
                out.append(r"\begin{abstract}")
                in_abstract = True
                continue
            if b[2] == "Appendix":
                out.append(r"\bibliographystyle{plainnat}" + "\n" + r"\bibliography{refs}" + "\n\n" + r"\appendix")
                appendix = True
                continue
            label = f"\\label{{{b[3]}}}" if b[3] else ""
            out.append(f"\\{cmds[b[1] - 1 if appendix else b[1]]}{{{inline_tex(b[2])}}}{label}")
        elif b[0] == "p":
            out.append(inline_tex(b[1]))
        elif b[0] == "list":
            env = "enumerate" if b[1] else "itemize"
            out.append(f"\\begin{{{env}}}\n" + "\n".join(r"  \item " + inline_tex(x) for x in b[2]) + f"\n\\end{{{env}}}")
        elif b[0] == "eq":
            if b[2]:
                out.append("\\begin{equation}\\label{" + b[2] + "}\n" + b[1] + "\n\\end{equation}")
            else:
                out.append("\\begin{equation*}\n" + b[1] + "\n\\end{equation*}")
        elif b[0] == "fig":
            out.append("\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{" + os.path.basename(b[2])
                       + "}\n\\caption{" + inline_tex(b[1]) + "}\\label{" + b[3] + "}\n\\end{figure}")
        elif b[0] == "table":
            head, *rows = b[3]
            spec = "l" * len(head)
            lines = [" & ".join(inline_tex(c) for c in head) + r" \\ \midrule"] + \
                    [" & ".join(inline_tex(c) for c in r) + r" \\" for r in rows]
            out.append("\\begin{table}[t]\n\\centering\n\\small\n\\caption{" + inline_tex(b[1]) + "}\\label{" + b[2]
                       + "}\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{" + spec + "}\n\\toprule\n" + "\n".join(lines)
                       + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}")
    if not appendix:
        out.append(r"\bibliographystyle{plainnat}" + "\n" + r"\bibliography{refs}")
    author = tex_escape(meta.get("author", ""))
    aff = tex_escape(meta.get("affiliation", ""))
    email = r"\\ \texttt{" + tex_escape(meta["email"]) + "}" if meta.get("email") else ""
    return (r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\graphicspath{{figures/}}
\title{""" + tex_escape(meta["title"]) + r"""}
\author{""" + author + r"""\\ \small """ + aff + email + r"""}
\date{""" + tex_escape(meta.get("date", "")) + r"""}
\begin{document}
\maketitle
""" + "\n\n".join(out) + "\n\\end{document}\n")


# ------------------------------------------------------------------------------------------------------------- main
def main():
    md = open(os.path.join(HERE, "paper.md"), encoding="utf-8").read()
    md = re.sub(r"^!include (\S+)$", lambda m: open(os.path.join(HERE, m.group(1)), encoding="utf-8").read().strip(),
                md, flags=re.M)
    if "--draft" not in sys.argv:
        assert "PENDING" not in md, "paper.md still contains PENDING markers (use --draft to build anyway)"
    bib = parse_bib(os.path.join(HERE, "refs.bib"))
    meta, blocks = parse(md)
    nums = number(blocks)
    for ref in set(re.findall(r"@((?:sec|fig|tab|eq):[\w-]+)", md)):
        assert ref in nums, f"undefined cross-reference {ref}"
    for keys in re.findall(r"\[(@[\w-]+(?:;\s*@[\w-]+)*)\]", md):
        for k in keys.split(";"):
            assert k.strip().lstrip("@") in bib, f"undefined citation {k}"
    html_out = to_html(meta, blocks, bib, nums)
    with open(os.path.join(HERE, "paper.html"), "w", encoding="utf-8") as f:
        f.write(html_out)
    tex_dir = os.path.join(HERE, "latex")
    os.makedirs(os.path.join(tex_dir, "figures"), exist_ok=True)
    with open(os.path.join(tex_dir, "paper.tex"), "w", encoding="utf-8") as f:
        f.write(to_tex(meta, blocks))
    shutil.copy(os.path.join(HERE, "refs.bib"), tex_dir)
    for b in blocks:
        if b[0] == "fig":
            shutil.copy(os.path.normpath(os.path.join(HERE, b[2])), os.path.join(tex_dir, "figures"))
    print("wrote docs/paper/paper.html and docs/paper/latex/ (paper.tex, refs.bib, figures/)")
    edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if os.path.exists(edge):
        prof = tempfile.mkdtemp()
        url = "file:///" + os.path.join(HERE, "paper.html").replace("\\", "/").replace(" ", "%20")
        subprocess.run([edge, "--headless=new", "--no-first-run", "--no-pdf-header-footer", f"--user-data-dir={prof}",
                        "--virtual-time-budget=15000", f"--print-to-pdf={os.path.join(HERE, 'paper.pdf')}", url],
                       check=False, timeout=180)
        print("printed docs/paper/paper.pdf (Edge returns before the file is fully written; wait a few seconds)")


if __name__ == "__main__":
    main()
