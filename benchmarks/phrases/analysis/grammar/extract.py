"""JATS full texts -> one JSON line per paragraph (abstract, body, captions; no references)."""
import glob, json, re, sys
from lxml import etree
SKIP = {"ref-list", "table", "tex-math", "inline-formula", "disp-formula", "xref", "ack", "back", "table-wrap-foot", "sup", "ext-link"}
def text(el):
    out = []
    def walk(e):
        if not isinstance(e.tag, str): 
            if e.tail: out.append(e.tail)
            return
        tag = etree.QName(e).localname
        if tag not in SKIP:
            if e.text: out.append(e.text)
            for c in e: walk(c)
        if e.tail: out.append(e.tail)
    if el.text: out.append(el.text)
    for c in el: walk(c)
    return re.sub(r"\s+", " ", "".join(out)).strip()
n = 0
with open(sys.argv[2], "w") as fh:
    for f in sorted(glob.glob(sys.argv[1] + "/*.xml")):
        try: root = etree.parse(f, etree.XMLParser(recover=True, huge_tree=True)).getroot()
        except Exception: continue
        pid = f.rsplit("/", 1)[1][:-4]
        for part, xp in (("abstract", ".//*[local-name()='abstract']//*[local-name()='p']"),
                         ("body", ".//*[local-name()='body']//*[local-name()='p']")):
            for p in root.xpath(xp):
                if p.xpath("ancestor::*[local-name()='p']"): continue
                sec = p.xpath("ancestor::*[local-name()='sec'][last()]/*[local-name()='title']")
                cap = bool(p.xpath("ancestor::*[local-name()='caption']"))
                t = text(p)
                if len(t) < 40: continue
                fh.write(json.dumps({"paper": pid, "part": "caption" if cap else part,
                                     "sec": text(sec[0])[:60] if sec else "", "text": t}) + "\n"); n += 1
print(n)
