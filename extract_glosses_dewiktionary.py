from __future__ import annotations
from pathlib import Path
import bz2
import re
import unicodedata
from lxml import etree

# Inputs
WORDS_FILE = Path("master_ranked_all_words.txt")  # your 1M ranked list
DUMP_BZ2   = Path("dewiktionary-latest-pages-articles.xml.bz2")

# Output
OUT_TSV = Path("master_ranked_all_words_gloss_de_en.tsv")

# --- regex helpers (Wiktionary markup is messy; this is a pragmatic extractor) ---
RE_DE_SECTION = re.compile(r"==\s*[^=\n]*\(\s*\{\{Sprache\|Deutsch\}\}\s*\)\s*==", re.I)
RE_BED = re.compile(r"\{\{Bedeutungen\}\}", re.I)
RE_GLOSS_LINE = re.compile(r"^\s*:\s*\[\s*1\s*\]\s*(.+)$", re.M)  # first meaning
RE_EN_TRANS = re.compile(r"^\s*\*\s*\{\{en\}\}\s*:\s*(.+)$", re.M)  # English translation line
RE_U_EN = re.compile(r"\{\{Ü\|en\|([^}|]+)", re.I)  # {{Ü|en|word}}
RE_LINK = re.compile(r"\[\[([^]\|]+)(?:\|[^]]+)?\]\]")  # [[link|text]]
RE_TEMPL = re.compile(r"\{\{[^{}]+\}\}")  # naive template remover

def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip())

def clean_wiki_text(s: str) -> str:
    s = RE_LINK.sub(r"\1", s)
    s = RE_TEMPL.sub("", s)
    s = s.replace("''", "").replace("<br />", " ").replace("<br/>", " ").replace("<br>", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_de_block(page_text: str) -> str | None:
    m = RE_DE_SECTION.search(page_text)
    if not m:
        return None
    start = m.start()
    # end at next language section "== ... ({{Sprache|...}}) ==" if present
    nxt = re.search(r"\n==\s*[^=\n]*\(\s*\{\{Sprache\|", page_text[m.end():], re.I)
    end = (m.end() + nxt.start()) if nxt else len(page_text)
    return page_text[start:end]

def extract_first_de_gloss(de_block: str) -> str:
    if not RE_BED.search(de_block):
        return ""
    m = RE_GLOSS_LINE.search(de_block)
    if not m:
        return ""
    return clean_wiki_text(m.group(1))

def extract_first_en_equiv(de_block: str) -> str:
    # Prefer explicit English translation line if present
    m = RE_EN_TRANS.search(de_block)
    if m:
        line = m.group(1)
        # Try to pull first {{Ü|en|...}} from that line
        m2 = RE_U_EN.search(line)
        if m2:
            return clean_wiki_text(m2.group(1))
        return clean_wiki_text(line)

    # Otherwise look anywhere for {{Ü|en|...}} and take first hit
    m3 = RE_U_EN.search(de_block)
    if m3:
        return clean_wiki_text(m3.group(1))

    return ""

def load_targets(path: Path) -> set[str]:
    # Using a set makes this fast. 1M lines is okay on modern machines.
    targets = set()
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = norm(line)
            if w:
                targets.add(w)
    return targets

def iter_wiktionary_pages(bz2_path: Path):
    # Stream parse the XML to avoid RAM blowups
    with bz2.open(bz2_path, "rb") as f:
        context = etree.iterparse(f, events=("end",), tag="{*}page")
        for _, elem in context:
            title_el = elem.find("{*}title")
            rev_el = elem.find("{*}revision")
            text_el = rev_el.find("{*}text") if rev_el is not None else None

            title = title_el.text if title_el is not None else ""
            text = text_el.text if text_el is not None else ""

            yield title, text

            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

def main():
    if not WORDS_FILE.exists():
        raise FileNotFoundError(WORDS_FILE)
    if not DUMP_BZ2.exists():
        raise FileNotFoundError(DUMP_BZ2)

    targets = load_targets(WORDS_FILE)
    print(f"Targets loaded: {len(targets):,}")

    found = 0

    with OUT_TSV.open("w", encoding="utf-8") as out:
        out.write("lemma\tde_gloss\ten_equiv\n")

        for title, text in iter_wiktionary_pages(DUMP_BZ2):
            title = norm(title)
            if title not in targets:
                continue

            de_block = extract_de_block(text or "")
            if not de_block:
                continue

            de_gloss = extract_first_de_gloss(de_block)
            en_equiv = extract_first_en_equiv(de_block)

            out.write(f"{title}\t{de_gloss}\t{en_equiv}\n")
            found += 1

            if found % 5000 == 0:
                print(f"Matched pages: {found:,}")

    print("DONE")
    print(f"Total matched pages written: {found:,}")
    print(f"Output: {OUT_TSV.resolve()}")

if __name__ == "__main__":
    main()
