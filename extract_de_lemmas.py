from __future__ import annotations

import bz2
import re
from pathlib import Path
from lxml import etree

# 1) CHANGE THIS PATH to where your .bz2 file is
DUMP_PATH = Path.home() / "Downloads" / "dewiktionary-latest-pages-articles.xml.bz2"

# Output files
OUT_ALL_DE = Path("de_lemmas_all.txt")
OUT_POS = Path("de_lemmas_with_pos.tsv")

# Markers inside Wiktionary page text
GERMAN_HEADER_RE = re.compile(
    r"^==\s*[^=\n]*\(\s*\{\{Sprache\|Deutsch\}\}\s*\)\s*==\s*$",
    re.MULTILINE
)
POS_RE = re.compile(
    r"\{\{Wortart\|([^|}]+)\|Deutsch\}\}",
    re.IGNORECASE
)

def is_good_title(title: str) -> bool:
    if not title or len(title) < 2:
        return False
    if ":" in title:  # namespace
        return False
    return True

def main() -> None:
    if not DUMP_PATH.exists():
        raise FileNotFoundError(f"Dump not found: {DUMP_PATH}")

    count_pages = 0
    count_de = 0

    OUT_ALL_DE.write_text("", encoding="utf-8")
    OUT_POS.write_text("lemma\tpos\n", encoding="utf-8")

    # Stream read the compressed XML
    with bz2.open(DUMP_PATH, "rb") as f:
        # iterate only over end events to keep memory low
        context = etree.iterparse(f, events=("end",), recover=True)

        for event, elem in context:
            # The MediaWiki XML uses <page> with <title> and <revision><text>
            if elem.tag.endswith("page"):
                title_elem = elem.find("./{*}title")
                text_elem = elem.find("./{*}revision/{*}text")
                ns_elem = elem.find("./{*}ns")
                ns = int(ns_elem.text) if ns_elem is not None and ns_elem.text else -1
                if ns != 0:
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
                    continue

                title = title_elem.text if title_elem is not None else None
                text = text_elem.text if text_elem is not None else ""

                count_pages += 1

                if title and is_good_title(title) and text and GERMAN_HEADER_RE.search(text):
                    count_de += 1

                    # Save lemma
                    with OUT_ALL_DE.open("a", encoding="utf-8") as out:
                        out.write(title + "\n")

                    # Save POS (0..n entries per lemma)
                    pos_matches = POS_RE.findall(text)
                    if pos_matches:
                        with OUT_POS.open("a", encoding="utf-8") as out:
                            for pos in sorted(set(pos_matches)):
                                out.write(f"{title}\t{pos}\n")
                    else:
                        with OUT_POS.open("a", encoding="utf-8") as out:
                            out.write(f"{title}\tUNKNOWN\n")

                # Important: free memory
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

                # Progress print every 50k pages
                if count_pages % 50000 == 0:
                    print(f"Pages: {count_pages:,} | German lemmas found: {count_de:,}")

    print("DONE")
    print(f"Total pages processed: {count_pages:,}")
    print(f"German lemmas found: {count_de:,}")
    print(f"Saved: {OUT_ALL_DE.resolve()}")
    print(f"Saved: {OUT_POS.resolve()}")

if __name__ == "__main__":
    main()
