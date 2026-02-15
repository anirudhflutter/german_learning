from __future__ import annotations

import re
from pathlib import Path

IN_TSV = Path("de_lemmas_with_pos.tsv")
OUT_TSV = Path("de_lemmas_clean.tsv")
OUT_TXT = Path("de_lemmas_clean.txt")

KEEP_POS = {"Substantiv", "Verb", "Adjektiv", "Adverb"}

DROP_POS_CONTAINS = {
    "Deklinierte Form",
    "Konjugierte Form",
    "Flexion",
    "Partizip",
    "Komparativ",
    "Superlativ",
}

# allow German letters + hyphen + apostrophe (optional)
TITLE_OK = re.compile(r"^[A-Za-zÄÖÜäöüß\-']+$")

def normalize(lemma: str) -> str:
    return lemma.strip()

def main() -> None:
    if not IN_TSV.exists():
        raise FileNotFoundError(f"Missing {IN_TSV}. Run your extractor first.")

    seen = set()
    kept_rows: list[tuple[str, str]] = []

    with IN_TSV.open("r", encoding="utf-8") as f:
        header = next(f, None)  # lemma\tpos
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            lemma, pos = parts
            lemma = normalize(lemma)
            pos = pos.strip()

            # basic filtering
            if ":" in lemma:
                continue  # namespaces
            if not TITLE_OK.match(lemma):
                continue  # drop spaces/punctuation; adjust later if you want multiword expressions

            # drop unwanted POS
            if any(x.lower() in pos.lower() for x in DROP_POS_CONTAINS):
                continue

            # keep only the core POS for writing power
            if pos not in KEEP_POS:
                continue

            key = (lemma, pos)
            if key in seen:
                continue
            seen.add(key)
            kept_rows.append(key)

    # write outputs
    with OUT_TSV.open("w", encoding="utf-8") as out:
        out.write("lemma\tpos\n")
        for lemma, pos in kept_rows:
            out.write(f"{lemma}\t{pos}\n")

    with OUT_TXT.open("w", encoding="utf-8") as out:
        for lemma, pos in kept_rows:
            out.write(lemma + "\n")

    print(f"Kept rows: {len(kept_rows):,}")
    print(f"Saved: {OUT_TSV.resolve()}")
    print(f"Saved: {OUT_TXT.resolve()}")

if __name__ == "__main__":
    main()
