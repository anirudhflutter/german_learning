from pathlib import Path
import spacy
import snowballstemmer

INPUT_FILES = [
    "dedup_by_lemma.txt",
]

OUTPUT_FILE = "dedup_family.txt"
LOG_FILE = "family_duplicates_removed.log"

# If True: write the canonical form (lemma if available else stem-key representative)
# If False: write the FIRST original form we saw for that family
WRITE_CANONICAL = False

def norm(s: str) -> str:
    return s.strip().replace("\ufeff", "")

def main():
    nlp = spacy.load("de_core_news_md", disable=["ner", "parser", "textcat"])
    stemmer = snowballstemmer.stemmer("german")

    seen_keys = set()
    kept = 0
    removed = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out, \
         open(LOG_FILE, "w", encoding="utf-8") as log:

        batch_raw, batch_norm = [], []

        def make_key(raw: str, token_lemma: str) -> str:
            w = raw.casefold()
            lem = (token_lemma or "").casefold().strip()

            # If spaCy produced a "real" lemma (different from surface),
            # trust it as the key.
            if lem and lem != w:
                return "LEM:" + lem

            # Otherwise fallback to stemming (collapses inflectional variants).
            # This is what reduces explosions like überzögest/überzögt/...
            st = stemmer.stemWord(w)
            return "STEM:" + st

        def flush():
            nonlocal kept, removed
            if not batch_norm:
                return False

            for raw, doc in zip(batch_raw, nlp.pipe(batch_norm, batch_size=2000)):
                if not doc:
                    continue

                lemma = doc[0].lemma_
                key = make_key(raw, lemma)

                if key in seen_keys:
                    removed += 1
                    log.write(f"DUP\t{raw}\t->\t{key}\n")
                    continue

                seen_keys.add(key)
                kept += 1

                if WRITE_CANONICAL:
                    # output lemma if meaningful; else output the raw word
                    w = raw
                    if lemma and lemma.casefold().strip() != raw.casefold():
                        w = lemma
                    out.write(w.strip() + "\n")
                else:
                    # keep first original form encountered for this family
                    out.write(raw.strip() + "\n")

            batch_raw.clear()
            batch_norm.clear()
            return False

        for fp in INPUT_FILES:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    raw = norm(line)
                    if not raw:
                        continue
                    batch_raw.append(raw)
                    batch_norm.append(raw)

                    if len(batch_norm) >= 5000:
                        flush()

        flush()

    print(f"Done. Kept {kept} unique families, removed {removed} variants.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Log:    {LOG_FILE}")

if __name__ == "__main__":
    main()
