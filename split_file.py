from pathlib import Path
import spacy

# ---- CONFIG ----
INPUT_FILES = [
    "master_ranked_all_words.txt",
    # or use a single file:
    # "/mnt/data/top_61524_words.txt",
]

OUTPUT_FILE = "dedup_by_lemma.txt"
LOG_FILE = "duplicates_removed.log"

# If True: output the lemma (base form) only.
# If False: keep the FIRST SEEN original form, but dedupe by lemma.
WRITE_LEMMA = True

# ---- MAIN ----
def normalize(s: str) -> str:
    return s.strip().replace("\ufeff", "")

def main():
    nlp = spacy.load("de_core_news_md", disable=["ner", "parser", "textcat"])

    seen_lemmas = set()
    kept = 0
    removed = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out, \
         open(LOG_FILE, "w", encoding="utf-8") as log:

        for file_path in INPUT_FILES:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                # Use nlp.pipe for speed (batching)
                batch_raw = []
                batch_norm = []

                def flush():
                    nonlocal kept, removed
                    if not batch_norm:
                        return
                    for raw, doc in zip(batch_raw, nlp.pipe(batch_norm, batch_size=2000)):
                        if not doc:
                            continue
                        lemma = doc[0].lemma_.casefold().strip()
                        if not lemma:
                            continue

                        if lemma in seen_lemmas:
                            removed += 1
                            log.write(f"DUP\t{raw}\t->\t{lemma}\n")
                            continue

                        seen_lemmas.add(lemma)
                        kept += 1
                        out.write((lemma if WRITE_LEMMA else raw) + "\n")

                    batch_raw.clear()
                    batch_norm.clear()

                for line in f:
                    raw = normalize(line)
                    if not raw:
                        continue
                    batch_raw.append(raw)
                    batch_norm.append(raw)

                    if len(batch_norm) >= 5000:
                        flush()

                flush()

    print(f"Done. Kept {kept} unique lemmas, removed {removed} duplicates.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Log:    {LOG_FILE}")

if __name__ == "__main__":
    main()
