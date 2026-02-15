from pathlib import Path

RANK_FILE = Path("master_ranked_all_words.txt")
TSV_FILE  = Path("master_ranked_all_words_de_en.tsv")
OUT_FILE  = Path("master_ranked_all_words_de_en_sorted.tsv")

# 1) Load translations into dict
translations = {}

with TSV_FILE.open("r", encoding="utf-8", errors="ignore") as f:
    header = f.readline()
    for line in f:
        parts = line.rstrip("\n").split("\t", 1)
        if len(parts) == 2:
            translations[parts[0]] = parts[1]

print(f"Translations loaded: {len(translations):,}")

# 2) Walk through ranked list and emit in order
with RANK_FILE.open("r", encoding="utf-8", errors="ignore") as rf, \
     OUT_FILE.open("w", encoding="utf-8") as out:

    out.write("lemma\ten_equiv\n")

    written = 0
    for line in rf:
        lemma = line.strip()
        if lemma in translations:
            out.write(f"{lemma}\t{translations[lemma]}\n")
            written += 1

print("DONE")
print(f"Written (ranked): {written:,}")
print(f"Output: {OUT_FILE.resolve()}")
