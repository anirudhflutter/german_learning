from pathlib import Path
import unicodedata
import argostranslate.translate as tr

IN_FILE = Path("master_ranked_all_words.txt")
OUT_FILE = Path("master_ranked_all_words_de_en.tsv")
PROGRESS_FILE = Path("translate_progress.txt")

def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip())

def main():
    start_idx = 0
    if PROGRESS_FILE.exists():
        start_idx = int(PROGRESS_FILE.read_text().strip() or "0")

    lines = IN_FILE.read_text(encoding="utf-8", errors="ignore").splitlines()
    total = len(lines)

    mode = "a" if OUT_FILE.exists() else "w"
    with OUT_FILE.open(mode, encoding="utf-8") as out:
        if mode == "w":
            out.write("de\ten\n")

        for i in range(start_idx, total):
            de = norm(lines[i])
            if not de:
                continue

            en = tr.translate(de, "de", "en")
            out.write(f"{de}\t{en}\n")

            # save progress every 1000 lines
            if (i + 1) % 1000 == 0:
                PROGRESS_FILE.write_text(str(i + 1), encoding="utf-8")
                print(f"{i+1:,}/{total:,}")

    PROGRESS_FILE.write_text(str(total), encoding="utf-8")
    print("DONE")

if __name__ == "__main__":
    main()
