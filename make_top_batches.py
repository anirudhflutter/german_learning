from pathlib import Path
import unicodedata
import re

WORDS_FILE = Path("de_lemmas_all.txt")
DEREWO_FILE = Path("german_freq/derewo_wordforms.txt")
OUT_FILE = Path("master_ranked_all_words.txt")

SPLIT_RE = re.compile(r"\s+")

def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip())

def main():
    # 1. Load universe
    universe = []
    cf_map = {}
    with WORDS_FILE.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = norm(line)
            if not w:
                continue
            universe.append(w)
            cf_map.setdefault(w.casefold(), []).append(w)

    ranked = []
    ranked_cf = set()

    # 2. Assign ranks using DeReWo
    with DEREWO_FILE.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = SPLIT_RE.split(line, maxsplit=1)[0]
            w = norm(token)
            cf = w.casefold()
            if cf in cf_map and cf not in ranked_cf:
                # choose shortest form as representative
                rep = min(cf_map[cf], key=len)
                ranked.append(rep)
                ranked_cf.add(cf)

    # 3. Collect unranked words
    unranked = []
    for w in universe:
        if w.casefold() not in ranked_cf:
            unranked.append(w)

    unranked = sorted(set(unranked), key=lambda x: x.casefold())

    # 4. Write final master list
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for w in ranked:
            out.write(w + "\n")
        for w in unranked:
            out.write(w + "\n")

    print("DONE")
    print(f"Ranked words    : {len(ranked):,}")
    print(f"Unranked words  : {len(unranked):,}")
    print(f"Total words     : {len(ranked) + len(unranked):,}")
    print(f"Output file     : {OUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
