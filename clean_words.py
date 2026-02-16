import re
import json
import time
import sqlite3
import subprocess
from pathlib import Path
from functools import lru_cache

import spacy
import snowballstemmer

# ================== CONFIG ==================
INPUT_FILES = ["master_ranked_all_words.txt"]  # you can add more parts later

DB_FILE = "vocab_store.sqlite"
CHECKPOINT_FILE = "checkpoint.json"

OUTPUT_FILE = "german_vocab_final.txt"
LOG_FILE = "german_vocab_final.log"

ALLOW_SPACES = True
ALLOW_HYPHEN = True

DROP_DIGITS = True
MIN_LEN = 2

PROGRESS_EVERY = 10_000
CHECKPOINT_EVERY = 10_000

SEPARABLE_PREFIXES = {
    "ab","an","auf","aus","ein","empor",
    "herauf","heraus","hervor","hoch",
    "unter","über","zu","zurück"
}

# ================== REGEX ==================
allowed = "A-Za-zÄÖÜäöüß"
if ALLOW_HYPHEN:
    allowed += r"\-"
if ALLOW_SPACES:
    allowed += r"\s"
re_ok = re.compile(rf"^[{allowed}]+$")

# ================== HELPERS ==================
def normalize_line(s: str) -> str:
    return s.strip().replace("\ufeff", "")

def is_obvious_junk(raw: str) -> bool:
    if not raw or len(raw) < MIN_LEN:
        return True
    if DROP_DIGITS and any(ch.isdigit() for ch in raw):
        return True
    if not re_ok.match(raw):
        return True
    if set(raw) <= {" ", "-"}:
        return True
    return False

@lru_cache(maxsize=1_000_000)
def hunspell_ok(word: str) -> bool:
    if not word:
        return False
    for form in (word, word.capitalize()):
        p = subprocess.run(
            ["hunspell", "-d", "de_DE", "-a", "-i", "UTF-8"],
            input=(form + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        out = p.stdout.decode("utf-8", errors="ignore").splitlines()
        if len(out) >= 2 and out[1].startswith("*"):
            return True
    return False

def entity_is_whole_name(doc) -> str | None:
    """
    Returns label if doc is exactly one named entity (PER/LOC/ORG), else None.
    """
    text = doc.text.strip()
    for ent in doc.ents:
        if ent.text.strip() == text and ent.label_ in ("PER", "LOC", "ORG"):
            return ent.label_
    return None

def verb_base_fallback(token: str) -> str | None:
    """
    Very cautious verb normalization:
    tries to convert forms to infinitive ONLY if Hunspell accepts the result.
    """
    w = token.casefold().strip()
    if not w:
        return None

    candidates = []

    for suf in ("end", "ende", "endem", "enden", "ender", "endes"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            candidates.append(w[:-len(suf)] + "en")

    for suf in ("etest", "etet", "eten", "ete"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            candidates.append(w[:-len(suf)] + "en")

    for suf in ("test", "tet", "ten", "te", "st"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            candidates.append(w[:-len(suf)] + "en")

    # imperative: häut -> häuten (prefer keep-t), macht -> machen (drop-t)
    if w.endswith("t") and len(w) > 3:
        c1 = w + "en"
        c2 = w[:-1] + "en"
        if hunspell_ok(c1):
            return c1
        if hunspell_ok(c2):
            return c2

    for c in candidates:
        if hunspell_ok(c):
            return c
    return None

def best_single_token(raw_cf: str, lemma_cf: str, *, is_capitalized: bool, pos_hint: str | None) -> str:
    """
    Picks a stable representative for ONE token.
    Key rules:
    - If raw is a valid German word, don't let spaCy replace it with a weird -en lemma.
    - For nouns/plurals: try collapsing -en plural to base if that base is Hunspell-valid.
    - For verbs: try verb_base_fallback (only if Hunspell accepts).
    """
    raw_cf = (raw_cf or "").strip()
    lemma_cf = (lemma_cf or "").strip()

    raw_ok = hunspell_ok(raw_cf)
    lemma_ok = hunspell_ok(lemma_cf)

    nounish = is_capitalized or (pos_hint in ("NOUN", "PROPN"))

    if nounish:
        # plural collapse: fronten -> front (ONLY if base is valid)
        if raw_ok and raw_cf.endswith("en") and len(raw_cf) > 4:
            cand = raw_cf[:-2]
            if hunspell_ok(cand):
                return cand

        # if raw is valid, do NOT accept different "-en" lemma (common spaCy mistake for compounds)
        if raw_ok and lemma_cf.endswith("en") and lemma_cf != raw_cf:
            return raw_cf

        # prefer valid lemma if not suspicious
        if lemma_ok and not (lemma_cf.endswith("en") and lemma_cf != raw_cf):
            return lemma_cf

        if raw_ok:
            return raw_cf
        return lemma_cf or raw_cf

    # verb/adjective side
    fb = verb_base_fallback(raw_cf) or verb_base_fallback(lemma_cf)
    if fb:
        return fb

    if lemma_ok:
        return lemma_cf
    if raw_ok:
        return raw_cf
    return lemma_cf or raw_cf

def normalize_rep(raw_orig: str, raw_cf: str, lemma_cf: str, doc) -> str:
    """
    - Separable verbs: keep prefix EXACTLY as written (ein/ab/aus...), never "einen".
    - Other phrases: conservative token-by-token (and if token counts mismatch, use raw phrase).
    - Single token: use POS hint from spaCy when possible.
    """
    raw_orig = raw_orig.strip()
    raw_cf = raw_cf.strip()
    lemma_cf = (lemma_cf or "").strip()

    if " " in raw_cf:
        parts_orig = raw_orig.split()
        parts_cf = raw_cf.split()
        lemma_parts = lemma_cf.split()

        # if spaCy tokenization doesn't align (often with hyphens), DON'T do risky alignment
        if len(parts_cf) != len(lemma_parts):
            lemma_parts = parts_cf[:]  # fall back to raw

        # separable verb phrase (exactly 2 tokens)
        if len(parts_cf) == 2 and parts_cf[1] in SEPARABLE_PREFIXES:
            first = best_single_token(
                parts_cf[0],
                lemma_parts[0] if lemma_parts else parts_cf[0],
                is_capitalized=parts_orig[0][:1].isupper(),
                pos_hint=None
            )
            return f"{first} {parts_cf[1]}"

        # general phrase token-by-token (conservative)
        m = min(len(parts_cf), len(lemma_parts), len(parts_orig))
        normalized = []
        for i in range(m):
            normalized.append(
                best_single_token(
                    parts_cf[i],
                    lemma_parts[i],
                    is_capitalized=parts_orig[i][:1].isupper(),
                    pos_hint=None
                )
            )
        if len(parts_cf) > m:
            normalized.extend(parts_cf[m:])
        return " ".join(normalized).strip() or (lemma_cf or raw_cf)

    # single token: POS hint if doc is exactly one token
    pos_hint = doc[0].pos_ if len(doc) == 1 else None
    return best_single_token(raw_cf, lemma_cf, is_capitalized=raw_orig[:1].isupper(), pos_hint=pos_hint)

# ================== DB + CHECKPOINT ==================
def init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vocab (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            key  TEXT UNIQUE NOT NULL,
            rep  TEXT NOT NULL
        );
    """)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.commit()

def load_checkpoint() -> dict:
    if Path(CHECKPOINT_FILE).exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "file_index": 0,
        "byte_offset": 0,
        "total_lines": 0,
        "kept": 0,
        "junk": 0,
        "dupes": 0,
        "names": 0,
        "last_written_id": 0,
        "updated_at": None,
    }

def save_checkpoint(state: dict):
    state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    Path(tmp).replace(CHECKPOINT_FILE)

def append_new_output(conn: sqlite3.Connection, last_written_id: int) -> int:
    rows = list(conn.execute(
        "SELECT id, rep FROM vocab WHERE id > ? ORDER BY id ASC",
        (last_written_id,)
    ))
    if not rows:
        return last_written_id

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for _id, rep in rows:
            out.write(rep + "\n")
    return rows[-1][0]

def ensure_output_consistency(state_last_written_id: int) -> int:
    """
    If output file is missing/empty, reset last_written_id so we can rebuild it by appending from DB.
    """
    p = Path(OUTPUT_FILE)
    if (not p.exists()) or p.stat().st_size == 0:
        # start fresh output file
        p.write_text("", encoding="utf-8")
        return 0
    return state_last_written_id

# ================== MAIN ==================
def main():
    try:
        nlp = spacy.load("de_core_news_lg")
    except Exception:
        nlp = spacy.load("de_core_news_md")

    stemmer = snowballstemmer.stemmer("german")
    conn = sqlite3.connect(DB_FILE)
    init_db(conn)

    state = load_checkpoint()

    kept = state.get("kept", 0)
    junk = state.get("junk", 0)
    dupes = state.get("dupes", 0)
    names = state.get("names", 0)
    total_lines = state.get("total_lines", 0)
    last_written_id = ensure_output_consistency(state.get("last_written_id", 0))
    print(f"Resuming from checkpoint: file_index={state['file_index']}, byte_offset={state['byte_offset']}, total_lines={total_lines}, kept={kept}, dupes={dupes}, junk={junk}, names={names}, last_written_id={last_written_id}")
    # log header (safe)
    log_path = Path(LOG_FILE)
    need_header = (not log_path.exists()) or (log_path.exists() and log_path.stat().st_size == 0)
    log = open(LOG_FILE, "a", encoding="utf-8")
    if need_header:
        log.write("ACTION\tRAW\tREP\tLEMMA\tKEY\n")

    start_time = time.time()

    def checkpoint_save(file_index: int, byte_offset: int):
        nonlocal last_written_id
        conn.commit()
        last_written_id = append_new_output(conn, last_written_id)
        print(f'opped dupli {file_index}')
        save_checkpoint({
            "file_index": file_index,
            "byte_offset": byte_offset,
            "total_lines": total_lines,
            "kept": kept,
            "junk": junk,
            "dupes": dupes,
            "names": names,
            "last_written_id": last_written_id,
            "updated_at": None,
        })

    try:
        for file_index in range(state["file_index"], len(INPUT_FILES)):
            fp = INPUT_FILES[file_index]
            offset = state["byte_offset"] if file_index == state["file_index"] else 0

            with open(fp, "rb") as f:
                if offset:
                    f.seek(offset)
                preview = f.readline().decode("utf-8", errors="ignore").strip()
                print(f"Resuming from word: {preview}")
                f.seek(offset)
                while True:
                    line_bytes = f.readline()
                    if not line_bytes:
                        break

                    total_lines += 1
                    raw_orig = normalize_line(line_bytes.decode("utf-8", errors="ignore"))
                    if not raw_orig:
                        continue

                    if is_obvious_junk(raw_orig):
                        junk += 1
                        continue

                    raw_cf = raw_orig.casefold().strip()

                    # spaCy on ORIGINAL text (keeps case info for nouns)
                    doc = nlp(raw_orig.strip())

                    # Name removal: only drop whole-entity PER/LOC/ORG if Hunspell rejects (prevents deleting real vocab)
                    if (" " not in raw_cf):
                        label = entity_is_whole_name(doc)
                        if label in ("PER", "LOC", "ORG") and (not hunspell_ok(raw_cf)):
                            names += 1
                            log.write(f"NAME\t{raw_orig}\t\t\t\n")
                            continue

                    lemma_tokens = [t.lemma_.casefold() for t in doc if t.text.strip()]
                    lemma_cf = " ".join(lemma_tokens).strip() or raw_cf

                    rep = normalize_rep(raw_orig, raw_cf, lemma_cf, doc)

                    key = "KEY:" + "|".join(stemmer.stemWord(t) for t in rep.split())

                    try:
                        conn.execute("INSERT INTO vocab(key, rep) VALUES (?, ?)", (key, rep))
                        kept += 1
                        log.write(f"KEEP\t{raw_orig}\t{rep}\t{lemma_cf}\t{key}\n")
                    except sqlite3.IntegrityError:
                        dupes += 1

                    if total_lines % PROGRESS_EVERY == 0:
                        elapsed = time.time() - start_time
                        print(f"[{fp}] lines={total_lines:,} kept={kept:,} dupes={dupes:,} junk={junk:,} names={names:,} elapsed={elapsed:.1f}s")

                    if total_lines % CHECKPOINT_EVERY == 0:
                        checkpoint_save(file_index=file_index, byte_offset=f.tell())

            # finished this file
            checkpoint_save(file_index=file_index + 1, byte_offset=0)

        # final flush
        conn.commit()
        last_written_id = append_new_output(conn, last_written_id)
        save_checkpoint({
            "file_index": len(INPUT_FILES),
            "byte_offset": 0,
            "total_lines": total_lines,
            "kept": kept,
            "junk": junk,
            "dupes": dupes,
            "names": names,
            "last_written_id": last_written_id,
            "updated_at": None,
        })

    except KeyboardInterrupt:
        # Stop safely
        print("\nCtrl+C received — saving safely...")
        try:
            checkpoint_save(file_index=state["file_index"], byte_offset=state["byte_offset"])
        except Exception:
            conn.commit()
        print("Saved. Re-run the script to continue from checkpoint.")
    finally:
        log.close()
        conn.close()

    print("\nDone/Stopped.")
    print("Kept:", f"{kept:,}")
    print("Dropped duplicates:", f"{dupes:,}")
    print("Dropped junk:", f"{junk:,}")
    print("Dropped names:", f"{names:,}")
    print("Output:", OUTPUT_FILE)
    print("DB:", DB_FILE)
    print("Checkpoint:", CHECKPOINT_FILE)
    print("Log:", LOG_FILE)

if __name__ == "__main__":
    main()