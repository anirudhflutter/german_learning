import json
import time
from pathlib import Path

import requests

# ================== CONFIG ==================
INPUT_FILE = "master_ranked_all_words.txt"
OUTPUT_FILE = "german_only.txt"
REJECT_FILE = "not_german.txt"        # optional; set to None to disable
LOG_FILE = "ollama_filter.log"
CHECKPOINT_FILE = "ollama_checkpoint.json"

MODEL = "llama3:latest"              # or "deepseek-r1:latest"
OLLAMA_URL = "http://localhost:11434/api/generate"

BATCH_SIZE = 10                     # 50–200 is a reasonable range
PRINT_EVERY = 5_00
CHECKPOINT_EVERY = 5_00

REQUEST_TIMEOUT = 600                # seconds

# ================== PROMPT ==================
SYSTEM_RULES = """
You are a strict binary classifier for German vocabulary.

Input: each line is ONE item (a token or short phrase).
Task: decide if the item is valid German to keep in a German vocabulary list.

OUTPUT FORMAT (must follow exactly):
- Return ONLY ONE line of JSON for each input line (JSONL).
- No prose, no explanations, no markdown, no extra whitespace lines.
- Schema:
  {"word":"<original>","keep":true|false}

KEEP = true if the item is:
1) A correct German word (including inflected forms: ich, ihn, war, dieses, des, etc.)
2) A very common German function word/particle/preposition/article/pronoun/conjunction
3) A common German contraction: im, am, ins, ums, zum, zur, beim, vom, durchs, fürs
4) A widely used German phrase (2–4 words max) written in German

KEEP = false if the item is:
- A person name (Peter, Michael, Hans, etc.)
- A place name / city / region (Berlin, München, etc.)  [NOTE: even if German]
- A company/brand name
- An abbreviation or acronym (EU, CDU, SPD, dpa, BER, etc.)
- English-only word or non-German word
- Random string, fragment, or unclear token (e.g., "Mu", "ST" unless it is clearly a German word)
- If you are unsure → keep=false

Important edge rules:
- Case matters: capitalize only if the original item is capitalized; do not modify the word.
- Do not translate, do not normalize, do not add/remove punctuation.
- Your output must be valid JSON (double quotes).

Now classify the following item(s).
"""

def load_checkpoint():
    if Path(CHECKPOINT_FILE).exists():
        return json.loads(Path(CHECKPOINT_FILE).read_text(encoding="utf-8"))
    return {"byte_offset": 0, "total_lines": 0, "kept": 0, "rejected": 0}

def save_checkpoint(state):
    tmp = CHECKPOINT_FILE + ".tmp"
    Path(tmp).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(tmp).replace(CHECKPOINT_FILE)

def ollama_classify_batch(words):
    # Build a single prompt containing the batch as lines
    user_block = "\n".join(words)

    payload = {
        "model": MODEL,
        "prompt": SYSTEM_RULES.strip() + "\n\nINPUT:\n" + user_block + "\n\nOUTPUT:",
        "stream": False,
"options": {"temperature": 0, "top_p": 1, "num_predict": 50}
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    text = r.json().get("response", "")

    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if "word" in obj and "keep" in obj:
                results.append(obj)
        except json.JSONDecodeError:
            # If model ever outputs non-JSON garbage, ignore it
            continue

    # If parsing failed badly, return "all false" (safer than accidentally keeping junk)
    if len(results) != len(words):
        return [{"word": w, "keep": False} for w in words]

    return results

def main():
    state = load_checkpoint()
    byte_offset = state["byte_offset"]
    total_lines = state["total_lines"]
    kept = state["kept"]
    rejected = state["rejected"]

    print(f"Resuming: byte_offset={byte_offset}, total_lines={total_lines}, kept={kept}, rejected={rejected}")
    print(f"Model: {MODEL}")

    # Ensure output files exist (append mode)
    Path(OUTPUT_FILE).touch()
    if REJECT_FILE:
        Path(REJECT_FILE).touch()

    start = time.time()

    try:
        with open(INPUT_FILE, "rb") as f_in, \
             open(OUTPUT_FILE, "a", encoding="utf-8") as f_out, \
             (open(REJECT_FILE, "a", encoding="utf-8") if REJECT_FILE else open(LOG_FILE, "a", encoding="utf-8")) as _dummy:

            f_in.seek(byte_offset)

            batch = []
            batch_orig = []

            while True:
                pos_before = f_in.tell()
                line_bytes = f_in.readline()
                if not line_bytes:
                    break

                s = line_bytes.decode("utf-8", errors="ignore").strip()
                if not s:
                    # empty line: just advance counters, no AI call
                    total_lines += 1
                    continue

                total_lines += 1
                batch.append(s)
                batch_orig.append(s)

                if len(batch) >= BATCH_SIZE:
                    results = ollama_classify_batch(batch_orig)

                    for obj in results:
                        w = obj["word"]
                        if bool(obj["keep"]):
                            f_out.write(w + "\n")
                            kept += 1
                        else:
                            rejected += 1
                            if REJECT_FILE:
                                with open(REJECT_FILE, "a", encoding="utf-8") as f_rej:
                                    f_rej.write(w + "\n")

                    batch.clear()
                    batch_orig.clear()

                # prints
                if total_lines % PRINT_EVERY == 0:
                    elapsed = time.time() - start
                    print(f"lines={total_lines:,} kept={kept:,} rejected={rejected:,} elapsed={elapsed:.1f}s")

                # checkpoint
                if total_lines % CHECKPOINT_EVERY == 0:
                    byte_offset = f_in.tell()
                    save_checkpoint({
                        "byte_offset": byte_offset,
                        "total_lines": total_lines,
                        "kept": kept,
                        "rejected": rejected,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })

            # flush remaining batch
            if batch_orig:
                results = ollama_classify_batch(batch_orig)
                for obj in results:
                    w = obj["word"]
                    if bool(obj["keep"]):
                        f_out.write(w + "\n")
                        kept += 1
                    else:
                        rejected += 1
                        if REJECT_FILE:
                            with open(REJECT_FILE, "a", encoding="utf-8") as f_rej:
                                f_rej.write(w + "\n")

            # final checkpoint
            byte_offset = f_in.tell()
            save_checkpoint({
                "byte_offset": byte_offset,
                "total_lines": total_lines,
                "kept": kept,
                "rejected": rejected,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            })

    except KeyboardInterrupt:
        print("\nCtrl+C received — saving checkpoint...")
        # best-effort save (byte offset where we are)
        save_checkpoint({
            "byte_offset": byte_offset,
            "total_lines": total_lines,
            "kept": kept,
            "rejected": rejected,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        print("Saved. Re-run to continue.")
        return

    print("\nDone.")
    print(f"Total lines: {total_lines:,}")
    print(f"Kept:        {kept:,}")
    print(f"Rejected:    {rejected:,}")
    print(f"Output:      {OUTPUT_FILE}")
    if REJECT_FILE:
        print(f"Rejected:    {REJECT_FILE}")
    print(f"Checkpoint:  {CHECKPOINT_FILE}")

if __name__ == "__main__":
    main()
