import os
import re
import sys
import yt_dlp
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel
import textwrap
import spacy
nlp = spacy.load("de_core_news_sm")

def normalize_for_lemma(word):
    word = word.lower().strip()

    # Remove articles
    articles = {
        "der", "die", "das", "den", "dem", "des",
        "ein", "eine", "einen", "einem", "eines"
    }
    parts = word.split()
    if parts and parts[0] in articles:
        parts = parts[1:]
    word = " ".join(parts)

    # Lemmatize
    doc = nlp(word)
    if len(doc) == 0:
        return word
    return doc[0].lemma_.lower()

MAIN_FILE = "my_vocabulary.txt"
INPUT_FILE = "new_words.txt"
MODEL_SIZE = "small"   # fast + accurate


def insert_transcript_into_existing_file(title, transcript):
    with open(MAIN_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    inside_target_video = False
    inserted = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect start of the correct video block
        if stripped == f"Video: {title}":
            inside_target_video = True

        # Detect start of next video block → insert transcript BEFORE it
        if inside_target_video and stripped.startswith("Video:") and stripped != f"Video: {title}":
            # Insert transcript BEFORE this new video block
            new_lines.append("\n--- Transcript ---\n")
            sentences = split_into_sentences(transcript)

            for s in sentences:
                wrapped_de = textwrap.fill(s, width=80)
                new_lines.append(wrapped_de + "\n")

                try:
                    translated = GoogleTranslator(source="de", target="en").translate(s)
                except:
                    translated = "N/A"

                wrapped_en = textwrap.fill(translated, width=80)
                new_lines.append(wrapped_en + "\n\n")

            inserted = True
            inside_target_video = False  # stop tracking

        new_lines.append(line)

    # If transcript belongs at the end (last video in file)
    if inside_target_video and not inserted:
        new_lines.append("\n--- Transcript ---\n")
        sentences = split_into_sentences(transcript)

        for s in sentences:
            wrapped_de = textwrap.fill(s, width=80)
            new_lines.append(wrapped_de + "\n")

            try:
                translated = GoogleTranslator(source="de", target="en").translate(s)
            except:
                translated = "N/A"

            wrapped_en = textwrap.fill(translated, width=80)
            new_lines.append(wrapped_en + "\n\n")

    # Write updated file
    with open(MAIN_FILE, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return True


# -----------------------------
# 1. TRANSLATION
# -----------------------------
def normalize_word(word):
    articles = {
        "der", "die", "das", "den", "dem", "des",
        "ein", "eine", "einen", "einem", "eines"
    }

    parts = word.lower().split()

    # Remove article if present
    if parts[0] in articles:
        parts = parts[1:]

    # Return normalized base word
    return "".join(parts)

def translate_word(word):
    try:
        return GoogleTranslator(source="de", target="en").translate(word)
    except:
        return "N/A"


# -----------------------------
# 2. UNIQUE WORD EXTRACTION
# -----------------------------
def extract_unique_words(text):
    words = re.findall(r"[a-zA-ZäöüÄÖÜß]+", text.lower())
    return sorted(set(words))


# -----------------------------
# 3. LOAD EXISTING VOCAB
# -----------------------------
def load_existing_words():
    if not os.path.exists(MAIN_FILE):
        return set(), set()

    with open(MAIN_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    existing_surface = set()
    existing_lemmas = set()

    for line in lines:
        if " - " in line or " — " in line:
            left = line.split(" - ")[0].split(" — ")[0].strip()
            existing_surface.add(left.lower())

            lemma = normalize_for_lemma(left)
            existing_lemmas.add(lemma)

    return existing_surface, existing_lemmas



# -----------------------------
# 4. APPEND TO VOCAB FILE
# -----------------------------
def append_to_main_file(entries):
    with open(MAIN_FILE, "a", encoding="utf-8") as f:
        for e in entries:
            f.write(e + "\n")


# -----------------------------
# 5. DOWNLOAD YOUTUBE AUDIO
# -----------------------------
def download_youtube_audio(url, start_time=0):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "temp_audio.%(ext)s",
        "quiet": True,
        "no_warnings": True,
    }

    # If start time is given, download only from that point
    if start_time > 0:
        ydl_opts["download_sections"] = f"*{start_time}-"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Untitled Video")
        filename = ydl.prepare_filename(info)

    if not filename.endswith(".mp3"):
        new_name = "temp_audio.mp3"
        os.system(f'ffmpeg -y -ss {start_time} -i "{filename}" "{new_name}"')
        os.remove(filename)
        filename = new_name

    return filename, title


def extract_start_time(url, args):
    # Case 1: URL contains &t=45 or &t=45s
    match = re.search(r"[?&]t=(\d+)", url)
    if match:
        return int(match.group(1))

    # Case 2: user gives second argument as seconds
    if len(args) > 2 and args[2].isdigit():
        return int(args[2])

    return 0


# -----------------------------
# 6. TRANSCRIBE AUDIO
# -----------------------------
def transcribe_audio(audio_path):
    model = WhisperModel(MODEL_SIZE, device="cpu")
    segments, _ = model.transcribe(audio_path, beam_size=5)
    text = " ".join([seg.text for seg in segments])
    return text

def split_into_sentences(text):
    # Split on . ? ! while keeping punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# -----------------------------
# 7. MAIN LOGIC
# -----------------------------
if __name__ == "__main__":
    existing_words = load_existing_words()
    new_entries = []

    # CASE 1: YouTube link provided
    if len(sys.argv) > 1 and sys.argv[1].startswith("http"):
        url = sys.argv[1]
        print("Downloading audio...")
        start_time = extract_start_time(url, sys.argv)
        audio_file, title = download_youtube_audio(url, start_time)

        print("Transcribing...")
        transcript = transcribe_audio(audio_file)
        # If video already exists, insert transcript instead of appending
        if title in open(MAIN_FILE, "r", encoding="utf-8").read():
            print("Video already exists — inserting transcript into existing section.")
            insert_transcript_into_existing_file(title, transcript)
            sys.exit(0)

        print("Extracting words...")
        words = extract_unique_words(transcript)

        # Add title + link (T3 format)
        new_entries.append(f"Video: {title}")
        new_entries.append(f"Link: {url}")
        new_entries.append("")

    # CASE 2: Local audio file provided
    elif len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        audio_file = sys.argv[1]
        title = os.path.basename(audio_file)

        print("Transcribing...")
        transcript = transcribe_audio(audio_file)

        print("Extracting words...")
        words = extract_unique_words(transcript)

        new_entries.append(f"Audio: {title}")
        new_entries.append("")

    # CASE 3: No input → use new_words.txt
    else:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            text = f.read()
        words = extract_unique_words(text)
        transcript = None  # no transcript in this mode

    # Add only new words
    print(f'words {words}')
    existing_surface, existing_lemmas = load_existing_words()

    for w in words:
        lemma = normalize_for_lemma(w)

        # Skip if lemma already exists
        if lemma in existing_lemmas:
            continue

        # Add new word
        eng = translate_word(w)
        line = f"{w} - {eng}"
        new_entries.append(line)
        print("Added:", line)

        # IMPORTANT: update lemma set immediately
        existing_lemmas.add(lemma)

    # Append transcript inside the same file
    if transcript:
        new_entries.append("")
        new_entries.append("--- Transcript ---")

        sentences = split_into_sentences(transcript)

        for s in sentences:
            # Wrap German sentence
            wrapped_de = textwrap.fill(s, width=80)
            new_entries.append(wrapped_de)

            # Translate whole sentence
            try:
                translated = GoogleTranslator(source="de", target="en").translate(s)
            except:
                translated = "N/A"

            # Wrap English translation
            wrapped_en = textwrap.fill(translated, width=80)
            new_entries.append(wrapped_en)
            new_entries.append("")  # blank line between entries


    append_to_main_file(new_entries)
    print("\nDone! Updated vocabulary saved in:", MAIN_FILE)
