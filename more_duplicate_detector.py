import re
import unicodedata
from collections import defaultdict

INPUT_FILE = "merged_lemmas.txt"

OUT_DUP_TSV = "possible_duplicates.tsv"
OUT_DUP_TXT = "possible_duplicates.txt"
OUT_CANONICAL_TXT = "normalized_canonical.txt"
OUT_CANONICAL_MAP = "canonical_map.tsv"

# ---------- basic helpers ----------

def normalize(token: str) -> str:
    token = token.strip()
    token = unicodedata.normalize("NFC", token)
    token = token.lower()
    token = re.sub(r"^[^\wäöüß]+|[^\wäöüß]+$", "", token, flags=re.UNICODE)
    return token

def is_alpha_de(token: str) -> bool:
    return bool(re.fullmatch(r"[a-zäöüß]+", token))

def load_tokens(path: str):
    toks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = normalize(line)
            if t and is_alpha_de(t):
                toks.append(t)
    return sorted(set(toks))

# ---------- normalization variants ----------

def ss_variant(s: str) -> str:
    """Canonical form with ß normalized to ss (safe for grouping variants)."""
    return s.replace("ß", "ss")

def umlaut_ascii_variants(s: str):
    """
    Generate cautious transliteration variants:
      ä <-> ae, ö <-> oe, ü <-> ue
    This is useful for grouping, but risky for automatic replacement.
    """
    variants = {s}

    # direct umlaut -> digraph
    v1 = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    variants.add(v1)

    # digraph -> umlaut (all occurrences)
    v2 = s.replace("ae", "ä").replace("oe", "ö").replace("ue", "ü")
    variants.add(v2)

    return variants

def old_new_spelling_variants(s: str):
    """
    Generate common orthography variants (not exhaustive).
    Examples:
      fluß <-> fluss
      muß <-> muss
    """
    variants = {s}
    variants.add(s.replace("ß", "ss"))
    # also reverse, but only on common "ss" strings:
    # we won't generate all possible ss->ß aggressively (too noisy)
    return variants

# ---------- token quality / noise heuristics ----------

def looks_noisy(token: str) -> bool:
    """
    Heuristics for likely broken/truncated/noisy forms.
    Conservative: flags for review, does not auto-delete.
    """
    # too short usually not useful (except valid short words, so keep very conservative)
    if len(token) <= 2:
        return True

    # strange endings often indicate truncation in your kind of list
    suspicious_endings = (
        "plän", "syst", "krümm", "kanäl", "öf", "itngen", "gsk", "stng"
    )
    if token.endswith(suspicious_endings):
        return True

    # repeated weird consonant clusters (soft heuristic)
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", token):
        # German compounds can be long, so only flag if no vowel at all in tail
        tail = token[-6:]
        if not re.search(r"[aeiouäöü]", tail):
            return True

    return False

# ---------- duplicate candidate logic ----------

def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def truncation_candidate(shorter: str, longer: str) -> bool:
    """
    Detect likely truncation:
      abfallsyst -> abfallsystem
      abgaskrümm -> abgaskrümmer
    """
    if len(shorter) < 5:
        return False
    if len(longer) - len(shorter) < 1 or len(longer) - len(shorter) > 5:
        return False
    if not longer.startswith(shorter):
        return False

    # avoid merging valid inflections blindly (e.g. haus/hausen etc.)
    # this is a review signal, not an auto merge
    return True

def one_suffix_family(a: str, b: str):
    """
    Check if two words differ by a common inflection-like suffix.
    Returns base if likely.
    """
    suffixes = ["e", "en", "er", "em", "es", "n", "s", "t", "te", "end"]
    for suf in suffixes:
        if a + suf == b:
            return a, b, suf
        if b + suf == a:
            return b, a, suf
    return None

def score_pair(a: str, b: str):
    """
    Score likely duplicate/variant relation between two tokens.
    Returns (score, reason, confidence)
    """
    reasons = []
    score = 0

    # exact after ß->ss normalization
    if ss_variant(a) == ss_variant(b) and a != b:
        score += 5
        reasons.append("ß↔ss")

    # exact after umlaut translit normalization (grouping signal)
    a_um = umlaut_ascii_variants(a)
    b_um = umlaut_ascii_variants(b)
    if a_um.intersection(b_um) and a != b:
        score += 3
        reasons.append("umlaut/translit")

    # prefix truncation signal
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if truncation_candidate(s, l):
        score += 4
        reasons.append("prefix-truncation")

    # simple suffix family
    fam = one_suffix_family(a, b)
    if fam:
        score += 2
        reasons.append(f"suffix+{fam[2]}")

    # strong shared prefix but small tail difference
    cpl = common_prefix_len(a, b)
    if cpl >= 6:
        tail_delta = abs(len(a) - len(b))
        if tail_delta <= 3:
            score += 1
            reasons.append("long-common-prefix")

    # confidence buckets
    if score >= 6:
        conf = "safe"
    elif score >= 4:
        conf = "medium"
    else:
        conf = "risky"

    return score, "|".join(reasons), conf

# ---------- canonicalization ----------

def choose_canonical(group):
    """
    Pick a canonical representative for a variant group.
    Preference:
      1) modern spelling (ss over ß for consistency) -> internal score only
      2) longer token (avoid truncations)
      3) lexicographically stable fallback
    """
    def key_fn(tok):
        # prefer non-noisy, longer, "ss" normalized style for consistency
        noisy_penalty = 1 if looks_noisy(tok) else 0
        has_sz = 1 if "ß" in tok else 0  # prefer 'ss' canonical if equivalent
        return (noisy_penalty, has_sz, -len(tok), tok)

    # min by this key gives best
    return sorted(group, key=key_fn)[0]

# ---------- union-find ----------

class DSU:
    def __init__(self, items):
        self.p = {x: x for x in items}
        self.r = {x: 0 for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1

# ---------- main ----------

def main():
    tokens = load_tokens(INPUT_FILE)
    token_set = set(tokens)

    # Build buckets to avoid O(n^2):
    # 1) ss-normalized buckets (great for ß/ss)
    # 2) first-6-char buckets (good for truncation/prefix similarity)
    buckets = defaultdict(set)

    for t in tokens:
        buckets[("ss", ss_variant(t))].add(t)
        buckets[("p6", t[:6])].add(t)
        # also bucket translit-normalized forms
        translit = t.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
        buckets[("tr", translit)].add(t)

    # candidate pairs
    candidate_pairs = set()
    for _, group in buckets.items():
        if len(group) < 2:
            continue
        glist = sorted(group)
        # local pair generation only within bucket
        for i in range(len(glist)):
            for j in range(i + 1, len(glist)):
                a, b = glist[i], glist[j]
                # quick cheap filter
                if abs(len(a) - len(b)) > 5:
                    continue
                if common_prefix_len(a, b) < 4 and ss_variant(a) != ss_variant(b):
                    continue
                candidate_pairs.add((a, b))

    # score pairs
    rows = []
    dsu = DSU(tokens)

    for a, b in sorted(candidate_pairs):
        score, reason, conf = score_pair(a, b)
        if score <= 0:
            continue

        # save review row
        rows.append((a, b, score, conf, reason))

        # auto-union only SAFE and orthography-ish/truncation-ish
        # (still conservative)
        if conf == "safe":
            dsu.union(a, b)

    # Build groups
    groups = defaultdict(list)
    for t in tokens:
        groups[dsu.find(t)].append(t)

    canonical_map = {}
    final_canon_set = set()

    for root, group in groups.items():
        if len(group) == 1:
            canon = group[0]
        else:
            canon = choose_canonical(group)

        for t in group:
            canonical_map[t] = canon
        final_canon_set.add(canon)

    # Write candidate review TSV
    with open(OUT_DUP_TSV, "w", encoding="utf-8") as f:
        f.write("token_a\ttoken_b\tscore\tconfidence\treason\n")
        for a, b, score, conf, reason in sorted(rows, key=lambda x: (-x[2], x[3], x[0], x[1])):
            f.write(f"{a}\t{b}\t{score}\t{conf}\t{reason}\n")

    # Write candidate review TXT (human-readable)
    with open(OUT_DUP_TXT, "w", encoding="utf-8") as f:
        f.write("Possible duplicate / variant pairs (review)\n")
        f.write("=======================================\n\n")
        for a, b, score, conf, reason in sorted(rows, key=lambda x: (-x[2], x[3], x[0], x[1])):
            f.write(f"[{conf:6}] score={score:2d}  {a}  <->  {b}   ({reason})\n")

    # Write canonical map
    with open(OUT_CANONICAL_MAP, "w", encoding="utf-8") as f:
        f.write("original\tcanonical\n")
        for t in sorted(canonical_map):
            f.write(f"{t}\t{canonical_map[t]}\n")

    # Write canonicalized txt (deduped output)
    with open(OUT_CANONICAL_TXT, "w", encoding="utf-8") as f:
        for t in sorted(final_canon_set):
            f.write(t + "\n")

    # Extra: noisy token report (appended to txt)
    noisy = [t for t in tokens if looks_noisy(t)]
    with open(OUT_DUP_TXT, "a", encoding="utf-8") as f:
        f.write("\n\nLikely noisy / broken tokens (review)\n")
        f.write("=====================================\n")
        for t in noisy[:5000]:  # cap to keep file manageable
            f.write(t + "\n")

    print(f"Input tokens: {len(tokens)}")
    print(f"Candidate pairs found: {len(rows)}")
    print(f"Canonical unique tokens: {len(final_canon_set)}")
    print(f"Wrote: {OUT_DUP_TSV}")
    print(f"Wrote: {OUT_DUP_TXT}")
    print(f"Wrote: {OUT_CANONICAL_MAP}")
    print(f"Wrote: {OUT_CANONICAL_TXT}")

if __name__ == "__main__":
    main()
