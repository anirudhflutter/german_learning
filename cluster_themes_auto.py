import os
from collections import Counter, defaultdict

from sentence_transformers import SentenceTransformer
import hdbscan
import umap
import numpy as np

INPUT_FILE = "normalized_canonical.txt"   # your 219,314 lemmas
OUTPUT_DIR = "themes_auto"

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def load_words(path):
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w)
    return words


def embed_words(model_name, words):
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(words):,} words...")
    embeddings = model.encode(words, batch_size=256, show_progress_bar=True)
    return embeddings


def reduce_dim(embeddings):
    print("Reducing dimensionality with UMAP (for better clustering)...")
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=50,
        metric="cosine",
        random_state=42,
    )
    emb_reduced = reducer.fit_transform(embeddings)
    return emb_reduced


def cluster_hdbscan(embeddings):
    print("Clustering with HDBSCAN (automatic number of clusters)...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings)
    return labels


def build_clusters(words, labels):
    clusters = defaultdict(list)
    for w, lab in zip(words, labels):
        clusters[lab].append(w)
    return clusters


def name_cluster(words_in_cluster, top_n=10):
    # Simple heuristic: use first N words as "representatives"
    # You can later refine this manually if you want.
    reps = words_in_cluster[:top_n]
    return "_".join(reps[:3])  # short label from first 3 words


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file not found: {INPUT_FILE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load words
    words = load_words(INPUT_FILE)
    print(f"Loaded {len(words):,} words from {INPUT_FILE}")

    # 2) Embed
    embeddings = embed_words(EMBEDDING_MODEL, words)

    # 3) Reduce dimension (optional but helps HDBSCAN)
    emb_reduced = reduce_dim(embeddings)

    # 4) Cluster
    labels = cluster_hdbscan(emb_reduced)

    # 5) Build clusters
    clusters = build_clusters(words, labels)

    # 6) Stats
    label_counts = Counter(labels)
    n_noise = label_counts.get(-1, 0)
    n_clusters = len([l for l in label_counts.keys() if l != -1])

    print("\n==================== CLUSTERING RESULTS ====================")
    print(f"Total words:              {len(words):,}")
    print(f"Number of clusters:       {n_clusters:,}")
    print(f"Noise / unclustered:      {n_noise:,}")
    print("============================================================\n")

    total_in_clusters = 0

    # 7) Write cluster files
    for label, wlist in clusters.items():
        if label == -1:
            # noise cluster
            fname = os.path.join(OUTPUT_DIR, "theme_noise_unclustered.txt")
            with open(fname, "w", encoding="utf-8") as f:
                for w in wlist:
                    f.write(w + "\n")
            print(f"[noise]  {len(wlist):6d} words → {fname}")
            continue

        total_in_clusters += len(wlist)

        # Generate a rough name from representative words
        theme_name = name_cluster(wlist)
        fname = os.path.join(OUTPUT_DIR, f"theme_{label:03d}_{theme_name}.txt")

        with open(fname, "w", encoding="utf-8") as f:
            for w in wlist:
                f.write(w + "\n")

        print(f"[cluster {label:3d}] {len(wlist):6d} words → {fname}")

    print("\n==================== SUMMARY ====================")
    print(f"Total words:                 {len(words):,}")
    print(f"Words in clusters (≠ -1):    {total_in_clusters:,}")
    print(f"Noise / unclustered (-1):    {n_noise:,}")
    print(f"Sum check:                   {total_in_clusters + n_noise:,}")
    print("=================================================")


if __name__ == "__main__":
    main()
