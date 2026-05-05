import argparse
import os
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_prompts(path: str, column: str = "prompt"):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(path)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")
        prompts = df[column].dropna().astype(str).str.strip().tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(prompts)} prompts from {path}")
    return prompts


def analyze_semantic_diversity(prompts, sim_threshold=0.90):
    print(f"\nAnalyzing {len(prompts)} prompts...")

    # -------------------------
    # Exact duplicates
    # -------------------------
    exact_counts = Counter(prompts)
    num_unique = len(exact_counts)
    num_duplicates = sum(count - 1 for count in exact_counts.values() if count > 1)
    duplicate_ratio = num_duplicates / len(prompts)

    print(f"Unique prompts: {num_unique}")
    print(f"Duplicate ratio: {duplicate_ratio:.4f}")

    # -------------------------
    # Embeddings
    # -------------------------
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(prompts, convert_to_numpy=True, show_progress_bar=True)

    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # -------------------------
    # Pairwise similarity
    # -------------------------
    sim_matrix = cosine_similarity(embeddings)

    n = sim_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    pairwise_sims = sim_matrix[mask]

    avg_similarity = pairwise_sims.mean()
    diversity = 1 - avg_similarity

    print(f"\nAverage Pairwise Similarity: {avg_similarity:.4f}")
    print(f"Pairwise Diversity Score: {diversity:.4f}")

    # -------------------------
    # Semantic duplicates
    # -------------------------
    visited = set()
    num_repeated_prompts = 0

    for i in range(n):
        if i in visited:
            continue

        group = [i]
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= sim_threshold:
                group.append(j)

        if len(group) > 1:
            num_repeated_prompts += 1
            visited.update(group)

    print(f"Repeated prompt types (≥ {sim_threshold} sim): {num_repeated_prompts}")


def main():
    parser = argparse.ArgumentParser(description="Analyze semantic diversity of prompts.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to CSV or TXT file")
    parser.add_argument("--column", type=str, default="prompt", help="Column name for CSV")
    parser.add_argument("--sim_threshold", type=float, default=0.90)

    args = parser.parse_args()

    prompts = load_prompts(args.input_path, args.column)
    analyze_semantic_diversity(prompts, args.sim_threshold)


if __name__ == "__main__":
    main()