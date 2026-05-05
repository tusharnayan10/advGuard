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

    # Exact duplicates
    exact_counts = Counter(prompts)
    num_unique = len(exact_counts)
    num_duplicates = sum(count - 1 for count in exact_counts.values() if count > 1)
    duplicate_ratio = num_duplicates / len(prompts)

    print(f"\n--- Exact Duplicates ---")
    print(f"Unique prompts: {num_unique}")
    print(f"Duplicate ratio: {duplicate_ratio:.4f}")

    # Embeddings
    print("\nEncoding embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(prompts, convert_to_numpy=True, show_progress_bar=True)

    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # Average cosine similarity to centroid
    centroid = np.mean(embeddings, axis=0, keepdims=True)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

    cosine_to_centroid = cosine_similarity(embeddings, centroid).flatten()
    avg_cosine_similarity = cosine_to_centroid.mean()

    print(f"\n--- Centroid Similarity ---")
    print(f"Average Cosine Similarity to Centroid: {avg_cosine_similarity:.4f}")

    # Pairwise similarity
    print("\nComputing pairwise similarity...")
    sim_matrix = cosine_similarity(embeddings)

    n = sim_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    pairwise_sims = sim_matrix[mask]

    avg_pairwise_similarity = pairwise_sims.mean()
    diversity_score = 1 - avg_pairwise_similarity

    print(f"\n--- Pairwise Similarity ---")
    print(f"Average Pairwise Similarity: {avg_pairwise_similarity:.4f}")
    print(f"Pairwise Diversity Score: {diversity_score:.4f}")

    # Semantic duplicates
    visited = set()
    num_repeated_groups = 0

    for i in range(n):
        if i in visited:
            continue

        group = [i]
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= sim_threshold:
                group.append(j)

        if len(group) > 1:
            num_repeated_groups += 1
            visited.update(group)

    print(f"\n--- Semantic Duplicates ---")
    print(f"Repeated prompt groups (≥ {sim_threshold} sim): {num_repeated_groups}")


def main():
    parser = argparse.ArgumentParser(description="Analyze semantic diversity of prompts.")
    parser.add_argument("--column", type=str, default="prompt", help="Column name for CSV")
    parser.add_argument("--sim_threshold", type=float, default=0.90)

    args = parser.parse_args()

    input_path = "/aul/homes/tnaya002/Desktop/lab/advpromptdetector/advGuard-main/data/baseline/baseline-prompt.csv"

    prompts = load_prompts(input_path, args.column)
    analyze_semantic_diversity(prompts, args.sim_threshold)


if __name__ == "__main__":
    main()