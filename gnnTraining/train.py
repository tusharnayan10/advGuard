"""Train AdvGuard's line-graph GCN on deployment-matched provenance graphs.

The online detector links each incoming prompt to at most one previous prompt:
its nearest semantic neighbour, provided the cosine similarity exceeds a
threshold estimated from benign baseline traffic.  This trainer deliberately
uses the same construction.  It then converts candidate connected components
to line graphs, where each original edge becomes a node whose feature is PAS
(the cosine-similarity score).

Training samples follow the paper's intended cadence:

* up to ``attack_sequences_per_source`` attack sequences are selected from
  every attack source/algorithm and snapshotted every ``graph_size`` queries;
* benign prompts are shuffled and processed in independent blocks of
  ``benign_snapshot_interval`` queries; the full provenance graph is saved for
  each block, as described in the paper, and unusually large benign components
  may additionally be retained as hard negatives;
* sequence/block groups are kept intact across train, validation, and test
  sets so overlapping snapshots cannot leak across splits.

Unlike the paper's description, the test set is not used for checkpoint
selection.  A validation split selects the checkpoint and the test set is
evaluated once at the end.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from torch_geometric.utils import from_networkx


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class PromptSequence:
    source: str
    sequence_id: str
    prompts: list[str]


@dataclass
class GraphSample:
    data: Data
    label: int
    group_id: str
    source: str


class GCN(torch.nn.Module):
    """Three-layer graph classifier described by the AdvGuard paper."""

    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # Mean pooling alone can erase the structural difference between a
        # weakly aggregated benign forest and a tightly aggregated attack
        # component. Mean + max preserves both the overall response and the
        # strongest local structural response while retaining one linear head.
        self.lin = Linear(2 * hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        graph_embedding = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
        )
        graph_embedding = F.dropout(
            graph_embedding, p=0.5, training=self.training
        )
        return self.lin(graph_embedding)


def _clean_prompts(values: Iterable[object]) -> list[str]:
    prompts = []
    for value in values:
        if pd.isna(value):
            continue
        prompt = str(value).strip()
        if prompt:
            prompts.append(prompt)
    return prompts


def load_prompt_sequences(
    paths: Sequence[str],
    prompt_column: str,
    sequence_column: str | None,
) -> list[PromptSequence]:
    """Load TXT/CSV files, preserving explicit CSV sequence boundaries."""

    sequences: list[PromptSequence] = []
    for raw_path in paths:
        path = Path(raw_path)
        # Repository attack files usually follow
        # ``.../<attack algorithm>/<target model>/<file>``.  Grouping by the
        # grandparent therefore implements "five sequences per algorithm" and
        # the full path keeps sequence identifiers unique when many files are
        # all named e.g. ``prompt-2k.txt``.
        if "3attack_prompt" in path.parts:
            attack_root_index = path.parts.index("3attack_prompt")
            source = path.parts[attack_root_index + 1]
        else:
            source = path.parent.parent.name or path.parent.name or path.stem
        file_id = path.as_posix()
        suffix = path.suffix.lower()

        if suffix == ".txt":
            with path.open("r", encoding="utf-8") as handle:
                prompts = _clean_prompts(handle)
            sequences.append(PromptSequence(source, f"{file_id}:0", prompts))
            continue

        if suffix != ".csv":
            raise ValueError(f"Unsupported prompt file: {path}")

        frame = pd.read_csv(path, engine="python", on_bad_lines="skip")
        if prompt_column not in frame.columns:
            raise ValueError(
                f"Column '{prompt_column}' not found in {path}; "
                f"available columns: {frame.columns.tolist()}"
            )

        if sequence_column:
            if sequence_column not in frame.columns:
                raise ValueError(
                    f"Sequence column '{sequence_column}' not found in {path}"
                )
            for sequence_id, group in frame.groupby(sequence_column, sort=False):
                prompts = _clean_prompts(group[prompt_column])
                if prompts:
                    sequences.append(
                        PromptSequence(source, f"{file_id}:{sequence_id}", prompts)
                    )
        else:
            prompts = _clean_prompts(frame[prompt_column])
            sequences.append(PromptSequence(source, f"{file_id}:0", prompts))

    return [sequence for sequence in sequences if sequence.prompts]


def load_flat_prompts(paths: Sequence[str], prompt_column: str) -> list[str]:
    return [
        prompt
        for sequence in load_prompt_sequences(paths, prompt_column, None)
        for prompt in sequence.prompts
    ]


def encode_prompts(
    encoder: SentenceTransformer,
    prompts: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    embeddings = encoder.encode(
        list(prompts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-8, None)


def baseline_similarity_threshold(
    baseline_embeddings: np.ndarray,
    percentile: float,
    block_size: int = 1024,
) -> float:
    """Match AdvGuard's percentile of each baseline node's nearest neighbour."""

    count = len(baseline_embeddings)
    if count < 2:
        raise ValueError("At least two baseline prompts are required")

    nearest = np.full(count, -np.inf, dtype=np.float32)
    for start in range(0, count, block_size):
        stop = min(start + block_size, count)
        similarities = baseline_embeddings[start:stop] @ baseline_embeddings.T
        row_ids = np.arange(stop - start)
        similarities[row_ids, np.arange(start, stop)] = -np.inf
        nearest[start:stop] = similarities.max(axis=1)

    return float(np.percentile(nearest, percentile))


class ProvenanceGraphBuilder:
    """Incremental one-nearest-neighbour graph used by online AdvGuard."""

    def __init__(self, baseline_embeddings: np.ndarray, threshold: float):
        self.baseline_embeddings = baseline_embeddings
        self.threshold = threshold
        self.stream_embeddings: list[np.ndarray] = []
        self.graph = nx.Graph()

    def add(self, embedding: np.ndarray) -> int:
        stream_index = len(self.stream_embeddings)
        node_id = ("query", stream_index)
        self.graph.add_node(node_id, is_baseline=False)

        baseline_similarities = self.baseline_embeddings @ embedding
        baseline_index = int(np.argmax(baseline_similarities))
        best_similarity = float(baseline_similarities[baseline_index])
        nearest_node: tuple[str, int] = ("baseline", baseline_index)

        if self.stream_embeddings:
            stream_matrix = np.vstack(self.stream_embeddings)
            stream_similarities = stream_matrix @ embedding
            nearest_stream_index = int(np.argmax(stream_similarities))
            stream_similarity = float(stream_similarities[nearest_stream_index])
            if stream_similarity > best_similarity:
                best_similarity = stream_similarity
                nearest_node = ("query", nearest_stream_index)

        self.stream_embeddings.append(embedding)

        if best_similarity > self.threshold:
            if nearest_node[0] == "baseline":
                self.graph.add_node(nearest_node, is_baseline=True)
            self.graph.add_edge(nearest_node, node_id, label=best_similarity)

        return stream_index

    def candidate_components(self, minimum_nodes: int) -> list[nx.Graph]:
        components = []
        for nodes in nx.connected_components(self.graph):
            query_count = sum(node[0] == "query" for node in nodes)
            if query_count >= minimum_nodes:
                components.append(self.graph.subgraph(nodes).copy())
        return components


def graph_to_pyg_line_graph(graph: nx.Graph, label: int) -> Data:
    """Turn edge PAS values into line-graph node features."""

    if graph.number_of_edges() == 0:
        return Data(
            x=torch.zeros((1, 1), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            y=torch.tensor([label], dtype=torch.long),
        )

    line_graph = nx.line_graph(graph)
    for original_edge in line_graph.nodes:
        u, v = original_edge
        line_graph.nodes[original_edge]["pas"] = float(graph.edges[u, v]["label"])

    data = from_networkx(line_graph, group_node_attrs=["pas"])
    data.x = data.x.float().reshape(-1, 1)
    data.y = torch.tensor([label], dtype=torch.long)
    return data


def _split_flat_sequence(
    sequence: PromptSequence,
    desired_sequences: int,
    minimum_length: int,
) -> list[PromptSequence]:
    """Create independent contiguous sequences when a file has no sequence IDs."""

    if desired_sequences <= 1 or len(sequence.prompts) < 2 * minimum_length:
        return [sequence]

    usable_count = min(desired_sequences, len(sequence.prompts) // minimum_length)
    chunks = np.array_split(np.asarray(sequence.prompts, dtype=object), usable_count)
    return [
        PromptSequence(
            sequence.source,
            f"{sequence.sequence_id}:chunk-{index}",
            [str(prompt) for prompt in chunk.tolist()],
        )
        for index, chunk in enumerate(chunks)
        if len(chunk) >= minimum_length
    ]


def select_attack_sequences(
    sequences: Sequence[PromptSequence],
    per_source: int,
    graph_size: int,
    rng: random.Random,
) -> list[PromptSequence]:
    by_source: dict[str, list[PromptSequence]] = defaultdict(list)
    for sequence in sequences:
        by_source[sequence.source].append(sequence)

    selected = []
    for source, source_sequences in by_source.items():
        expanded = source_sequences
        if len(source_sequences) == 1:
            expanded = _split_flat_sequence(source_sequences[0], per_source, graph_size)
        rng.shuffle(expanded)
        selected.extend(expanded[:per_source])
        print(f"[DATA] attack source={source}: selected {min(per_source, len(expanded))} sequences")
    return selected


def build_attack_samples(
    sequences: Sequence[PromptSequence],
    sequence_embeddings: dict[str, np.ndarray],
    baseline_embeddings: np.ndarray,
    threshold: float,
    graph_size: int,
) -> list[GraphSample]:
    samples = []
    for sequence in sequences:
        builder = ProvenanceGraphBuilder(baseline_embeddings, threshold)
        for index, embedding in enumerate(sequence_embeddings[sequence.sequence_id]):
            builder.add(embedding)
            if (index + 1) % graph_size != 0:
                continue
            # The paper saves the query-provenance graph every s queries.  Do
            # not require all s queries to fall in one connected component:
            # that would preferentially keep only the easiest, most strongly
            # clustered attacks.
            snapshot = builder.graph.copy()
            samples.append(
                GraphSample(
                    graph_to_pyg_line_graph(snapshot, label=1),
                    label=1,
                    group_id=f"attack:{sequence.sequence_id}",
                    source=sequence.source,
                )
            )
    return samples


def build_benign_samples(
    embeddings: np.ndarray,
    baseline_embeddings: np.ndarray,
    threshold: float,
    graph_size: int,
    snapshot_interval: int,
    components_per_snapshot: int,
) -> list[GraphSample]:
    samples = []
    snapshot_count = 0
    hard_negative_count = 0
    for block_index, start in enumerate(range(0, len(embeddings), snapshot_interval)):
        block = embeddings[start:start + snapshot_interval]
        if len(block) < graph_size:
            continue
        builder = ProvenanceGraphBuilder(baseline_embeddings, threshold)
        for embedding in block:
            builder.add(embedding)

        # Benign queries are intentionally weakly aggregated. Requiring a
        # connected component of size s can therefore eliminate the entire
        # negative class. The paper instead saves one graph for every 500
        # benign queries, so the complete (possibly disconnected) provenance
        # graph is the primary negative sample.
        group_id = f"benign:block-{block_index}"
        samples.append(
            GraphSample(
                graph_to_pyg_line_graph(builder.graph.copy(), label=0),
                label=0,
                group_id=group_id,
                source="benign-snapshot",
            )
        )
        snapshot_count += 1

        # Large benign components are precisely the random aggregations the
        # GCN should learn to reject, so keep a bounded number as additional
        # hard negatives when they exist.
        candidates = builder.candidate_components(graph_size)
        candidates.sort(key=lambda graph: graph.number_of_nodes(), reverse=True)
        for component in candidates[:components_per_snapshot]:
            samples.append(
                GraphSample(
                    graph_to_pyg_line_graph(component, label=0),
                    label=0,
                    group_id=group_id,
                    source="benign-hard-negative",
                )
            )
            hard_negative_count += 1
    print(
        f"[DATA] benign snapshots={snapshot_count}, "
        f"hard-negative components={hard_negative_count}"
    )
    return samples


def split_samples_by_group(
    samples: Sequence[GraphSample],
    seed: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[list[GraphSample], list[GraphSample], list[GraphSample]]:
    """Split groups within each class to prevent snapshot leakage."""

    rng = random.Random(seed)
    output = {"train": [], "validation": [], "test": []}

    for label in (0, 1):
        groups: dict[str, list[GraphSample]] = defaultdict(list)
        for sample in samples:
            if sample.label == label:
                groups[sample.group_id].append(sample)

        group_ids = list(groups)
        rng.shuffle(group_ids)
        if len(group_ids) < 3:
            raise ValueError(
                f"Class {label} has only {len(group_ids)} independent groups; "
                "at least three are needed for leakage-safe train/validation/test splits"
            )

        train_end = max(1, int(round(len(group_ids) * train_ratio)))
        validation_count = max(1, int(round(len(group_ids) * validation_ratio)))
        train_end = min(train_end, len(group_ids) - 2)
        validation_end = min(train_end + validation_count, len(group_ids) - 1)

        assignments = {
            "train": group_ids[:train_end],
            "validation": group_ids[train_end:validation_end],
            "test": group_ids[validation_end:],
        }
        for split_name, ids in assignments.items():
            for group_id in ids:
                output[split_name].extend(groups[group_id])

    for split_samples in output.values():
        rng.shuffle(split_samples)
    return output["train"], output["validation"], output["test"]


def _classification_counts(logits: torch.Tensor, labels: torch.Tensor):
    predictions = logits.argmax(dim=1)
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    correct = int((predictions == labels).sum())
    return correct, tp, fp, fn, tn


def _metrics(
    loss_sum: float,
    total: int,
    correct: int,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_f1 = (
        2 * negative_precision * specificity / (negative_precision + specificity)
        if negative_precision + specificity
        else 0.0
    )
    return {
        "loss": loss_sum / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "macro_f1": (f1 + negative_f1) / 2.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = total = correct = tp = fp = fn = tn = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(DEVICE)
            if training:
                optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = batch.num_graphs
            (
                batch_correct,
                batch_tp,
                batch_fp,
                batch_fn,
                batch_tn,
            ) = _classification_counts(logits, batch.y)
            loss_sum += float(loss.item()) * batch_size
            total += batch_size
            correct += batch_correct
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            tn += batch_tn

    return _metrics(loss_sum, total, correct, tp, fp, fn, tn)


def describe_samples(name: str, samples: Sequence[GraphSample]) -> None:
    benign = sum(sample.label == 0 for sample in samples)
    attack = len(samples) - benign
    groups = len({sample.group_id for sample in samples})
    print(f"[SPLIT] {name}: graphs={len(samples)}, benign={benign}, attack={attack}, groups={groups}")


def describe_graph_features(samples: Sequence[GraphSample]) -> None:
    """Print compact diagnostics to expose indistinguishable graph classes."""

    for label, label_name in ((0, "benign"), (1, "attack")):
        selected = [sample.data for sample in samples if sample.label == label]
        line_nodes = np.asarray([data.num_nodes for data in selected], dtype=float)
        line_edges = np.asarray([data.num_edges for data in selected], dtype=float)
        pas_means = np.asarray(
            [float(data.x.mean()) for data in selected], dtype=float
        )
        dummy_count = sum(float(data.x.abs().sum()) == 0.0 for data in selected)
        print(
            f"[GRAPH] {label_name}: count={len(selected)}, "
            f"line_nodes median={np.median(line_nodes):.1f} "
            f"p10={np.percentile(line_nodes, 10):.1f} "
            f"p90={np.percentile(line_nodes, 90):.1f}, "
            f"line_edges median={np.median(line_edges):.1f}, "
            f"PAS mean={pas_means.mean():.4f}, dummy={dummy_count}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AdvGuard's deployment-matched line-graph GCN"
    )
    parser.add_argument("--baseline_path", nargs="+", required=True)
    parser.add_argument("--benign_path", nargs="+", required=True)
    parser.add_argument("--adv_path", nargs="+", required=True)
    parser.add_argument("--baseline_column", default="prompt")
    parser.add_argument("--benign_column", default="prompt")
    parser.add_argument("--adv_column", default="prompt")
    parser.add_argument(
        "--attack_sequence_column",
        default=None,
        help="Optional CSV column identifying independent attack sequences",
    )
    parser.add_argument("--baseline_size", type=int, default=10_000)
    parser.add_argument("--threshold_percentile", type=float, default=80.0)
    parser.add_argument(
        "--graph_size",
        type=int,
        default=10,
        help="Paper parameter s and online detector TTD/minimum component size",
    )
    parser.add_argument("--attack_sequences_per_source", type=int, default=5)
    parser.add_argument("--benign_snapshot_interval", type=int, default=500)
    parser.add_argument("--benign_components_per_snapshot", type=int, default=5)
    parser.add_argument("--embedding_batch_size", type=int, default=128)
    parser.add_argument("--encoder_name", default="all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--model_out", default="gnnTraining/model/gcn_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.graph_size < 2:
        raise ValueError("--graph_size must be at least 2")
    if args.benign_snapshot_interval < args.graph_size:
        raise ValueError("--benign_snapshot_interval must be >= --graph_size")

    set_seed(args.seed)
    rng = random.Random(args.seed)
    print(f"[INFO] device={DEVICE}")

    baseline_prompts = load_flat_prompts(args.baseline_path, args.baseline_column)
    if len(baseline_prompts) < args.baseline_size:
        raise ValueError(
            f"Requested {args.baseline_size} baseline prompts, found {len(baseline_prompts)}"
        )
    baseline_prompts = baseline_prompts[:args.baseline_size]

    benign_prompts = load_flat_prompts(args.benign_path, args.benign_column)
    rng.shuffle(benign_prompts)
    attack_sequences = load_prompt_sequences(
        args.adv_path, args.adv_column, args.attack_sequence_column
    )
    attack_sequences = select_attack_sequences(
        attack_sequences,
        args.attack_sequences_per_source,
        args.graph_size,
        rng,
    )

    print(f"[INFO] loading encoder={args.encoder_name}")
    encoder = SentenceTransformer(args.encoder_name)
    baseline_embeddings = encode_prompts(
        encoder, baseline_prompts, args.embedding_batch_size
    )
    threshold = baseline_similarity_threshold(
        baseline_embeddings, args.threshold_percentile
    )
    print(f"[INFO] deployment similarity threshold={threshold:.6f}")

    benign_embeddings = encode_prompts(
        encoder, benign_prompts, args.embedding_batch_size
    )
    sequence_embeddings = {
        sequence.sequence_id: encode_prompts(
            encoder, sequence.prompts, args.embedding_batch_size
        )
        for sequence in attack_sequences
    }

    attack_samples = build_attack_samples(
        attack_sequences,
        sequence_embeddings,
        baseline_embeddings,
        threshold,
        args.graph_size,
    )
    benign_samples = build_benign_samples(
        benign_embeddings,
        baseline_embeddings,
        threshold,
        args.graph_size,
        args.benign_snapshot_interval,
        args.benign_components_per_snapshot,
    )
    samples = benign_samples + attack_samples
    print(f"[DATA] generated benign={len(benign_samples)}, attack={len(attack_samples)} graphs")
    if not benign_samples or not attack_samples:
        raise ValueError(
            "Graph generation produced an empty class. Add more data, lower "
            "--graph_size, or inspect the learned threshold."
        )
    describe_graph_features(samples)

    train_samples, validation_samples, test_samples = split_samples_by_group(
        samples, args.seed
    )
    describe_samples("train", train_samples)
    describe_samples("validation", validation_samples)
    describe_samples("test", test_samples)

    train_loader = DataLoader(
        [sample.data for sample in train_samples],
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        [sample.data for sample in validation_samples],
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        [sample.data for sample in test_samples],
        batch_size=args.batch_size,
        shuffle=False,
    )

    class_counts = np.bincount([sample.label for sample in train_samples], minlength=2)
    class_weights = len(train_samples) / (2.0 * np.maximum(class_counts, 1))
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
    )
    model = GCN(1, args.hidden_channels, 2).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    best_balanced_accuracy = -1.0
    best_validation_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        validation_metrics = run_epoch(model, validation_loader, criterion)
        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} f1={train_metrics['f1']:.4f} | "
            f"val loss={validation_metrics['loss']:.4f} "
            f"acc={validation_metrics['accuracy']:.4f} "
            f"precision={validation_metrics['precision']:.4f} "
            f"recall={validation_metrics['recall']:.4f} "
            f"specificity={validation_metrics['specificity']:.4f} "
            f"balanced_acc={validation_metrics['balanced_accuracy']:.4f} "
            f"macro_f1={validation_metrics['macro_f1']:.4f}"
        )

        balanced_accuracy = validation_metrics["balanced_accuracy"]
        validation_loss = validation_metrics["loss"]
        improved = balanced_accuracy > best_balanced_accuracy + 1e-6
        tied_but_lower_loss = (
            abs(balanced_accuracy - best_balanced_accuracy) <= 1e-6
            and validation_loss < best_validation_loss - 1e-6
        )
        if improved or tied_but_lower_loss:
            best_balanced_accuracy = balanced_accuracy
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[INFO] early stopping after {epoch} epochs")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, criterion)
    print("[TEST] " + ", ".join(f"{key}={value:.4f}" for key, value in test_metrics.items()))

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the serialized-module format expected by advGuard.py.
    torch.save(model.cpu(), model_path)

    metadata = {
        "encoder_name": args.encoder_name,
        "threshold": threshold,
        "threshold_percentile": args.threshold_percentile,
        "baseline_size": args.baseline_size,
        "graph_size": args.graph_size,
        "hidden_channels": args.hidden_channels,
        "best_validation_balanced_accuracy": best_balanced_accuracy,
        "best_validation_loss": best_validation_loss,
        "test_metrics": test_metrics,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "seed": args.seed,
    }
    metadata_path = model_path.with_suffix(model_path.suffix + ".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[SAVE] model={model_path}")
    print(f"[SAVE] metadata={metadata_path}")


if __name__ == "__main__":
    main()
