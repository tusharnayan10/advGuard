"""Evaluate AdvGuard's per-prompt and sequential leakage detection."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from advGuard import AdvGuard

# Keep the class import available for checkpoints serialized with torch.save.
from gnnTraining.train import GCN  # noqa: F401


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def load_prompts(path: str, column: str = "prompt") -> list[str]:
    suffix = Path(path).suffix.lower()
    if suffix == ".txt":
        with open(path, "r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
    elif suffix == ".csv":
        try:
            frame = pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception as exc:
            raise ValueError(f"Failed to read CSV {path}: {exc}") from exc
        if column not in frame.columns:
            raise ValueError(
                f"Column '{column}' not found in {path}; "
                f"available columns: {frame.columns.tolist()}"
            )
        prompts = (
            frame[column]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda values: values != ""]
            .tolist()
        )
    else:
        raise ValueError(f"Unsupported prompt file extension: {path}")

    print(f"[LOAD] {len(prompts)} prompts from {path} column='{column}'")
    return prompts


def build_stream(
    benign_prompts: list[str],
    attack_prompts: list[str],
    mode: str,
    seed: int,
    attack_start: int | None,
    attack_interval: int,
) -> list[tuple[str, str]]:
    """Build an evaluation stream while preserving attack order when requested."""

    rng = random.Random(seed)
    benign = list(benign_prompts)
    attacks = list(attack_prompts)

    if mode == "shuffled":
        stream = [("benign", prompt) for prompt in benign]
        stream.extend(("adv", prompt) for prompt in attacks)
        rng.shuffle(stream)
        return stream

    if mode == "burst":
        start = len(benign) // 2 if attack_start is None else attack_start
        if not 0 <= start <= len(benign):
            raise ValueError("--attack_start must be between 0 and benign count")
        return (
            [("benign", prompt) for prompt in benign[:start]]
            + [("adv", prompt) for prompt in attacks]
            + [("benign", prompt) for prompt in benign[start:]]
        )

    if mode == "interleaved":
        if attack_interval < 1:
            raise ValueError("--attack_interval must be at least 1")
        start = 0 if attack_start is None else attack_start
        if not 0 <= start <= len(benign):
            raise ValueError("--attack_start must be between 0 and benign count")

        stream = [("benign", prompt) for prompt in benign[:start]]
        benign_index = start
        for attack_prompt in attacks:
            stream.append(("adv", attack_prompt))
            stop = min(benign_index + attack_interval, len(benign))
            stream.extend(
                ("benign", prompt) for prompt in benign[benign_index:stop]
            )
            benign_index = stop
        stream.extend(("benign", prompt) for prompt in benign[benign_index:])
        return stream

    raise ValueError(f"Unsupported stream mode: {mode}")


def validate_model_metadata(args) -> dict:
    metadata_path = Path(str(args.model_path) + ".json")
    if not metadata_path.exists():
        print(
            f"[WARN] Model metadata not found at {metadata_path}; "
            "training/deployment configuration cannot be verified"
        )
        return {}

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    mismatches = []
    trained_percentile = metadata.get("threshold_percentile")
    if trained_percentile is not None and not np.isclose(
        float(trained_percentile), args.baseline_percentile
    ):
        mismatches.append(
            f"threshold percentile trained={trained_percentile}, "
            f"runtime={args.baseline_percentile}"
        )

    trained_graph_size = metadata.get("graph_size")
    if trained_graph_size is not None and int(trained_graph_size) != args.ttd:
        mismatches.append(
            f"graph size trained={trained_graph_size}, runtime TTD={args.ttd}"
        )

    if mismatches:
        message = "Model configuration mismatch: " + "; ".join(mismatches)
        if not args.allow_model_config_mismatch:
            raise ValueError(
                message + ". Pass --allow_model_config_mismatch only for an intentional experiment."
            )
        print(f"[WARN] {message}")
    else:
        print(f"[INFO] Model metadata validated from {metadata_path}")
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Run the AdvGuard evaluation pipeline")
    parser.add_argument("--baseline_file", required=True)
    parser.add_argument("--benign_file", required=True)
    parser.add_argument("--adv_file", required=True)
    parser.add_argument("--model_path", required=True)

    parser.add_argument("--baseline_column", default="prompt")
    parser.add_argument("--benign_column", default="prompt")
    parser.add_argument("--adv_column", default="prompt")

    parser.add_argument("--baseline_size", type=int, default=10_000)
    parser.add_argument("--baseline_percentile", type=float, default=90.0)
    parser.add_argument("--individual_threshold", type=float, default=0.80)
    parser.add_argument("--recent_window", type=int, default=500)
    parser.add_argument("--top_k_neighbors", type=int, default=1)
    parser.add_argument("--ttd", type=int, default=5)
    parser.add_argument(
        "--detection_interval",
        type=int,
        default=1,
        help="Run Grubbs+GCN every N prompts; 1 gives the lowest sequential TTD",
    )

    parser.add_argument(
        "--stream_mode",
        choices=("burst", "interleaved", "shuffled"),
        default="burst",
        help="Burst/interleaved preserve attack order; shuffled is legacy behavior",
    )
    parser.add_argument(
        "--attack_start",
        type=int,
        default=None,
        help="Benign prompts emitted before the ordered attack starts",
    )
    parser.add_argument(
        "--attack_interval",
        type=int,
        default=1,
        help="For interleaved mode, benign prompts between ordered attack prompts",
    )

    parser.add_argument("--json_output", action="store_true")
    parser.add_argument("--json_dir", default="output")
    parser.add_argument("--json_name", default="advguard_results.json")
    parser.add_argument("--allow_model_config_mismatch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.detection_interval < 1:
        raise ValueError("--detection_interval must be at least 1")
    if args.ttd < 1:
        raise ValueError("--ttd must be at least 1")

    random.seed(args.seed)
    validate_model_metadata(args)

    baseline_prompts = load_prompts(args.baseline_file, args.baseline_column)
    benign_prompts = load_prompts(args.benign_file, args.benign_column)
    attack_prompts = load_prompts(args.adv_file, args.adv_column)

    if len(baseline_prompts) < args.baseline_size:
        raise ValueError(
            f"Requested {args.baseline_size} baseline prompts, "
            f"but only {len(baseline_prompts)} were loaded"
        )
    baseline = baseline_prompts[:args.baseline_size]

    stream = build_stream(
        benign_prompts,
        attack_prompts,
        args.stream_mode,
        args.seed,
        args.attack_start,
        args.attack_interval,
    )
    attack_positions = {
        index for index, (label, _) in enumerate(stream) if label == "adv"
    }
    print(
        f"[INFO] stream mode={args.stream_mode}, total={len(stream)}, "
        f"benign={len(stream) - len(attack_positions)}, attacks={len(attack_positions)}"
    )

    adv_guard = AdvGuard(
        baseline_prompts=baseline,
        model_path=args.model_path,
        ttd=args.ttd,
        baseline_percentile=args.baseline_percentile,
        individual_threshold=args.individual_threshold,
        recent_window=args.recent_window,
        top_k_neighbors=args.top_k_neighbors,
    )

    detected_positions: set[int] = set()
    individual_positions: set[int] = set()
    sequential_positions: set[int] = set()
    detection_time: dict[int, int] = {}
    events = []

    def record_sequential_detections(step_index: int, anomaly_subgraphs) -> None:
        for component_index, subgraph in enumerate(anomaly_subgraphs):
            newly_detected = []
            component_positions = []
            query_nodes = 0

            for node_id, node_data in subgraph.graph.nodes(data=True):
                stream_position = adv_guard.node_to_stream_idx.get(node_id)
                if stream_position is None:
                    continue
                query_nodes += 1
                component_positions.append(stream_position)
                sequential_positions.add(stream_position)
                detected_positions.add(stream_position)
                if stream_position not in detection_time:
                    detection_time[stream_position] = step_index
                    label, prompt = stream[stream_position]
                    newly_detected.append(
                        {
                            "stream_idx": stream_position,
                            "label": label,
                            "prompt": prompt,
                            "detection_delay": step_index - stream_position,
                        }
                    )

            if newly_detected:
                events.append(
                    {
                        "type": "sequential",
                        "detection_step": step_index,
                        "component_index": component_index,
                        "component_nodes": subgraph.node_nums,
                        "component_query_nodes": query_nodes,
                        "component_score": subgraph.GetGraphScore(),
                        "component_stream_positions": sorted(component_positions),
                        "newly_detected": newly_detected,
                    }
                )
                print(
                    f"\n[BLOCKED-SEQUENTIAL] step={step_index} "
                    f"new={len(newly_detected)} query_nodes={query_nodes} "
                    f"score={subgraph.GetGraphScore():.4f}"
                )

    progress = tqdm(
        enumerate(stream),
        total=len(stream),
        desc="Processing stream",
        ncols=110,
    )
    for index, (label, prompt) in progress:
        similarity, nearest_node, flagged, injection_score = adv_guard.add(
            prompt=prompt,
            source=label,
            stream_idx=index,
        )

        if flagged:
            detected_positions.add(index)
            individual_positions.add(index)
            detection_time.setdefault(index, index)
            events.append(
                {
                    "type": "individual",
                    "detection_step": index,
                    "stream_idx": index,
                    "label": label,
                    "prompt": prompt,
                    "injection_score": injection_score,
                    "nearest_node": nearest_node,
                    "nearest_similarity": similarity,
                    "detection_delay": 0,
                }
            )
            print(
                f"\n[BLOCKED-INDIVIDUAL] idx={index} label={label} "
                f"inj={injection_score:.4f} sim={similarity:.4f}"
            )

        if (index + 1) % args.detection_interval == 0:
            record_sequential_detections(index, adv_guard.detector())

        if index % 10 == 0:
            progress.set_postfix(
                processed=index + 1,
                individual=len(individual_positions),
                sequential=len(sequential_positions),
                blocked=len(detected_positions),
            )

    # Ensure the final partial interval is evaluated.
    if stream and len(stream) % args.detection_interval != 0:
        record_sequential_detections(len(stream) - 1, adv_guard.detector())

    true_positives = detected_positions & attack_positions
    false_positives = detected_positions - attack_positions
    false_negatives = attack_positions - detected_positions
    benign_positions = set(range(len(stream))) - attack_positions
    true_negatives = benign_positions - detected_positions

    precision = (
        len(true_positives) / (len(true_positives) + len(false_positives))
        if true_positives or false_positives
        else 0.0
    )
    recall = (
        len(true_positives) / (len(true_positives) + len(false_negatives))
        if true_positives or false_negatives
        else 0.0
    )
    specificity = (
        len(true_negatives) / (len(true_negatives) + len(false_positives))
        if true_negatives or false_positives
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    balanced_accuracy = (recall + specificity) / 2.0
    false_positive_rate = 1.0 - specificity

    detected_attack_positions = sorted(true_positives)
    delays = np.asarray(
        [detection_time[position] - position for position in detected_attack_positions],
        dtype=float,
    )
    first_attack_position = min(attack_positions) if attack_positions else None
    first_attack_detection_step = (
        min(detection_time[position] for position in true_positives)
        if true_positives
        else None
    )
    first_detection_ttd = (
        first_attack_detection_step - first_attack_position + 1
        if first_attack_position is not None and first_attack_detection_step is not None
        else None
    )
    attacks_seen_before_first_detection = (
        sum(position <= first_attack_detection_step for position in attack_positions)
        if first_attack_detection_step is not None
        else None
    )

    metrics = {
        "processed": len(stream),
        "attacks": len(attack_positions),
        "tp": len(true_positives),
        "fp": len(false_positives),
        "fn": len(false_negatives),
        "tn": len(true_negatives),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": false_positive_rate,
        "individual_detections": len(individual_positions),
        "sequential_detections": len(sequential_positions),
        "first_detection_ttd": first_detection_ttd,
        "attacks_seen_before_first_detection": attacks_seen_before_first_detection,
        "median_detection_delay": float(np.median(delays)) if len(delays) else None,
        "p90_detection_delay": float(np.percentile(delays, 90)) if len(delays) else None,
        "max_detection_delay": float(delays.max()) if len(delays) else None,
    }

    print("\n========================================")
    print(f"Processed: {metrics['processed']} | attacks: {metrics['attacks']}")
    print(
        f"TP: {metrics['tp']} | FP: {metrics['fp']} | "
        f"FN: {metrics['fn']} | TN: {metrics['tn']}"
    )
    print(
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | "
        f"Specificity: {specificity:.4f}"
    )
    print(
        f"F1: {f1:.4f} | Balanced accuracy: {balanced_accuracy:.4f} | "
        f"FPR: {100 * false_positive_rate:.4f}%"
    )
    print(
        f"First-detection TTD: {first_detection_ttd} | "
        f"attacks seen before first detection: {attacks_seen_before_first_detection}"
    )
    print(
        f"Detection delay median={metrics['median_detection_delay']} | "
        f"p90={metrics['p90_detection_delay']} | max={metrics['max_detection_delay']}"
    )
    print("========================================\n")

    if args.json_output:
        os.makedirs(args.json_dir, exist_ok=True)
        output_path = Path(args.json_dir) / args.json_name
        result = {
            "configuration": vars(args),
            "metrics": metrics,
            "events": events,
            "detected_attack_positions": detected_attack_positions,
            "false_positive_positions": sorted(false_positives),
            "missed_attack_positions": sorted(false_negatives),
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, cls=NumpyEncoder)
        print(f"[SAVE] results={output_path}")


if __name__ == "__main__":
    main()
