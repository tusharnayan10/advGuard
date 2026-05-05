import argparse
import os
import random
import json
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
from tqdm import tqdm
import numpy as np

from advGuard import AdvGuard
# from test import AdvGuard
from gnnTraining.train import GCN

MAX_JSON_FILES = 3


# DATA LOADING
def load_prompts(path: str, column: str = "prompt"):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"[LOAD] {len(prompts)} prompts from TXT {path}")
        return prompts

    elif ext == ".csv":
        try:
            df = pd.read_csv(
                path,
                engine="python",      
                on_bad_lines="skip"   
            )
        except Exception as e:
            raise ValueError(f"Failed to read CSV {path}: {e}")

        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found in {path}. "
                f"Available: {df.columns.tolist()}"
            )

        prompts = (
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

        print(f"[LOAD] {len(prompts)} prompts from CSV {path} column='{column}'")
        return prompts

    else:
        raise ValueError(f"Unsupported file extension for {path}")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# MAIN PIPELINE
def main():

    parser = argparse.ArgumentParser(description="Run AdvGuard pipeline")

    parser.add_argument("--baseline_file", type=str, required=True,
                        help="Path to baseline prompts")
    parser.add_argument("--benign_file", type=str, required=True,
                        help="Path to benign prompts")
    parser.add_argument("--adv_file", type=str, required=True,
                        help="Path to adversarial prompts")

    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--benign_column", type=str, default="prompt")
    parser.add_argument("--adv_column", type=str, default="prompt")

    parser.add_argument("--baseline_size", type=int, default=100)
    parser.add_argument("--ttd", type=int, default=5)
    parser.add_argument("--detection_interval", type=int, default=50)

    parser.add_argument("--json_output", action="store_true")
    parser.add_argument("--json_dir", type=str, default="blocked_json")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    # LOAD DATA
    baseline_prompts = load_prompts(args.baseline_file, args.benign_column)
    benign_prompts = load_prompts(args.benign_file, args.benign_column)
    adv_prompts = load_prompts(args.adv_file, args.adv_column)

    print(f"[INFO] total baseline={len(baseline_prompts)}, "
          f"total benign={len(benign_prompts)}, total adv={len(adv_prompts)}")

    if len(baseline_prompts) < args.baseline_size:
        raise ValueError("baseline_file must have enough prompts")

    baseline = baseline_prompts[:args.baseline_size]
    benign_rest = benign_prompts

    # INIT ADVGUARD
    adv_guard = AdvGuard(
        baseline_prompts=baseline,
        model_path=args.model_path,
        ttd=args.ttd,
    )

    # BUILD STREAM
    stream = []

    for p in benign_rest:
        stream.append(("benign", p))

    for p in adv_prompts:
        stream.append(("adv", p))

    random.shuffle(stream)

    inserted_positions = [
        i for i, (label, _) in enumerate(stream) if label == "adv"
    ]

    print(f"[INFO] Stream size={len(stream)}, adv count={len(inserted_positions)}")

    # JSON SETUP
    if args.json_output:
        os.makedirs(args.json_dir, exist_ok=True)

    json_buffer = []
    saved_jsons = []
    FLUSH_INTERVAL = 100  # tune this if needed

    detected_stream_positions = set()
    attack_counter = 0

    # STREAM PROCESSING
    qry_stream = tqdm(
        enumerate(stream),
        total=len(stream),
        desc="Processing Stream",
        ncols=100
    )

    for idx, (label, prompt) in qry_stream:

        if label == "adv":
            attack_counter += 1

        sim, nearest_node, flagged, inj_score = adv_guard.add(
            prompt=prompt,
            source=label,
            stream_idx=idx,
        )

        # LOCAL DETECTION
        if flagged:
            detected_stream_positions.add(idx)

            print(f"\n BLOCKED idx={idx} | label={label}")
            print(f"prompt={prompt[:10]}")
            # print(f"sim={sim:.4f} | inj_score={inj_score:.4f}")

        # GLOBAL DETECTION
        if (idx + 1) % args.detection_interval == 0:

            anomaly_subgraphs = adv_guard.detector()

            for sg_id, sg in enumerate(anomaly_subgraphs):

                blocked_entries = []

                for node_id in sg.graph.nodes():
                    stream_pos = adv_guard.node_to_stream_idx.get(node_id, None)

                    if stream_pos is not None:
                        detected_stream_positions.add(stream_pos)

                        blocked_label, blocked_prompt = stream[stream_pos]

                        print(f"\n[BLOCKED-GLOBAL] idx={stream_pos} | label={blocked_label}")
                        print(f"prompt={blocked_prompt[:200]}")

                        blocked_entries.append({
                            "stream_idx": stream_pos,
                            "label": blocked_label,
                            "prompt": blocked_prompt
                        })

                if args.json_output and blocked_entries:

                    meta = {
                        "step_index": idx,
                        "component_id": sg_id,
                        "num_nodes": sg.node_nums,
                        "graph_score": sg.GetGraphScore(),
                    }

                    json_path = os.path.join(
                        args.json_dir,
                        f"blocked_step{idx}_comp{sg_id}.json"
                    )

                    json_buffer.append({
                        "path": json_path,
                        "data": {
                            "meta": meta,
                            "blocked": blocked_entries
                        }
                    })

                    print(f"[JSON-BUFFERED] {json_path}")

 
        if args.json_output and len(json_buffer) >= FLUSH_INTERVAL:

            for item in json_buffer:
                with open(item["path"], "w", encoding="utf-8") as f:
                    json.dump(item["data"], f, indent=2)

                saved_jsons.append(item["path"])

            json_buffer.clear()

            while len(saved_jsons) > MAX_JSON_FILES:
                old_file = saved_jsons.pop(0)
                if os.path.exists(old_file):
                    os.remove(old_file)

    
        if idx % 10 == 0:
            qry_stream.set_postfix({
                "processed": idx + 1,
                "blocked": len(detected_stream_positions)
                # "attacks_seen": attack_counter
            })


    if args.json_output and json_buffer:
        for item in json_buffer:
            with open(item["path"], "w", encoding="utf-8") as f:
                json.dump(item["data"], f, indent=2, cls=NumpyEncoder)
    # METRICS
    inserted_positions_set = set(inserted_positions)

    TP = detected_stream_positions & inserted_positions_set
    FP = detected_stream_positions - inserted_positions_set
    FN = inserted_positions_set - detected_stream_positions

    total_processed = len(stream)
    total_attacks = len(inserted_positions)

    negatives = total_processed - total_attacks
    TN = negatives - len(FP)

    fpr = (len(FP) / negatives) if negatives else 0.0
    fpr_percent = fpr * 100

    precision = len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) else 0.0
    recall = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("\n====================================")
    print(f"Processed: {total_processed}")
    print(f"Inserted attacks: {total_attacks}")
    print(f"TP: {len(TP)} | FP: {len(FP)} | FN: {len(FN)} | TN: {TN}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"FPR (%): {fpr_percent:.2f}")
    print("====================================\n")


if __name__ == "__main__":
    main()