#!/usr/bin/env python3
"""Build a *proxy* session CSV for experiments/gcn_study.py from two prompt lists.

Usage (from the repository root):
    python experiments/make_proxy_session_input.py

The source files do not contain conversation/session IDs. SST-2 contains standalone
movie-review sentences, so benign sessions below are synthetic editing trajectories
anchored on one sentence. Attack rows are non-overlapping chunks of the prompt list,
not verified independent attacks. Some attack rows include target/system context.
The output exercises the GCN study code but is NOT a valid paper evaluation.
"""

import argparse
import csv
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENIGN_SOURCE = ROOT / "data/2benign/2sst2/sst2_test_sentences-1k.csv"
ATTACK_SOURCE = ROOT / "data/3attack_prompt/Prompt-Extraction/Llama-3.3-70B-Instruct/prompt-2k.csv"
DEFAULT_OUTPUT = ROOT / "experiments/data/sst2_prompt_extraction_proxy_sessions.csv"

EDIT_STEPS = (
    "Rewrite this review more clearly.",
    "Make this review more concise.",
    "Keep the meaning but change the opening.",
    "Use a more neutral tone.",
    "Make the final clause easier to read.",
    "Correct punctuation without changing the opinion.",
    "Turn this into a short headline.",
    "Expand this into two sentences.",
    "Remove unnecessary adjectives.",
    "Make the wording more conversational.",
    "Use a more formal style.",
    "Preserve the sentiment while varying the syntax.",
    "Explain the same opinion for a general audience.",
    "Shorten the opening phrase.",
    "Make the ending more direct.",
    "Replace vague wording with specific wording.",
    "Improve the sentence flow.",
    "Keep the review brief and polished.",
    "Rewrite this as a one-sentence summary.",
    "Check the review for repetition.",
    "Make the tone less emphatic.",
    "Make the tone slightly warmer.",
    "Clarify what the reviewer liked or disliked.",
    "Rephrase the opinion without adding facts.",
    "Make the first half shorter.",
    "Make the second half more precise.",
    "Use plain language throughout.",
    "Produce a polished final revision.",
    "Summarize the final review in one sentence.",
    "Give one alternative final wording.",
)


def load_unique_prompts(path):
    seen, prompts = set(), []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if "prompt" not in (reader.fieldnames or []):
            raise ValueError(f"Missing 'prompt' column in {path}")
        for source_row, row in enumerate(reader, start=2):
            prompt = (row["prompt"] or "").strip()
            if prompt and prompt not in seen:
                seen.add(prompt)
                prompts.append((source_row, prompt))
    return prompts


def source_label(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def generate(benign_path, attack_path, output_path, turns, sessions_per_class, seed):
    if turns < 2:
        raise ValueError("--turns must be at least 2")
    benign = load_unique_prompts(benign_path)
    attack = load_unique_prompts(attack_path)
    possible = min(len(benign), len(attack) // turns)
    count = possible if sessions_per_class is None else sessions_per_class
    if count < 6 or count > possible:
        raise ValueError(f"Need 6 to {possible} sessions per class for these files and turns={turns}")

    rng = random.Random(seed)
    # Source review anchors do not repeat. Attack rows are consumed once, in source order.
    benign_anchors = rng.sample(benign, count)
    attack_rows = attack[: count * turns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "session_id", "turn_index", "label", "text", "family",
            "source_file", "source_row", "synthetic_session", "attack_text_kind",
        ))
        writer.writeheader()
        for session_index, (source_row, review) in enumerate(benign_anchors):
            for turn_index in range(turns):
                instruction = EDIT_STEPS[turn_index % len(EDIT_STEPS)]
                if turn_index >= len(EDIT_STEPS):
                    instruction = f"Revision pass {turn_index + 1}: {instruction}"
                writer.writerow({
                    "session_id": f"sst2-edit-proxy-{session_index:03d}",
                    "turn_index": turn_index,
                    "label": 0,
                    "text": f"Movie review: {review}\n{instruction}",
                    "family": "sst2_synthetic_editing_proxy",
                    "source_file": source_label(benign_path),
                    "source_row": source_row,
                    "synthetic_session": 1,
                    "attack_text_kind": "",
                })
        for session_index in range(count):
            chunk = attack_rows[session_index * turns:(session_index + 1) * turns]
            for turn_index, (source_row, prompt) in enumerate(chunk):
                writer.writerow({
                    "session_id": f"prompt-extraction-proxy-{session_index:03d}",
                    "turn_index": turn_index,
                    "label": 1,
                    "text": prompt,
                    "family": "prompt_extraction_chunk_proxy",
                    "source_file": source_label(attack_path),
                    "source_row": source_row,
                    "synthetic_session": 1,
                    "attack_text_kind": "unverified_composite_prompt",
                })
    print(f"Wrote {count * 2} proxy sessions and {count * 2 * turns} rows to {output_path}")
    print("PROXY ONLY: source files lack session IDs; attack prompts may contain system context.",
          file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benign", type=Path, default=BENIGN_SOURCE)
    parser.add_argument("--attack", type=Path, default=ATTACK_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--sessions-per-class", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(args.benign.resolve(), args.attack.resolve(), args.output.resolve(),
             args.turns, args.sessions_per_class, args.seed)


if __name__ == "__main__":
    main()
