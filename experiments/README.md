# GCN justification study

`gcn_study.py` runs three linked diagnostics on the **live detector's nearest-neighbor forest construction**: a score-versus-topology comparison on hard benign sessions, original-graph versus line-graph GCNs, and fixed-weight chain/branch/star probes. It does not modify the deployed AdvGuard model.

## Input needed for a paper result

Provide a UTF-8 CSV with `session_id,turn_index,label,text` and optional `family`. `label` is 0 for a benign iterative workflow and 1 for a prompt extraction session. Each session must be an independently collected, ordered trace. Include real benign text/code refinement, clarification, and planning sessions with at least 30 queries in their largest similarity component, along with attack sessions. Do not divide one attack trace into overlapping windows and call them independent sessions. This repository currently has prompt lists and flattened OASST rows, but no reliable session IDs for the required benign workflows. Those files alone cannot establish Experiment 1.

Run with a Python environment containing `numpy`, `networkx`, `scikit-learn`, `torch`, `torch-geometric`, and `sentence-transformers`:

```bash
python experiments/gcn_study.py \
  --sessions /path/to/ordered_sessions.csv \
  --baseline-file data/1baseline/baseline-prompt-10k.csv \
  --baseline-limit 1000 \
  --epochs 20 \
  --output experiments/results/real_sessions.json
```

The baseline file calibrates the 80th percentile of each benign baseline query's closest *other* baseline query, matching the repository's threshold rule. Increase `--baseline-limit` to 10000 for the paper configuration. Every classifier uses the same components and train/validation/test split. The score controls use both the live `GraphCache` score (`mean_similarity × density`) and the sum of edge weights described in the paper. The original graph GCN receives constant node features and similarity edge weights; the line graph GCN receives similarity values as node features. The handcrafted classifier uses size, degree, path, and edge-weight statistics. Models use a common 0.5 decision cutoff for this diagnostic; any paper comparison should tune the cutoff on validation sessions to a common false-positive target and repeat with multiple seeds.

The current runner selects the largest component per session and **does not reproduce Grubbs candidate selection, prompt-level detection, or online blocking**. The results therefore measure representation quality conditional on the constructed graph, not end-to-end AdvGuard performance. The fixed five-node topology probes are deliberately schematic; their scores and model probabilities must not be presented as evidence about real attack prevalence or detection accuracy. Train/test leakage remains possible if nominally separate sessions share the same underlying attack campaign or source conversation; group those at the source level before preparing the CSV.

## Pipeline smoke test

```bash
python -m unittest discover -s experiments -p 'test_*.py'
python experiments/gcn_study.py --demo --threshold 0.5 --epochs 10 \
  --checkpoint gnnTraining/model/gcn_model.pt \
  --output experiments/results/demo.json
```

`--demo` creates 24 **synthetic**, 12-query sessions. Its five-session test set is only a plumbing check. Its accuracy, F1, AUROC, and probe probabilities are not research results.
The optional checkpoint probe evaluates the repository's existing GCN on three 30-node, equal-weight schematic graphs. It does not retrain that checkpoint and should be interpreted only as a diagnostic.

## Important repository mismatch

`gnnTraining/train.py` builds all-pairs threshold graphs, while `AdvGuard.add` inserts at most one edge for each new query. These are different graph distributions. The original graph in live inference is always a forest, so descriptions of cycles or dense original-graph components do not follow from the implementation. The line graph can still become clique-like around a shared anchor. Also, `GraphCache.py` currently uses mean similarity times density, whereas the manuscript describes a sum of edge weights. This study reports both rather than assuming they are interchangeable.
