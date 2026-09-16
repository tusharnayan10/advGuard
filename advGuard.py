"""Online individual and sequential prompt-leakage detection."""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.utils import from_networkx
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from GraphCache import CacheGraph
from smirnov_grubbs import max_test_outliers


class AdvGuard:
    """Layered prompt detector for one conversation/user scope.

    Each prompt is evaluated independently and is also inserted into the
    sequential provenance graph. High per-prompt risk is blocked immediately
    (TTD=1). Sequential candidates are selected by a Grubbs maximum-outlier
    test over raw component edge-weight sums and confirmed by the trained GCN.

    Create a separate ``AdvGuard`` instance per relevant security scope (for
    example, per conversation or authenticated user). Baseline prompts remain
    available as stable reference anchors; only recent streamed prompts are
    eligible as dynamic neighbours.
    """

    def __init__(
        self,
        baseline_prompts,
        model_path,
        ttd=5,
        encoder_name="all-MiniLM-L6-v2",
        inj_model_name="ProtectAI/deberta-v3-base-prompt-injection-v2",
        alpha=0.01,
        baseline_percentile=90.0,
        individual_threshold=0.80,
        recent_window=500,
        top_k_neighbors=1,
        threshold_block_size=1024,
    ):
        if len(baseline_prompts) < 2:
            raise ValueError("AdvGuard requires at least two baseline prompts")
        if not 0.0 < float(baseline_percentile) < 100.0:
            raise ValueError("baseline_percentile must be between 0 and 100")
        if not 0.0 <= float(individual_threshold) <= 1.0:
            raise ValueError("individual_threshold must be between 0 and 1")
        if int(recent_window) < 1:
            raise ValueError("recent_window must be at least 1")
        if int(top_k_neighbors) < 1:
            raise ValueError("top_k_neighbors must be at least 1")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ttd = int(ttd)
        self.alpha = float(alpha)
        self.baseline_percentile = float(baseline_percentile)
        self.individual_threshold = float(individual_threshold)
        self.recent_window = int(recent_window)
        self.top_k_neighbors = int(top_k_neighbors)
        self.threshold_block_size = int(threshold_block_size)

        self.encoder = SentenceTransformer(encoder_name)
        self.graph_model = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False,
        ).to(self.device)
        self.graph_model.eval()

        self.g = nx.Graph()
        self.cache: list[np.ndarray] = []
        self.cache_idx_map: list[int] = []
        self.node_to_stream_idx: dict[int, int | None] = {}
        self.input_idx = 0
        self.alerted_nodes: set[int] = set()

        # Baseline embeddings establish the normal similarity distribution.
        # Their injection scores are not needed, so avoid running the more
        # expensive prompt-injection model over every baseline prompt.
        for prompt in baseline_prompts:
            self.add_baseline_prompt(prompt)
        self.baseline_count = len(self.cache)
        self.threshold = self.compute_threshold()

        self.inj_tokenizer = AutoTokenizer.from_pretrained(inj_model_name)
        self.inj_model = AutoModelForSequenceClassification.from_pretrained(
            inj_model_name
        ).to(self.device).eval()

        print(f"[AdvGuard] device: {self.device}")
        print(
            f"[AdvGuard] baseline similarity threshold "
            f"(p{self.baseline_percentile:g}): {self.threshold:.4f}"
        )
        print(
            f"[AdvGuard] per-prompt block threshold: "
            f"{self.individual_threshold:.4f} (TTD=1)"
        )
        print(
            f"[AdvGuard] recent window={self.recent_window}, "
            f"top-k neighbours={self.top_k_neighbors}, sequential TTD={self.ttd}"
        )

    def _inj_score(self, text: str) -> float:
        inputs = self.inj_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.inj_model(**inputs).logits, dim=-1)
        return float(probabilities[0, 1].item())

    def _embed(self, text: str) -> np.ndarray:
        embedding = self.encoder.encode(text, convert_to_numpy=True).astype(
            np.float32, copy=False
        )
        return embedding / (np.linalg.norm(embedding) + 1e-8)

    def add_baseline_prompt(self, prompt: str) -> None:
        embedding = self._embed(prompt)
        self.input_idx += 1
        node_id = self.input_idx
        self.cache.append(embedding)
        self.cache_idx_map.append(node_id)
        self.node_to_stream_idx[node_id] = None
        self.g.add_node(
            node_id,
            prompt=prompt,
            source="baseline",
            stream_idx=None,
            inj_score=0.0,
        )

    def compute_threshold(self) -> float:
        """Return the configured percentile of baseline nearest-neighbour PAS."""

        embeddings = np.vstack(self.cache).astype(np.float32, copy=False)
        count = len(embeddings)
        nearest = np.full(count, -np.inf, dtype=np.float32)

        for start in range(0, count, self.threshold_block_size):
            stop = min(start + self.threshold_block_size, count)
            similarities = embeddings[start:stop] @ embeddings.T
            local_rows = np.arange(stop - start)
            similarities[local_rows, np.arange(start, stop)] = -np.inf
            nearest[start:stop] = similarities.max(axis=1)

        print(
            "[AdvGuard] baseline nearest-neighbour stats: "
            f"mean={nearest.mean():.4f}, std={nearest.std():.4f}"
        )
        return float(np.percentile(nearest, self.baseline_percentile))

    def _candidate_cache_indices(self) -> np.ndarray:
        """Return all baseline indices plus the recent streamed-query window."""

        query_start = max(self.baseline_count, len(self.cache) - self.recent_window)
        baseline_indices = np.arange(self.baseline_count, dtype=np.int64)
        recent_indices = np.arange(query_start, len(self.cache), dtype=np.int64)
        return np.concatenate((baseline_indices, recent_indices))

    def _nearest_neighbors(self, embedding: np.ndarray):
        candidate_indices = self._candidate_cache_indices()
        candidate_embeddings = np.vstack(
            [self.cache[index] for index in candidate_indices]
        )
        similarities = candidate_embeddings @ embedding
        order = np.argsort(-similarities)
        count = min(self.top_k_neighbors, len(order))

        neighbors = []
        for position in order[:count]:
            cache_index = int(candidate_indices[position])
            neighbors.append(
                (
                    self.cache_idx_map[cache_index],
                    float(similarities[position]),
                )
            )
        return neighbors

    def add(self, prompt: str, source: str = None, stream_idx: int | None = None):
        """Evaluate one prompt and always update sequential graph state.

        Returns the historical four-item API used by ``main.py``:
        ``best_similarity, nearest_node, flagged, injection_score``.
        ``flagged`` represents immediate per-prompt blocking; sequential GCN
        decisions are returned by :meth:`detector`.
        """

        embedding = self._embed(prompt)
        neighbors = self._nearest_neighbors(embedding)
        nearest_node, best_similarity = neighbors[0]

        # Insert the prompt before making the individual blocking decision so
        # even an immediately blocked attack contributes to sequential state.
        self.input_idx += 1
        node_id = self.input_idx
        self.cache.append(embedding)
        self.cache_idx_map.append(node_id)
        self.node_to_stream_idx[node_id] = stream_idx

        injection_score = self._inj_score(prompt)
        self.g.add_node(
            node_id,
            prompt=prompt,
            source=source if source else "user",
            stream_idx=stream_idx,
            inj_score=injection_score,
        )

        for neighbor_node, similarity in neighbors:
            if similarity > self.threshold:
                self.g.add_edge(neighbor_node, node_id, label=float(similarity))

        # Individual detection remains active for every prompt. This is a
        # high-confidence hard gate with TTD=1, independent of stream position.
        flagged = injection_score >= self.individual_threshold
        if flagged:
            self.alerted_nodes.add(node_id)

        return best_similarity, nearest_node, flagged, injection_score

    def graph_checker(self, component_graph: nx.Graph) -> bool:
        """Confirm a statistically anomalous component with the line-graph GCN."""

        if component_graph.number_of_edges() == 0:
            return False

        for _, _, data in component_graph.edges(data=True):
            data["label"] = float(data.get("label", 0.0))

        line_graph = nx.line_graph(component_graph)
        for line_node in line_graph.nodes:
            u, v = line_node
            # Direct indexing is orientation-safe for undirected NetworkX
            # edges and avoids silently replacing valid PAS values with zero.
            line_graph.nodes[line_node]["label"] = float(
                component_graph.edges[u, v]["label"]
            )

        pyg_graph = from_networkx(line_graph, group_node_attrs=["label"])
        pyg_graph.x = pyg_graph.x.float().reshape(-1, 1)
        pyg_graph.batch = torch.zeros(pyg_graph.x.size(0), dtype=torch.long)
        pyg_graph = pyg_graph.to(self.device)

        with torch.no_grad():
            logits = self.graph_model(
                pyg_graph.x,
                pyg_graph.edge_index,
                pyg_graph.batch,
            )
        return int(logits.argmax(dim=1).item()) == 1

    @staticmethod
    def _query_node_count(graph: nx.Graph) -> int:
        """Count streamed queries without treating a baseline anchor as TTD."""

        return sum(
            data.get("source") != "baseline"
            for _, data in graph.nodes(data=True)
        )

    def detector(self):
        """Run total-weight outlier selection followed by GCN confirmation."""

        subgraphs = [
            CacheGraph(self.g.subgraph(nodes).copy())
            for nodes in nx.connected_components(self.g)
        ]
        scores = [float(subgraph.GetGraphScore()) for subgraph in subgraphs]
        anomaly_subgraphs = []

        if len(scores) <= 2:
            return anomaly_subgraphs

        outlier_scores = max_test_outliers(scores, alpha=self.alpha)
        if len(outlier_scores) == 0:
            return anomaly_subgraphs

        for subgraph, score in zip(subgraphs, scores):
            is_outlier = any(abs(score - value) < 1e-6 for value in outlier_scores)
            enough_queries = self._query_node_count(subgraph.graph) >= self.ttd
            if not is_outlier or not enough_queries:
                continue

            # The GCN is the final decision for sequential detection.
            if self.graph_checker(subgraph.graph):
                anomaly_subgraphs.append(subgraph)
                self.alerted_nodes.update(subgraph.graph.nodes())

        return anomaly_subgraphs
