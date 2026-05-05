import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch_geometric.utils import from_networkx
from smirnov_grubbs import max_test_outliers
from GraphCache import CacheGraph
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class AdvGuard:
    """
    Two-Stage Detection System:

    Stage 1 (Early Phase: first 30 nodes)
        - Uses ONLY soft prompt signal (inj_score)
        - If inj_score >= 0.80 → flag immediately

    Stage 2 (After 30 prompts)
        - We set an upper bound of normal similarity and treat anything above it as potentially anomalous.
        - Uses graph anomaly detection (Grubbs)
        - AND GCN confirmation
    """

    def __init__(
        self,
        baseline_prompts,
        model_path,
        ttd=5,
        encoder_name="all-MiniLM-L6-v2",
        inj_model_name="ProtectAI/deberta-v3-base-prompt-injection-v2",
        alpha=0.01,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Sentence encoder
        self.encoder = SentenceTransformer(encoder_name)

        # Load pretrained GCN
        self.graph_model = torch.load(model_path, map_location=self.device, weights_only=False)
        self.graph_model.eval()

        self.ttd = ttd
        self.g = nx.Graph()

        # Embedding cache
        self.cache = []
        self.cache_idx_map = []
        self.hash_dict = {}

        # Mapping
        self.node_to_stream_idx = {}

        self.input_idx = 0
        self.alerted_nodes = set()

        self.alpha = float(alpha)

        # Load prompt-injection model (soft signal)
        self.inj_tokenizer = AutoTokenizer.from_pretrained(inj_model_name)
        self.inj_model = AutoModelForSequenceClassification.from_pretrained(
            inj_model_name
        ).to(self.device).eval()

        # Initialize baseline
        for p in baseline_prompts:
            self.add_baseline_prompt(p)

        # Compute similarity threshold
        self.threshold = self.compute_threshold()
        print(f"[AdvGuard] Similarity threshold: {self.threshold:.4f}")
        print(f"[AdvGuard] Early-stage inj threshold: 0.80 ")

    # Prompt Classifier Signal
    def _inj_score(self, text: str) -> float:
        inputs = self.inj_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            logits = self.inj_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        return float(probs[0, 1].item())

    # Embeddings
    def _embed(self, text: str) -> np.ndarray:
        emb = self.encoder.encode(text, convert_to_numpy=True)
        return emb / (np.linalg.norm(emb) + 1e-8)
    
    # Cosine similarity to cache
    def cosine_similarity_to_cache(self, emb: np.ndarray) -> np.ndarray:
        if not self.cache:
            return np.array([])
        return np.vstack(self.cache) @ emb

    # Baseline Setup
    
    # Each baseline prompt is embedded and stored 
    # This creates our reference embedding space of benign queries

    def add_baseline_prompt(self, prompt: str):
        emb = self._embed(prompt)
        inj = self._inj_score(prompt)

        self.input_idx += 1
        node_id = self.input_idx

        self.cache.append(emb)
        self.cache_idx_map.append(node_id)
        self.hash_dict[node_id] = emb
        self.node_to_stream_idx[node_id] = None

        # It builds the initial graph structure with baseline prompts as nodes, which will be used for similarity comparisons and anomaly detection later on.
        self.g.add_node(
            node_id,
            prompt=prompt,
            source="baseline",
            stream_idx=None,
            inj_score=inj,
        )

    # Threshold Estimation
    def compute_threshold(self) -> float:
        sims_list = []

        # Loop over all baseline nodes
        for node_id in range(1, self.input_idx + 1):
            # Get embedding of current prompt
            emb = self.hash_dict[node_id]
            # compute similarity with ALL others in the cache 
            sims = self.cosine_similarity_to_cache(emb)

            # If there’s only one prompt: skip   
            if sims.size < 2:
                continue
            
            # Sort similarities - ighest similarity first
            idx_sorted = np.argsort(-sims)
            # Take second highest similarity because the first one will be the prompt itself. 
            sims_list.append(sims[idx_sorted[1]])

        if not sims_list:
            return 0.0

        # 80% of normal similarities are below this value
            # This comes from  analysis that in few rare case some adv query can be very similar to a baseline prompt, but most of them are not.
        print("Threshold stats:", np.mean(sims_list), np.std(sims_list))
        return float(np.percentile(sims_list, 80))

    # ADD QUERY 
    def add(self, prompt: str, source: str = None, stream_idx: int | None = None):
        emb = self._embed(prompt)
        # Compute similarity with all previous queries in the cache to find the nearest neighbor and its similarity score.
        sims = self.cosine_similarity_to_cache(emb)
    
        if sims.size == 0:
            nearest_node = None
            best_sim = 0.0
        else:
            # Find most similar past query
            idx = int(np.argmax(sims))
            best_sim = float(sims[idx])
            nearest_node = self.cache_idx_map[idx]
    
        self.input_idx += 1
        node_id = self.input_idx
    
        # Add this query to memory
        self.cache.append(emb)
        self.cache_idx_map.append(node_id)
        self.hash_dict[node_id] = emb
        self.node_to_stream_idx[node_id] = stream_idx
    
        # Compute injection score for this prompt, which serves as an early detection signal. 
        inj = self._inj_score(prompt)
    
        # always add node to graph
        self.g.add_node(
            node_id,
            prompt=prompt, # prompt text
            source=source if source else "user1",
            stream_idx=stream_idx, # stream position
            inj_score=inj,
        )
    
        flagged = False
    
        # top_k = np.argsort(-sims)[:3] if sims.size > 0 else []
    
        # --- BUILD GRAPH 
        # Connect nodes only if similarity is high
        if nearest_node is not None and best_sim > self.threshold:
            self.g.add_edge(nearest_node, node_id, label=best_sim)

            # Only allow propagation AFTER early stage
            if stream_idx is not None and stream_idx >= 30:
                if nearest_node in self.alerted_nodes:
                    flagged = True
                    self.alerted_nodes.add(node_id)

        # STAGE 1: EARLY DETECTION - Prompt-level classifier
        if stream_idx is not None and stream_idx < 30:
            if inj >= 0.80:
                flagged = True
                self.alerted_nodes.add(node_id)

        return best_sim, nearest_node, flagged, inj


    
    # GCN CLASSIFIER
    def graph_checker(self, component_graph: nx.Graph) -> bool:
        for u, v, data in component_graph.edges(data=True):
            data["label"] = float(data.get("label", 0.0))

        # Convert graph to line graph
        LG = nx.line_graph(component_graph)
        edge_attr = nx.get_edge_attributes(component_graph, "label")

        for node in LG.nodes:
            LG.nodes[node]["label"] = float(edge_attr.get(node, 0.0))

        # Converts graph - format usable by GCN
        pyg_graph = from_networkx(LG, group_node_attrs="all")

        if not hasattr(pyg_graph, "x"):
            pyg_graph.x = torch.zeros((1, 1))
            pyg_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            pyg_graph.x = pyg_graph.x.float()

        pyg_graph.batch = torch.zeros(pyg_graph.x.size(0), dtype=torch.long)
        pyg_graph = pyg_graph.to(self.device)

        with torch.no_grad():
            out = self.graph_model(pyg_graph.x, pyg_graph.edge_index, pyg_graph.batch)
            pred = out.argmax(dim=1).item()

        return pred == 1

    # DETECTOR- Stage 2 detector - anomaly detection over the graph
    def detector(self):
        # We break the graph into connected components - each component = group of similar queries
        subgraphs = [
            CacheGraph(self.g.subgraph(c).copy())
            for c in nx.connected_components(self.g)
        ]

        # Each cluster gets a score (now using improved CacheGraph logic: avg_sim * density)
        base_scores = [float(sg.GetGraphScore()) for sg in subgraphs]

        #  print(f"[Detector] Total clusters: {len(subgraphs)}")
        #  print("Score mean:", np.mean(base_scores), "std:", np.std(base_scores))
        #  print(f"[Detector] Cluster scores: {base_scores}")

        anomaly_subgraphs = []
# Outlier + GCN
        if len(base_scores) > 2:
            outliers = max_test_outliers(base_scores, alpha=self.alpha)

            if len(outliers) > 0:
                for sg, score in zip(subgraphs, base_scores):

                    # score is anomalous and cluster is large enough to be suspicious
                    if any(abs(score - s) < 1e-6 for s in outliers) and sg.node_nums >= self.ttd:

                        # GCN confirmation
                        if self.graph_checker(sg.graph):
                            # Mark as attack
                            anomaly_subgraphs.append(sg)
                            self.alerted_nodes |= set(sg.graph.nodes())
# Only Outlier               
#        if len(base_scores) > 2:
#            outliers = max_test_outliers(base_scores, alpha=self.alpha)
#
#            if len(outliers) > 0:
#                for sg, score in zip(subgraphs, base_scores):
#
#                    # Outlier condition ONLY
#                    if any(abs(score - s) < 1e-6 for s in outliers):
#
#                        # no ttd
#                        # anomaly_subgraphs.append(sg)
#                        # self.alerted_nodes |= set(sg.graph.nodes())
#
#                        # tdd
#                        if sg.node_nums >= self.ttd:
#                            anomaly_subgraphs.append(sg)
#                            self.alerted_nodes |= set(sg.graph.nodes())
#
        return anomaly_subgraphs