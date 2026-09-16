"""Scoring wrapper for connected prompt-provenance components."""


class CacheGraph:
    def __init__(self, graph):
        self.graph = graph
        self.node_nums = graph.number_of_nodes()
        self.edge_nums = graph.number_of_edges()

        # PAS/cosine similarity is stored in each edge's ``label`` field.
        # Summing it rewards both high association strength and repeated
        # aggregation, so a growing coordinated-query component receives a
        # progressively larger anomaly score.
        self.score = sum(
            float(data.get("label", 0.0))
            for _, _, data in graph.edges(data=True)
        )

    def GetGraphScore(self):
        return self.score
