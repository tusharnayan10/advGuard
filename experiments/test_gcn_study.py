import unittest

import networkx as nx
import numpy as np

from GraphCache import CacheGraph
from gcn_study import features, nearest_neighbor_graph, topology_probe_graphs


class GraphConstructionTests(unittest.TestCase):
    def test_incremental_nearest_neighbor_graph_is_a_forest(self):
        texts = [str(i) for i in range(7)]
        vectors = {text: np.array([1.0, i * 0.01]) /
                   np.linalg.norm([1.0, i * 0.01]) for i, text in enumerate(texts)}
        graph = nearest_neighbor_graph(texts, vectors, 0.5)
        self.assertTrue(nx.is_forest(graph))
        self.assertEqual(graph.number_of_edges(), 6)
        self.assertTrue(all(max(a, b) == i for i, (a, b) in
                            enumerate(sorted(graph.edges(), key=lambda e: max(e)), start=1)))

    def test_matched_topologies_have_equal_scores_but_different_line_graphs(self):
        probes = topology_probe_graphs()
        self.assertEqual(len({round(features(graph)[2], 8) for graph in probes.values()}), 1)
        self.assertEqual(len({round(features(graph)[3], 8) for graph in probes.values()}), 1)
        for graph in probes.values():
            self.assertAlmostEqual(features(graph)[2], CacheGraph(graph).GetGraphScore())
        self.assertTrue(all(nx.is_tree(graph) and len(graph) == 30 for graph in probes.values()))
        line_counts = [nx.line_graph(probes[name]).number_of_edges()
                       for name in ("chain", "branch", "star")]
        self.assertTrue(line_counts[0] < line_counts[1] < line_counts[2])


if __name__ == "__main__":
    unittest.main()
