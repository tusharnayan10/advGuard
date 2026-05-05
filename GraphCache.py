#import networkx as nx
#
#class CacheGraph(object):
#    def __init__(self, graph):
#        self.graph = graph
#        self.node_nums = len(graph.nodes)
#        self.edge_nums = len(graph.edges)
#
#        total_weight = 0.0
#
#        for u, v in graph.edges():
#            total_weight += graph.get_edge_data(u, v).get('label', 0.0)
#
#        if self.edge_nums > 0:
#            self.score = total_weight
#        else:
#            self.score = 0.0
#
#    def GetGraphScore(self):
#        return self.score
    


import networkx as nx

class CacheGraph(object):
    def __init__(self, graph):
        self.graph = graph
        self.node_nums = len(graph.nodes)
        self.edge_nums = len(graph.edges)

        total_weight = 0.0

        for u, v in graph.edges():
            total_weight += graph.get_edge_data(u, v).get('label', 0.0)

        if self.edge_nums > 0:
            avg_sim = total_weight / self.edge_nums
        else:
            avg_sim = 0.0

        if self.node_nums > 1:
            density = (2 * self.edge_nums) / (self.node_nums * (self.node_nums - 1))
        else:
            density = 0.0

        self.score = avg_sim * density

    def GetGraphScore(self):
        return self.score