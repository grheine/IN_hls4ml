import os
import numpy as np
from collections import namedtuple, OrderedDict
from utils.data.graphdata import GraphDataset


def load_graphs(graph_indir, n_graphs):
    Graph = namedtuple('Graph', ['x', 'edge_attr', 'edge_index', 'y', 'pid'])
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file) for graph_file in graph_files])

    graphs = []

    for file in graph_files[:n_graphs]:
        x, edge_attr, edge_index, y, pid = np.load(file, allow_pickle=True)
        G = Graph(x, edge_attr, edge_index, y, pid)
        graphs.append(G)
        
    return graphs


#dummy comment


