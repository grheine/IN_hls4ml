import os
import random
import numpy as np
from collections import namedtuple, OrderedDict
from utils.data.graphdata import GraphDataset


def load_graphs(graph_indir, n_graphs, node_dim, edge_dim):
    Graph = namedtuple('Graph', ['x', 'edge_attr', 'edge_index', 'y', 'pid'])
    graph_files = [i for i in (os.path.join(graph_indir, f) for f in os.listdir(graph_indir)) if os.path.isfile(i)]

    graphs = []

    for file in random.sample(graph_files, n_graphs):
        x, edge_attr, edge_index, y, pid = np.load(file, allow_pickle=True)
        G = Graph(x[:,:node_dim], edge_attr[:][:edge_dim], edge_index, y, pid)
        graphs.append(G)
        
    return graphs


#dummy comment


