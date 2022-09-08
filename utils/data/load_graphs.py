import os
import random
import numpy as np
from collections import namedtuple, OrderedDict
from utils.data.graphdata import GraphDataset


def load_graphs(graph_indir, n_graphs='all', node_dim=2, edge_dim=2):
    Graph = namedtuple('Graph', ['x', 'edge_attr', 'edge_index', 'y', 'pid'])
    graph_files = [i for i in (os.path.join(graph_indir, f) for f in os.listdir(graph_indir)) if os.path.isfile(i)]

    graphs = []
    if n_graphs == 'all': n_graphs = len(graph_files)

    for file in random.sample(graph_files, n_graphs):
        x, edge_attr, edge_index, y, pid = np.load(file, allow_pickle=True)
        G = Graph(x[:,:node_dim], edge_attr[:][:edge_dim], edge_index, y, pid)
        graphs.append(G)

    Nedges, Ntrue, Nnodes = 0,0,0     
    for g in graphs:
        Nedges += len(g.y)
        Ntrue += sum(g.y)
        Nnodes += len(g.x)
        
    print(f'Nedges: {Nedges}, Ntrue: {Ntrue}, Ntrue/Nedges: {Ntrue/Nedges}')
    print(f'Nnodes: {Nnodes}, Nedges/Nnodes: {Nedges/Nnodes}')
        
    return graphs


