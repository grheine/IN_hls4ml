import os
import numpy as np
from copy import deepcopy
from prettytable import PrettyTable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from collections import namedtuple, OrderedDict
from utils.data.graphdata import GraphDataset
from utils.hls4ml.fix_graph_size import fix_graph_size
from utils.hls4ml.wrappers import data_wrapper, model_wrapper


from hls4ml.utils.config import config_from_pyg_model
from hls4ml.converters import convert_from_pyg_model

from utils.models.interaction_network_hls4ml import InteractionNetwork

def load_graphs(graph_indir, out_dir, graph_dims, n_graphs):
    
    n_node_max = graph_dims['n_node']
    n_edge_max = graph_dims['n_edge']
    
    Graph = namedtuple('Graph', ['x', 'edge_attr', 'edge_index', 'y', 'pid'])
    graph_files = np.array(os.listdir(graph_indir))
    graph_files = np.array([os.path.join(graph_indir, graph_file) for graph_file in graph_files])
    
    graphs = []
    n_nodes, n_edges = [], []
    
    for file in graph_files[:n_graphs]:
        x, edge_attr, edge_index, y, pid = np.load(file, allow_pickle=True)
        G = Graph(x, edge_attr, edge_index, y, pid)
        graphs.append(G)
        n_nodes.append(len(x))
        n_edges.append(len(y))
        
    dataset = GraphDataset(graphs)
    
    graphs = []
    for data in dataset[:n_graphs]:
        node_attr, edge_attr, edge_index, target, bad_graph = fix_graph_size(data, n_node_max, n_edge_max)
        if not bad_graph:
            graphs.append(data_wrapper(node_attr, edge_attr, edge_index, target))
    print(f"n_graphs: {len(graphs)}")
            
    nodes_kept = np.sum([n<n_node_max for n in n_nodes])
    edges_kept = np.sum([e<n_edge_max for e in n_edges])
    
    print(f'node dimension: {node_attr.shape}, edge dimension: {edge_attr.shape}')
    print(f'{nodes_kept/n_graphs:.1%} of graphs without truncation of nodes')
    print(f'{edges_kept/n_graphs:.1%} of graphs without truncation of edges')
 

    print("writing test bench data for 1st graph")
    data = graphs[0]
    node_attr, edge_attr, edge_index = data.x.detach().cpu().numpy(), data.edge_attr.detach().cpu().numpy(), data.edge_index.transpose(
        0, 1).detach().cpu().numpy().astype(np.int32)
    os.makedirs('tb_data', exist_ok=True)
    input_data = np.concatenate([node_attr.reshape(1, -1), edge_attr.reshape(1, -1), edge_index.reshape(1, -1)], axis=1)
    np.savetxt('tb_data/input_data.dat', input_data, fmt='%f', delimiter=' ')

    return graphs



def load_models(model_dir, output_dir, n_neurons, precision, reuse, part, graph_dims, hls_only=True):
    
    if 'dict' in model_dir:
        torch_model = InteractionNetwork(hidden_size=n_neurons)
        torch_model_dict = torch.load(model_dir)
        torch_model_dict = deepcopy(torch_model_dict)
        torch_model.load_state_dict(torch_model_dict)
    else:
        torch_model = torch.load(model_dir)
        
    torch_model.eval()

    forward_dict = OrderedDict()
    forward_dict["R1"] = "EdgeBlock"
    forward_dict["O"] = "NodeBlock"
    forward_dict["R2"] = "EdgeBlock"

    # get hls model
    config = config_from_pyg_model(torch_model,
                                   default_precision=precision,
                                   default_index_precision='ap_uint<16>', 
                                   default_reuse_factor=reuse)
    hls_model = convert_from_pyg_model(torch_model,
                                       forward_dictionary=forward_dict,
                                       **graph_dims,
                                       activate_final="sigmoid",
                                       output_dir=output_dir,
                                       hls_config=config,
                                       part=part
                                       )
#     hls_model.compile()
    
    if hls_only:
        return hls_model
    
    else:
        torch_wrapper = model_wrapper(torch_model)
        return torch_model, hls_model, torch_wrapper