import os
import yaml
import argparse
import numpy as np

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
from utils.hls4ml.wrappers import model_wrapper


from hls4ml.utils.config import config_from_pyg_model
from hls4ml.converters import convert_from_pyg_model

os.environ['PATH'] = '/home/greta/prog/Xilinx/Vivado/2019.2/bin:' + os.environ['PATH']

# from abdel.utils.models.interaction_network_pyg import InteractionNetwork
from utils.models.interaction_network import InteractionNetwork
from utils.hls4ml.load_torch import load_graphs, load_models 


def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--n-graphs', type=int, default=100)
    add_arg('--n-nodes', type=int, default=27)
    add_arg('--n-edges', type=int, default=38)
    add_arg('--node-dim', type=int, default=3)
    add_arg('--edge-dim', type=int, default=4)	
    add_arg('--n-neurons', type=int, default=4)
    add_arg('--precision', type=str, default='ap_fixed<16,8>', help='precision to use')
    add_arg('--reuse', type=int, default=1, help="reuse factor")
    add_arg('--output-dir', type=str, default="hls_output", help='output directory')
    add_arg('--part', type=str, default='xczu9eg-ffvb1156-2-e', help='for which fpga board to compile')
    add_arg('--synth',action='store_true', help='whether to synthesize')
    args = parser.parse_args()
    return args

class data_wrapper(object):
    def __init__(self, node_attr, edge_attr, edge_index, target):
        self.x = node_attr
        self.edge_attr = edge_attr
        self.edge_index = edge_index.transpose(0,1)

        node_attr, edge_attr, edge_index = self.x.detach().cpu().numpy(), self.edge_attr.detach().cpu().numpy(), self.edge_index.transpose(0, 1).detach().cpu().numpy().astype(np.float32)
        node_attr, edge_attr, edge_index = np.ascontiguousarray(node_attr), np.ascontiguousarray(edge_attr), np.ascontiguousarray(edge_index)
        self.hls_data = [node_attr, edge_attr, edge_index]

        self.target = target
        self.np_target = np.reshape(target.detach().cpu().numpy(), newshape=(target.shape[0],))


def main():
    args = parse_args()
    graph_indir='data/graphs'
    trained_model_dir = 'models/IN_trained_events_100_neurons_4.pt'
    
    graph_dims = {
        "n_node": args.n_nodes,
        "n_edge": args.n_edges,
        "node_dim": args.node_dim,
        "edge_dim": args.edge_dim
    }
    
#     graphs = load_graphs(graph_indir, args.output_dir, graph_dims, 100)

    hls_model = load_models(trained_model_dir, output_dir=args.output_dir, n_neurons=args.n_neurons, precision=args.precision, reuse=args.reuse, part=args.part, graph_dims=graph_dims)

    if args.synth:
	    hls_model.build(csim=False,synth=True)

if __name__=="__main__":
	main()



