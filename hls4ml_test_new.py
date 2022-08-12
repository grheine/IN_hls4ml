
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
from utils.models.interaction_network_hls4ml import InteractionNetwork
from utils.hls4ml.load_torch import load_graphs, load_models 


def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--n-graphs', type=int, default=100)
    add_arg('--n-nodes', type=int, default=100)
    add_arg('--n-edges', type=int, default=200)
    add_arg('--node-dim', type=int, default=3)
    add_arg('--edge-dim', type=int, default=3)	
    add_arg('--n-neurons', type=int, default=6)
    add_arg('--precision', type=str, default='ap_fixed<16,8>', help='precision to use')
    add_arg('--reuse', type=int, default=1, help="reuse factor")
    add_arg('--strategy', type=str, default='latency', help='latency or throughput optimized design')
    add_arg('--output-dir', type=str, default="hls_output/ZCU102_new", help='output directory')
    add_arg('--part', type=str, default='xczu9eg-ffvb1156-2-e', help='for which fpga board to compile')
    add_arg('--synth',action='store_true', help='whether to synthesize')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    graph_indir='data/graphs'
    trained_model_dir = 'models/optimization/best/best_model_hidden_dim_6.pt'
    
    
    graph_dims = {
        "n_node": args.n_nodes,
        "n_edge": args.n_edges,
        "node_dim": args.node_dim,
        "edge_dim": args.edge_dim
    }
    
    #graphs = load_graphs(graph_indir, args.output_dir, graph_dims, 100)

    hls_model = load_models(trained_model_dir, output_dir=args.output_dir, n_neurons=args.n_neurons, precision=args.precision, reuse=args.reuse, part=args.part, graph_dims=graph_dims, strategy=args.strategy)
    hls_model.compile()

    if args.synth:
	    hls_model.build(csim=False,synth=True)

if __name__=="__main__":
	main()
