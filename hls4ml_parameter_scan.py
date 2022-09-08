
import os
import yaml
import argparse
import numpy as np
from time import time

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

# os.environ['PATH'] = '/home/greta/prog/Xilinx/Vivado/2019.2/bin:' + os.environ['PATH']
os.environ['PATH'] = '/opt/Xilinx/Vivado/2020.1/bin:' + os.environ['PATH']


# from abdel.utils.models.interaction_network_pyg import InteractionNetwork
from utils.models.interaction_network import InteractionNetwork
from utils.hls4ml.load_torch import load_graphs, load_models 


def parse_args():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--part', type=str, default='xczu11eg-ffvc1760-2-e', help='for which fpga board to compile')
    add_arg('--synth',action='store_true', help='whether to synthesize')
    add_arg('--strategy', type=str, default='Latency', help='latency or resource optimized design')
    args = parser.parse_args()
    return args

def get_utilization(synth_file):
    
    parts = ['BRAM_18K', 'DSP48E', 'FF', 'LUT', 'URAM']
    zcu_total = [1824, 2520, 548160, 274080]
    highflex_total = [1200, 2928,597120, 298560]
    versal_total = [None, 1968, None, 899840]

    with open(synth_file, 'r') as f:
        lines = f.readlines()[2:90]
        for i,line in enumerate(lines):
            if '|Total            |' in line:
                util = list(map(int, line.split('|')[2:6]))
            if '+ Latency:' in line:
                latencies = list(map(str, lines[i+6].split('|')[1:7]))
                latencies = [i.strip(' us') for i in latencies]
    return parts, util, zcu_total, highflex_total, versal_total, latencies
                
def main():
    args = parse_args()
    graph_indir='data/graphs_seg_1_pzmin_0.001_slopemax_2.0'
    n_neurons = config['n_neurons']
    run = config['run']
    trained_model_dir = f'models/optimization_seg_1/best/best_model_nevents_1000_hidden_dim_{n_neurons}.pt'
    output_dir = f'hls_output/par_scans_{run}/{key}_{par}'

    os.makedirs(output_dir, exist_ok=True)
    
    graph_dims = {
        "n_node": config['n_nodes'],
        "n_edge": config['n_edges'],
        "node_dim": config['node_dim'],
        "edge_dim": config['edge_dim']
    }
    hls_model = load_models(trained_model_dir, 
                            output_dir=output_dir, 
                            n_neurons=config['n_neurons'], 
                            precision=config['precision'], 
                            reuse=config['reuse'], 
                            part=args.part, 
                            graph_dims=graph_dims,
                            strategy=args.strategy)
    hls_model.compile()
    # graphs = load_graphs(graph_indir, output_dir, graph_dims, 100) #add testbench data

    if args.synth:
	    hls_model.build(csim=False,synth=True)

    parts, util, zcu_total, highflex_total, versal_total, latencies = get_utilization(f'{output_dir}/myproject_prj/solution1/syn/report/myproject_csynth.rpt')


    f.write(f'{config} \n')
    t_seconds = (time()-t0) %60
    t_minutes = (time()-t0) /60
    f.write(f'compilation time: {int(t_minutes)}min {int(t_seconds)}s \n')

    f.write('===================== \n')
    f.write('Highflex Latency (cycles, absolute, Interval) min/max:  \n')
    f.write('=====================  \n')
    f.write(str(latencies)+'\n')
        
    f.write('\n')

    f.write('===================== \n')
    f.write('Total utilization:  \n')
    f.write('=====================  \n')
    for i,r in enumerate(util):
        f.write(f'{parts[i]}: {r} \n')
        
    f.write('\n')              
                
    f.write('===================== \n')
    f.write('ZCU102 utilization: \n')
    f.write('===================== \n')
    for i, r in enumerate(util):
        if zcu_total[i] == None:
            f.write(f'{parts[i]}: ? \n')
        else:
            zcu_util = r/zcu_total[i]
            f.write(f'{parts[i]}: {zcu_util:.2%} of {zcu_total[i]} \n')
        
    f.write('\n')
        
    f.write('===================== \n')    
    f.write('Highflex utilization: \n')        
    f.write('===================== \n')
    for i, r in enumerate(util):
        if highflex_total[i] == None:
            f.write(f'{parts[i]}: ? \n')        
        else:
            highflex_util = r/highflex_total[i]
            f.write(f'{parts[i]}: {highflex_util:.2%} of {highflex_total[i]} \n')
        
    f.write('\n')
    
    f.write('===================== \n')    
    f.write('Versal utilization: \n')        
    f.write('===================== \n')
    for i, r in enumerate(util):
        if versal_total[i] == None:
            f.write(f'{parts[i]}: ? \n')
        else:
            versal_util = r/versal_total[i]
            f.write(f'{parts[i]}: {versal_util:.2%} of {versal_total[i]} \n')
        
    f.write('\n')



if __name__=="__main__":

    scans = {
        # 'reuse': np.arange(2,20), 
        'reuse': [8,16,32,1]
        # 'precision': np.arange(8,34,2), 
        # 'n_neurons': np.arange(3,4),
        # 'n_edges': np.arange(5,50,5),
        # 'n_nodes': np.arange(45,55,5)
    }

    for key, value in scans.items():
        config = {
            'n_nodes': 49,
            'n_edges': 98,
            'node_dim': 2,
            'edge_dim': 2,
            'n_neurons': 6,
            'precision': 'ap_fixed<16,8>', 
            'run': '49to98_resource_reuse',
            'reuse' : 8,
        }

        for par in value:
            f = open(f'hls4ml_parameter_scan_{config["run"]}.txt', 'a')
            if key == 'precision':
                par = f'ap_fixed<{par},{int(par/2)}>'
            print(par)
            config[key] = par
            t0 = time()
            f.write('====================================================== \n')
            f.write(f'scan for {key} in range {value}\n')
            f.write('====================================================== \n' )
            main()
            f.close()

        # for par in value:
        #     f = open(f'hls4ml_parameter_scan_{config["run"]}.txt', 'a')
        #     print(par)
        #     config[key] = par
        #     config['n_edges'] = 2*config['n_nodes']
        #     t0 = time()
        #     f.write('====================================================== \n')
        #     f.write(f'scan for n_nodes&2n_edges in range {value} \n')
        #     f.write('====================================================== \n' )
        #     main()
        #     f.close()
