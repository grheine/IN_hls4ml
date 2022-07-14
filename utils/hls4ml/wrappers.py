import torch
import numpy as np
import os
import shutil

class data_wrapper(object):
    def __init__(self, node_attr, edge_attr, edge_index, target):
        self.x = node_attr
        self.edge_attr = edge_attr
        self.edge_index = edge_index.transpose(0,1)

        node_attr, edge_attr, edge_index = self.x.detach().cpu().numpy(), self.edge_attr.detach().cpu().numpy(), self.edge_index.transpose(0,1).detach().cpu().numpy().astype(np.float32)
        node_attr, edge_attr, edge_index = np.ascontiguousarray(node_attr), np.ascontiguousarray(edge_attr), np.ascontiguousarray(edge_index)
        self.hls_data = [node_attr, edge_attr, edge_index]

        self.target = target
        self.np_target = np.reshape(target.detach().cpu().numpy(), newshape=(target.shape[0],))
        

class model_wrapper():
    def __init__(self, model):
        self.model = model
        self.r = []

    def EdgeBlock_forward(self, block, node_attr, edge_attr, edge_index, out_dim):

        # get graph dimensions
        n_node, node_dim = node_attr.shape
        n_edge, edge_dim = edge_attr.shape

        # initialize outputs
        edge_update = torch.zeros((n_edge, out_dim))
        edge_update_aggr = torch.zeros((n_node, out_dim))

        # edge counter
        num_edge_per_node = torch.zeros((n_node,))
        edge_aggr_mask = torch.zeros((n_node,))

        for i in range(n_edge):
            # get receiver, sender indices

            s = edge_index[0, i]
            r = edge_index[1, i]
            self.r = r

            num_edge_per_node[r] += 1
            edge_aggr_mask[r] = 1

            # construct NN input: <receiver, sender, edge>
            edge_i = edge_attr[i, :]  # edge
            node_s = node_attr[s, :]  # sender
            node_r = node_attr[r, :]  # receiver
            node_concat = torch.cat((node_r, node_s), dim=0)
            phi_input_i = torch.cat((node_concat, edge_i), dim=0)

            # send through NN
            edge_update[i] = block(phi_input_i)
            for j in range(out_dim):
                edge_update_aggr[r, j] += edge_update[i, j]

#             # aggregation step
#             if self.model.aggr in ['add', 'mean']:
#                 for j in range(out_dim):
#                     edge_update_aggr[r, j] += edge_update[i, j]
#             elif self.model.aggr == 'max':
#                 if num_edge_per_node[r] <= 1:  # if this is the first edge to be sent to the aggregate, it's the max by default
#                     edge_update_aggr[r] = edge_update[i]
#                 else:
#                     for j in range(out_dim):
#                         if edge_update_aggr[r, j] < edge_update[i, j]:
#                             edge_update_aggr[r, j] = edge_update[i, j]

#         # extra step for mean-aggregation
#         if self.model.aggr=='mean':
#             for i in range(n_node):
#                 n_edge_i = num_edge_per_node[i]
#                 if (n_edge_i > 1):
#                     edge_update_aggr[i] = edge_update_aggr[i] / n_edge_i


        return edge_update, edge_update_aggr

    def NodeBlock_forward(self, block, node_attr, edge_attr_aggr, out_dim):
        # get graph dimensions
        n_node, node_dim = node_attr.shape
        edge_dim = edge_attr_aggr.shape[1]

        # initialize output
        node_update = torch.zeros((n_node, out_dim))

        for i in range(n_node):

            # construct NN input: <node, edge_aggr>
            node_i = node_attr[i]
            edge_aggr_i = edge_attr_aggr[i]
            phi_input_i = torch.cat((node_i, edge_aggr_i))

            # send through NN
            node_update[i] = block(phi_input_i)

        return node_update

    def forward(self, data):
        node_attr = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        
        n_nodes = data.x.shape[1]
        n_edges = data.edge_attr.shape[1]

        edge_update_1, edge_update_aggr_1 = self.EdgeBlock_forward(self.model.R1, node_attr, edge_attr, edge_index, out_dim=n_edges)  #R1
        node_update = self.NodeBlock_forward(self.model.O, node_attr, edge_update_aggr_1, out_dim=n_nodes)  #O
        edge_update_2, edge_update_aggr_2 = self.EdgeBlock_forward(self.model.R2, node_update, edge_update_1, edge_index, out_dim=1)  #R2
        out = torch.sigmoid(edge_update_2)
        return out
    
