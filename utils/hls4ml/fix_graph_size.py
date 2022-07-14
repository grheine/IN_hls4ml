#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 19:12:47 2021

@author: abdel
"""
import torch 

# adds dummy nodes, adds dummy edges between dummy nodes (i.e. a "dummy" graph that is disconnected from the "true" graph)
def pad_nodes_and_edges(node_attr, edge_attr, edge_index, n_node_max, n_edge_max):
    n_node_diff = n_node_max - node_attr.shape[0]
    node_attr_appendage = torch.zeros(n_node_diff, node_attr.shape[1], dtype=node_attr.dtype)
    node_attr_prime = torch.cat([node_attr, node_attr_appendage], dim=0)

    n_edge_diff = n_edge_max - edge_attr.shape[0]
    edge_attr_appendage = torch.zeros(n_edge_diff, edge_attr.shape[1], dtype=edge_attr.dtype)
    edge_attr_prime = torch.cat([edge_attr, edge_attr_appendage], dim=0)

    edge_index_appendage = torch.zeros(n_edge_diff, 2, dtype=edge_index.dtype)
    dummy_node_index = node_attr.shape[0]
    for i in range(n_edge_diff):
        edge_index_appendage[i,0] = dummy_node_index
        edge_index_appendage[i,1] = dummy_node_index
    edge_index_prime = torch.cat([edge_index, edge_index_appendage], dim=0)

    return node_attr_prime, edge_attr_prime, edge_index_prime

# adds completely-disconnected dummy nodes
def pad_nodes(node_attr, edge_attr, edge_index, n_node_max, n_edge_max):
    n_node_diff = n_node_max - node_attr.shape[0]
    node_attr_appendage = torch.zeros(n_node_diff, node_attr.shape[1], dtype=node_attr.dtype)
    node_attr_prime = torch.cat([node_attr, node_attr_appendage], dim=0)

    return node_attr_prime, edge_attr, edge_index

# adds completely-disconnected dummy nodes, removes true edges
def pad_nodes_truncate_edges(node_attr, edge_attr, edge_index, n_node_max, n_edge_max):
    n_node_diff = n_node_max - node_attr.shape[0]
    node_attr_appendage = torch.zeros(n_node_diff, node_attr.shape[1], dtype=node_attr.dtype)
    node_attr_prime = torch.cat([node_attr, node_attr_appendage], dim=0)

    # last hired, first fired
    edge_attr_prime = edge_attr[:n_edge_max,:]
    edge_index_prime = edge_index[:n_edge_max,:]

    return node_attr_prime, edge_attr_prime, edge_index_prime

# adds dummy edges between true nodes (namely just the last node)
def pad_edges(node_attr, edge_attr, edge_index, n_node_max, n_edge_max):
    n_edge_diff = n_edge_max - edge_attr.shape[0]
    edge_attr_appendage = torch.zeros(n_edge_diff, edge_attr.shape[1], dtype=edge_attr.dtype)
    edge_attr_prime = torch.cat([edge_attr, edge_attr_appendage], dim=0)

    edge_index_appendage = torch.zeros(n_edge_diff, 2, dtype=edge_index.dtype)
    final_node_index = node_attr.shape[0]-1
    for i in range(n_edge_diff):
        edge_index_appendage[i, 0] = final_node_index
        edge_index_appendage[i, 1] = final_node_index
    edge_index_prime = torch.cat([edge_index, edge_index_appendage], dim=0)

    return node_attr, edge_attr_prime, edge_index_prime

# removes true edges
def truncate_edges(node_attr, edge_attr, edge_index, n_node_max, n_edge_max):
    edge_attr_prime = edge_attr[:n_edge_max, :]
    edge_index_prime = edge_index[:n_edge_max, :]

    return node_attr, edge_attr_prime, edge_index_prime

def fix_graph_size(node_attr, edge_attr, edge_index, target, n_node_max=112, n_edge_max=148):
    # node_attr: [n_node, node_dim]
    # edge_attr: [n_edge, edge_dim]
    # edge_index: [2, n_edge]
    # first remove edges that connect out-of-bound nodes
    edge_index = edge_index.T
    
    mask = edge_index[:,0].lt(n_node_max) & edge_index[:,1].lt(n_node_max)
    edge_index = edge_index[mask]
    edge_attr = edge_attr[mask]
    target = target[mask]

    # second truncate up to n_node_max and n_edge_max
    node_attr = node_attr[:n_node_max]
    edge_attr = edge_attr[:n_edge_max]
    edge_index = edge_index[:n_edge_max]
    target = target[:n_edge_max]

    n_node = node_attr.shape[0]
    n_edge = edge_attr.shape[0]

    if (n_node < n_node_max) and (n_edge < n_edge_max):
        node_attr_prime, edge_attr_prime, edge_index_prime = pad_nodes_and_edges(node_attr, edge_attr, edge_index, n_node_max=n_node_max, n_edge_max=n_edge_max)
        bad_graph = False
    elif (n_node < n_node_max) and (n_edge == n_edge_max):
        node_attr_prime, edge_attr_prime, edge_index_prime = pad_nodes(node_attr, edge_attr, edge_index, n_node_max=n_node_max, n_edge_max=n_edge_max)
        bad_graph = False
    elif (n_node == n_node_max) and (n_edge < n_edge_max):
        node_attr_prime, edge_attr_prime, edge_index_prime = pad_edges(node_attr, edge_attr, edge_index, n_node_max=n_node_max, n_edge_max=n_edge_max)
        bad_graph = True
    else: # (n_node == n_node_max) and (n_edge == n_edge_max)
        node_attr_prime, edge_attr_prime, edge_index_prime = node_attr, edge_attr, edge_index
        bad_graph = False

    return node_attr_prime, edge_attr_prime, edge_index_prime, target, bad_graph

