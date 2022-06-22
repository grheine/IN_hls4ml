# System
import os
import sys
import time
import argparse
import logging
import multiprocessing as mp
from functools import partial
sys.path.append("../")

# Externals
import yaml
import pickle
import numpy as np
import pandas as pd
import csv 
from tqdm import tqdm

# Locals
from collections import namedtuple


class build_graphs:
    
    def __init__(self, events, start=0, end=None, shuffle=True,  pz_min=0.1, remove_duplicates=True, slope=2.):
        self.raw = events
        self.start = start
        self.end = end
        self.shuffle = shuffle
        self.pz_min = pz_min #GeV
        self.remove_duplicates = remove_duplicates
        self.slope = slope        
        self.events = self.preprocess()

    def remove_skewed_layers(self, df):
        return df.loc[df.skewed==0]
    
    def layer_mapping(self, hits):
        """ layer mapping as adjecent layers"""
        hits['layer'] = hits.layer_id.values
        layerids = [1,2,7,8,9,10,15,16,17,18,23,24,25,26,31,32,33,34,39,40,41,42,47,48]
        inverse_layer_mapping = {(i+1):x for i,x in enumerate(layerids)} # map index i to vertical layers
        layer_mapping = {v: k for k, v in inverse_layer_mapping.items()} # invert mapping
        hits['layer'].replace(layer_mapping, inplace=True)
        return hits        
    
    def preprocess(self):
        hits = self.raw.loc[self.start:self.end].reset_index('event_id')
        # Shuffle node indices
        if self.shuffle:
            hits = hits.sample(frac=1, random_state=42).reset_index(drop=True)
            hits = hits.assign(Hit_id=hits.hit_id).drop('hit_id', axis=1)
            hits.index.name = 'hit_id'
            hits = hits.reset_index('hit_id')

        hits = self.remove_skewed_layers(hits)
        hits = self.layer_mapping(hits)

        # apply pz cut, removes all noise hits
        hits = hits[hits['pz'] > self.pz_min]

        # add radius and theta
        hits = hits.assign(r=np.sqrt(hits.x**2 + hits.z**2))
        hits = hits.assign(theta=np.arctan2(hits.x, hits.z))
        
        # Remove duplicate hits
        if self.remove_duplicates:
            hits = hits.drop_duplicates(subset=('event_id', 'particle_id', 'layer'))
        return hits.set_index('event_id')
    

    def create_graph_list(self, dtype=object):
        evs = self.events
        feature_scale = np.array([20., 100., 0.1, 0.1])
        hits = evs[['x','z', 'theta', 'iso']]/feature_scale

        gb = evs[['hit_id','theta','layer','particle_id','iso','x','z']].groupby('event_id')

        graphs = []
        seg = []
        hitID = []
        Graph = namedtuple('Graph', ['x', 'edge_attr', 'edge_index', 'y', 'pid'])

        for idx in tqdm(gb.groups):
            df = gb.get_group(idx)
            if df.empty: 
                print('no event found')
                continue
            dfhits = hits.loc[idx]
            segments = []
            dx, dz, dtheta = [], [], []
            y        = []
            data     = df.to_numpy()
            
            for i, data_i in enumerate(data):
                for j, data_j in enumerate(data):
                    if (i>=j): continue
                    if (abs(data_i[2]-data_j[2])!=1): continue
                    Dx = data_j[-2]-data_i[-2]
                    Dz = data_j[-1]-data_i[-1]
                    Dtheta = data_j[1]-data_i[1]
                    angle = Dx/Dz 
                    if (abs(angle)>self.slope): continue
                        
                    segments.append([i, j])
                    dx.append(Dx)
                    dz.append(Dz)
                    dtheta.append(Dtheta)
                    y.append(int(data_i[3]==data_j[3]))
                    
            if (segments==[]): continue
            n_hits = len(dfhits)            
            n_edges = len(segments)
            
            X = dfhits.values
            edge_attr = (np.stack((dx, dz, dtheta))/feature_scale[:3,np.newaxis])
            edge_index = np.array(np.vstack(segments).T)
            y = np.array(y, dtype=np.int8)
            pid = df.particle_id
            G = Graph(X, edge_attr, edge_index, y, pid)
            graphs.append(G)
        
        return graphs
    
   

