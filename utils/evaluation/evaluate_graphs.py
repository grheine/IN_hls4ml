import numpy as np
import torch
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from ..plotting.plot import watermark


class evaluate_data:
    
    def __init__(self, events, ncuts=20):
        self.events = events
        self.ncuts = ncuts
        self.hits_curler = self.curler()

    def curler(self):
          return self.events.assign(curler = self.events.groupby(['event_id', 'particle_id'])['layer_id'].diff() < 0)

    def find_pzcut(self):
        df = self.hits_curler
        curler = df[df.curler]
        nocurler = df[df.curler==False]
        N = len(curler) #curler=False
        P = len(nocurler) #nocurler=True
        Ntotal = N+P #number of hits

        cuts = np.linspace(0., 2., self.ncuts)
        purity = [] #true positive rate, sensitivity
        efficiency = [] #positive predictive value/ precision
        TNR = [] #correctly removed curlers
        FNR = [] #incorrectly removed no-curlers

        for pz_min in tqdm(cuts):
            FP = len(curler[curler.pz>pz_min]) #false positive
            TP = len(nocurler[nocurler.pz>pz_min]) #true positive
            purity.append(TP/P)
            efficiency.append(TP/(TP+FP))
            TNR.append(1 - FP/N)
            FNR.append(1 - TP/P)
        return purity, efficiency, TNR, FNR

    def plot_pzcut(self, cutPos=2):
        purity, efficiency, TNR, FNR = self.find_pzcut()
        plt.style.use("kit")
        cuts = np.linspace(0., 2., self.ncuts)

        fig, ax1 = plt.subplots(figsize=(8,6))
        

        ax1.plot(cuts, purity, marker='None', label='purity')
        ax1.set_ylim(top=1.01)
        ax1.set_ylabel('purity')

        ax2 = ax1.twinx()
        ax2.plot([], [], ' ')
        ax2.plot(cuts, efficiency, marker='None', label='efficiency', linestyle='--')
        ax2.axvline(cuts[cutPos], ymax=0.85, linestyle=':', color='black',label=f'best $p_z$ cut={cuts[cutPos]:.2f}')
        ax2.plot([], [], ' ', label=f'pur = {purity[cutPos]:.3f}')
        ax2.plot([], [], ' ', label=f'eff = {efficiency[cutPos]:.3f}')
        ax2.set_ylabel('efficiency')

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        watermark(py=0.9, fontsize=18, shift=0.16, scale=1.8)
        ax1.set_xlabel(r'$p_z$ cut (GeV)')
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=14, frameon = True, framealpha = 0.6, facecolor = 'white', edgecolor = 'white')
        plt.savefig('img/pz_cut.pdf', bbox_inches='tight')
        plt.show()

    def curler_dist(self):
        hits = self.hits_curler
        curler = hits[hits.curler]
        nocurler = hits[hits.curler==False]

        plt.style.use("kit_hist")
        plt.hist([nocurler.pz, curler.pz], histtype='stepfilled', facecolor='white',stacked=True, label=['no curler', 'curler'])
        plt.yscale('log')
        # plt.ylim(top=200000)
        watermark(py=0.9, fontsize=18, shift=0.16,scale=1.03)
        plt.xlabel(r'$p_z$ (GeV)')
        plt.ylabel('counts')
        plt.legend()
        plt.savefig('img/pzCut.pdf', bbox_inches='tight')
        plt.show()

class evaluate_graphs():
    
    def __init__(self, data, graphs, show_false=False, skip_duplicates=True, skip_same_layer=False):
        self.data = data
        self.graphs = graphs
        self.show_false = show_false
        self.skip_duplicates = skip_duplicates
        self.skip_same_layer = skip_same_layer
        
        
    def evaluate_graph(self, graph, hits):

        x = torch.from_numpy(graph.x)
        y = torch.from_numpy(graph.y)
        pid = graph.pid
        
        particle_ids = hits.particle_id.unique()
        hit_ids = np.unique(hits.hit_id)

        purities, efficiencies = [],[]
        bad_graph = []
        truth_edges = 0    

        # need to find the number of actual true edges in the data  
        for particle in particle_ids:
            particle_data = hits[hits['particle_id']==particle]
        # find all combination of layers possible for this particle 
            layers = np.array(particle_data.layer.values)
            lo, li = layers[:-1], layers[1:]
            layer_pairs = np.column_stack((lo, li))      
            if self.skip_duplicates:
                layer_pairs = np.unique(layer_pairs, axis=1) 
            if self.skip_same_layer:
                ids=[]
                for idx,lp in enumerate(layer_pairs):
                    if lp[0]==lp[1]:
                        ids.append(idx)
                layer_pairs = np.delete(layer_pairs,ids, axis=0)

            truth_edges += len(layer_pairs)

            if self.show_false:
                layerids = [1,2,7,8,9,10,15,16,17,18,23,24,25,26,31,32,33,34,39,40,41,42,47,48]
                valid_layer_pairs = np.stack((layerids[:-1], layerids[1:]), axis=1)
                for lp in layer_pairs:
                    if (lp==valid_layer_pairs).all(1).any():
                        continue
                    else:
                        print(lp)


        # number of edges labelled as true compared to actual true number of edges 
        if truth_edges==0:
            efficiency = 0
        else:
            efficiency = torch.sum(y).item()/truth_edges
#         print('number of labelled true edges', torch.sum(y), 'number of counted true edges', truth_edges)
        # number of true edges to total number of edges 
        purity = torch.sum(y).item()/len(y)
        if (efficiency > 1.0): 
            print('\nERROR: Efficiency>1!\n')
            bad_graph.append((graph,hits))

#         purities.append(purity)
#         efficiencies.append(efficiency)
        n_edges = len(y)
        n_nodes = len(x)
        n_particles_after = pid.nunique()
        n_particles_before = len(particle_ids)

        result = {'efficiency':efficiency, 'purity':purity, 'graph_edges':n_edges, 'graph_true_edges': torch.sum(y).item(), 'n_true_edges': truth_edges, 'graph_nodes':n_nodes, 'n_particles_before_cut':n_particles_before,  'n_particles_after_cut':n_particles_after}
        if efficiency > 1: print(result)
        return result, bad_graph
    
    def evaluate_graphs(self, show_progress=True):
        nevents = len(self.graphs)
        data = self.data
        efficiency, purity = [], []
        bad_graphs = []

#         for evt_id in tqdm(range(nevents)):
        for graph in tqdm(self.graphs, disable=not(show_progress)):
            evt_id = graph.pid.index.unique()[0]
            df = data.loc[evt_id]
            evaluation, bad_graph = self.evaluate_graph(graph, df)
            efficiency.append(evaluation['efficiency'])
            purity.append(evaluation['purity'])
            if bad_graph:
                bad_graphs.append(bad_graph)
        return purity, efficiency, bad_graphs
