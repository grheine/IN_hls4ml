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
        diff = self.events.groupby(['event_id'])[['layer_id', 'particle_id']].diff()
        back = (diff.layer_id < 0) & (diff.particle_id == 0)
        s1 = self.events.assign(back=back).set_index('particle_id', append=True)
        s2 = s1.rename(columns={'back':'curler'}).groupby(['event_id', 'particle_id'])['curler'].sum()>0
        s3 = s1.merge(s2, left_index=True, right_on=['event_id', 'particle_id'])        
        return s3
    
#     def pz_cut(self, hits):
#         df = hits.groupby(['event_id', 'particle_id']).pz.mean().rename('curl') < self.pz_min
# #         hits.set_index(['particle_id'], inplace = True, append = True)
#         ev = hits.merge(df, left_index=True, right_on=['event_id', 'particle_id'])
#         df2 = ev[ev.curl==False].drop(columns=['curl'])
#         return df2

    def find_pzcut(self):
        df = self.hits_curler
        curler = df[df.curler]
        nocurler = df[df.curler==False]
        N = len(curler.index.unique()) #curler=False
        P = len(nocurler.index.unique()) #nocurler=True
        Ntotal = N+P #number of hits

        cuts = np.linspace(-0.05, 0.05, self.ncuts)
        purity = [] #positive predictive value/ precision
        efficiency = [] #true positive rate, sensitivity    
        TNR = [] #correctly removed curlers
        FNR = [] #incorrectly removed no-curlers

        for pz_min in tqdm(cuts):
            FP = (curler.groupby(['event_id', 'particle_id']).pz.mean()>pz_min).sum() #false positive
            TP = (nocurler.groupby(['event_id', 'particle_id']).pz.mean()>pz_min).sum() #true positive
            purity.append(TP/(TP+FP))
            efficiency.append(TP/P)

            TN = (curler.groupby(['event_id', 'particle_id']).pz.mean()<=pz_min).sum()
            FN = (nocurler.groupby(['event_id', 'particle_id']).pz.mean()<=pz_min).sum()
            TNR.append(TN/N)
            FNR.append(FN/P)
            
        cutPos =  np.argmax((np.array(purity)+np.array(efficiency))[1:]) +1
        best_cut = cuts[cutPos]
        print(f'best pz cut at {best_cut}, removed curlers (TNR): {TNR[cutPos]}, lost no-curlers (FNR): {FNR[cutPos]}')
        
        return purity, efficiency, cuts, cutPos, TNR, FNR

    def plot_pzcut(self, eff_scale=1.):
        plt.style.use("kit")
        purity, efficiency, cuts, cutPos, TNR, FNR = self.find_pzcut()
        cutPos = 0.005

        fig, ax1 = plt.subplots(figsize=(8,6))
        

        ax1.plot(cuts, purity, marker='None', label='purity')
        ax1.set_ylabel('purity')
        watermark(py=0.9, fontsize=18, shift=0.16, scale=1.2)

        ax2 = ax1.twinx()
        ax2.plot([], [], ' ')
        ax2.plot(cuts, efficiency, marker='None', label='efficiency', linestyle='--')
        ax2.axvline(cuts[cutPos], ymax=0.8, linestyle= (0, (1, 10)), color='black',label=f'best $p_z$ cut={cuts[cutPos]:.2f}')
        ax2.plot([], [], ' ', label=f'pur = {purity[cutPos]:.3f}')
        ax2.plot([], [], ' ', label=f'eff = {efficiency[cutPos]:.3f}')
        ax2.set_ylabel('efficiency')

        bottomylim, topylim = ax2.get_ylim()
        ax2.set_ylim(top=bottomylim+(topylim-bottomylim)*eff_scale)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        
        ax1.set_xlabel(r'$p_z$ cut (GeV)')
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=14, frameon = True, framealpha = 0.6, facecolor = 'white', edgecolor = 'white')
        plt.savefig('img/3_pz_cut.pdf', bbox_inches='tight')
        plt.show()
        

    def curler_dist(self):
        fig = plt.figure(figsize=(8,6))
        df = self.hits_curler
        curler = df[df.curler].groupby(['event_id', 'particle_id']).pz.mean()
        nocurler = df[df.curler == False].groupby(['event_id', 'particle_id']).pz.mean()
        print(f'number of no-curlers: {len(nocurler)}, number of curlers: {len(curler)}, proportion of curlers: {len(curler)/len(nocurler)}')
        plt.style.use("kit_hist")
        plt.hist(curler, bins=20, histtype='stepfilled', facecolor='white', label='curler')
        plt.yscale('log')
        watermark(py=0.9, fontsize=18, shift=0.16, scale=1.2)
        plt.xlabel(r'$p_z$ (GeV)')
        plt.ylabel('counts')
        plt.legend()
        plt.savefig('img/3_curler_histo.pdf', bbox_inches='tight')
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

        # metrics
        Ntotal = 0.5 * len(x) * (len(x)-1)
        P = truth_edges
        N = Ntotal - P
        TP = torch.sum(y).item()
        FP = len(y) - TP
        FN = P - TP
        TN = N - FP
        
        TNR = TN/N
        FNR = FN/P        

        # number of edges labelled as true compared to actual true number of edges 
        if truth_edges==0:
            efficiency = 0
        else:
            efficiency = TP/P
        
        if (efficiency > 1.0): 
            print('\nERROR: Efficiency>1!\n')
            bad_graph.append((graph,hits))

        # number of true edges to total number of edges 
        purity = TP/len(y)

        n_edges = len(y)
        n_nodes = len(x)
        n_particles_after = pid.nunique()
        n_particles_before = len(particle_ids)

        result = {
            'efficiency':efficiency, 
            'purity':purity, 
            'graph_true_edges': TP, 
            'n_true_edges': P,
            'TNR': TNR,
            'FNR': FNR,
            'graph_edges':n_edges, 
            'graph_nodes':n_nodes, 
            'n_particles_before_cut':n_particles_before,  
            'n_particles_after_cut':n_particles_after
        }
            
        return result, bad_graph
    
    def evaluate_graphs(self, show_progress=True):
        nevents = len(self.graphs)
        data = self.data
        efficiency, purity, TNR, FNR = [], [], [], []
        bad_graphs = []

#         for evt_id in tqdm(range(nevents)):
        for graph in tqdm(self.graphs, disable=not(show_progress)):
            evt_id = graph.pid.index.unique()[0]
            df = data.loc[evt_id]
            evaluation, bad_graph = self.evaluate_graph(graph, df)
            efficiency.append(evaluation['efficiency'])
            purity.append(evaluation['purity'])
            TNR.append(evaluation['TNR'])
            FNR.append(evaluation['FNR'])
            if bad_graph:
                bad_graphs.append(bad_graph)
        return np.mean(purity), np.mean(efficiency), np.mean(TNR), np.mean(FNR), bad_graphs
