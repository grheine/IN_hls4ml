import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .plot import watermark
from ..data.graphdata import GraphDataset

class plot_information:
    
    def __init__(self, events=None, graphs=None):
        self.events = events
        self.graphs = graphs

    def plot_ntracks_nhits(self):
        # check how many particles and hits there are per event
        plt.style.use("kit_hist")
        nparticles = self.events.groupby('event_id')['particle_id'].nunique()
        nhits = self.events.groupby('event_id')['hit_id'].nunique()

        hits = nhits.to_numpy().T
        particles = nparticles.to_numpy().T

        plt.figure(figsize=(15,6))
        plt.subplot(121)
        hist = plt.hist(particles, histtype='stepfilled', facecolor=(0,0,0,0))
        plt.yscale("log")
        plt.xlabel("particles per event")
        plt.ylabel("counts")
        watermark(py=0.9, shift=0.2)
        plt.subplot(122)
        hist = plt.hist(hits, histtype='stepfilled', facecolor=(0,0,0,0))
        plt.yscale("log")
        plt.xlabel("hits per event")
        plt.ylabel("counts")
        watermark(shift=0.2)
        plt.subplots_adjust(wspace=0.3)
        plt.savefig("img/NtracksAndHits.pdf")
        plt.show()
        print(f'mean number of particles: {np.mean(particles):.2f}, mean number of hits: {np.mean(hits):.2f}')
       
    
    def merge_graphs(self):
        
        nodes, edges, _, _, _ = self.graphs[0]
        for g in tqdm(self.graphs):
            nodes = np.concatenate((nodes, np.fliplr(g.x)))
            edges = np.concatenate((edges, g.edge_attr), axis=1)
        
        return nodes, edges
    
        
    def plot_graph_data_info(self):
        plt.style.use("kit_hist")
        
        nodes, edges = self.merge_graphs()
        
        plt.figure(figsize=(15,6))
        plt.subplot(121)
        hist1 = plt.hist(nodes, label=['theta','z','x'], histtype='stepfilled', facecolor=(0,0,0,0), stacked=True)
        plt.yscale("log")
        plt.xlabel("node attributes")
        plt.ylabel("counts")
        plt.legend(loc='upper right', frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white', fontsize=12)
        watermark(shift=0.2)
        plt.subplot(122)
        hist2 = plt.hist(np.fliplr(edges.T), label=['dtheta','dz','dx'], histtype='stepfilled', facecolor=(0,0,0,0), stacked=True)
        plt.yscale("log")
        plt.xlabel("edge attributes")
        plt.ylabel("counts")
        plt.legend(loc='upper right', frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white', fontsize=12)
        watermark(shift=0.2)
        plt.subplots_adjust(wspace=0.3)
        plt.savefig("img/3_graphdata.pdf")
        plt.show()
        
        
    def plot_graph_dimensions(self, nnodes, nedges, slope):
        plt.style.use("kit_hist")
    
        plt.figure(figsize=(15,6))
        plt.subplot(121)
        hist3 = plt.hist(nnodes, histtype='stepfilled', stacked=True, facecolor=(0,0,0,0))
        plt.yscale("log")
        plt.xlabel("hits per event")
        plt.ylabel("counts")
        watermark(shift=0.2)
        plt.subplot(122)
        hist4 = plt.hist(nedges, histtype='stepfilled', stacked=True, facecolor=(0,0,0,0), label=list(map(str, slope)))
        plt.yscale("log")
        plt.xlabel("edges per event")
        plt.ylabel("counts")
        leg = plt.legend(title='slope cut')
        leg._legend_box.align = "right"
        leg.get_title().set_fontsize('15')
        watermark(shift=0.2)
        plt.subplots_adjust(wspace=0.3)
        plt.savefig("img/3_graph_dimensions.pdf")
        plt.show()  
        
    
    def plot_purity_efficiency(self, cuts, cut_pos, purity, efficiency, TNR, FNR, variable=None, xname='cut', yname=None, save_name=None):           
        plt.style.use("kit") 
        
        plt.figure(figsize=(8,6))
        plt.plot(cuts, purity, label='purity', marker='None')
        plt.plot(cuts, efficiency, label='efficiency', marker='None')

        if cut_pos:
            plt.axvline(cuts[cut_pos], ymax=0.8, linestyle=':', color='black',label=f'best {variable} cut={cuts[cut_pos]:.3f}')
            plt.plot([], [], ' ', label=f'pur = {purity[cut_pos]:.3f}')
            plt.plot([], [], ' ', label=f'eff = {efficiency[cut_pos]:.3f}')
        
        watermark(scale=1.5)
        plt.xlabel(xname)
        if yname:
            plt.ylabel(yname)
        plt.legend(loc='upper right', frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white', fontsize=12)
        plt.savefig(save_name, bbox_inches='tight')
        plt.show() 
        
        print(f'best pz cut at {cuts[cut_pos]:.4f}, removed bad (TNR): {TNR[cut_pos]:.3f}, lost good (FNR): {FNR[cut_pos]:.3f}')

        
        

class plot_event:
    
    def __init__(self, event=None, graph=None, scale=1.4, shift=0.13):
        self.event = event
        self.graph = graph
        self.scale = scale
        self.shift = shift
        
    def __plot_display(self, name, title=None, xlabel='z (cm)', ylabel='x (cm)', py=0.9, fontsize=18):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc='upper right', frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white', fontsize=12)
        watermark(py=py, fontsize=fontsize,  shift=self.shift, scale=self.scale)
        plt.savefig(name)
        plt.show()
        
    def get_edges(self, ids):
        
        id1, id2 = ids
        X = np.array(self.graph.x)
        
        if self.graph.x.shape[1] == 4:
            x,z,_,_ = np.vstack((X[int(id1)], X[int(id2)])).T
        elif self.graph.x.shape[1] == 3:
            x,z,_ = np.vstack((X[int(id1)], X[int(id2)])).T
        else:
            x,z = np.vstack((X[int(id1)], X[int(id2)])).T
            
        return x, z
    
    def get_hits(self, pid):
        
        X = np.array(self.graph.x)
        
        if self.graph.x.shape[1] == 4:
            x,z,_, iso = X[self.graph.pid==pid].T
        elif self.graph.x.shape[1] == 3:
            x,z,_ = X[self.graph.pid==pid].T 
        else:
            x,z = X[self.graph.pid==pid].T 
        
        return x, z
        
        
        

    def plot_eventdisplay(self):
        '''
        A method to plot one event in the 'x-z' projection for event ID: evID
        '''
        plt.style.use("kit")
        event = self.event
        ids = np.unique(event.particle_id)
        evID = event.index.unique()[0]

        plt.figure(figsize=(10,6))
        for pid in ids:
            df = event.loc[event.particle_id==pid]
            plt.scatter(df.z, df.x, s=df.iso*200, marker=r'$\odot$', linestyle='None', label=f'MC particle {pid:.0f}')
        self.__plot_display('img/3_rawdata_event.pdf', f'event ID = {evID}')
        
        
    def plot_graphdisplay(self):
        '''
        A method to plot one graph in the 'x-z' projection for event ID: evID
        '''
        segments = self.graph.edge_index
        segments = np.stack((segments[0], segments[1]), axis=1)

        y = np.array(self.graph.y)
        evID = self.graph.pid.index.unique()[0]
        plt.style.use("kit")
        plt.figure(figsize=(10,6))

        for seg in segments:
            x, z = self.get_edges(seg)                
            plt.plot(z*100, x*10, linewidth=1.0, linestyle='-', marker='None', color='black') 


        ids = np.unique(self.graph.pid)    
        for pid in ids:
            x, z = self.get_hits(pid)               
            plt.plot(z*100, x*10, linestyle='None', label=f'MC particle {pid:.0f}')
            
        self.__plot_display('img/3_graph_event.pdf', f'event ID = {evID}')
        
    def plot_traineddisplay(self, model, disc=0, device='cpu'):       
        '''
        A method to plot one graph of the trained model in the 'x-z' projection for event ID: evID
        '''
        
        plt.style.use("kit")
        data = GraphDataset(self.graph)[0]
        output = model(data)

        evID = self.graph.pid.index.unique()[0]
        X = np.array(data.x)
        segments = data.edge_index.T
        p_t = np.hstack((segments, np.vstack(output.detach().numpy())))
        
        plt.figure(figsize=(12.5,6))
        cmap = plt.get_cmap('viridis')

        for row in p_t:
            id1, id2, output = row
            if output < disc: continue
                
            x, z = self.get_edges([id1, id2]) 
            plt.plot(z*100, x*10, linewidth=1.0, linestyle='-', marker='None', c=cmap.reversed()(output),) 

        ids = np.unique(data.pid.T[1])    
        for pid in ids:
            x, z = self.get_hits(pid)
            
            plt.plot(z*100, x*10, linestyle='None', label=f'MC particle {pid:.0f}')

        sm = plt.cm.ScalarMappable(cmap=cmap.reversed())
        sm.set_array([]) 
        plt.colorbar(sm, label='GNN output')       

        self.__plot_display('img/3_trained_event.pdf', f'event ID = {evID}')
        
        
    def plot_tracklet_display(self, tracklets):
        '''
        A method to plot one graph in the 'x-z' projection for event ID: evID
        '''

        y = np.array(self.graph.y)
        evID = self.graph.pid.index.unique()[0]
        plt.style.use("kit")
        plt.figure(figsize=(10,6))

        for t in tracklets:
            x, z = t[:,1:3].T      
            plt.plot(z*100, x*10, linewidth=1.0, linestyle='-', marker='None') 


        ids = np.unique(self.graph.pid)    
        for pid in ids:
            x, z = self.get_hits(pid)               
            plt.plot(z*100, x*10, linestyle='None', label=f'MC particle {pid:.0f}')
            
        self.__plot_display('img/3_fulltracks.pdf', f'event ID = {evID}')