import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from .plot import watermark
from ..data.graphdata import GraphDataset

class plot_information:
    
    def __init__(self, events=None, graphs=None, nevents=30000, pz_min=None, slope_max=None):
        self.events = events
        self.graphs = graphs
        self.nevents = nevents
        self.pz_min = pz_min
        self.slope_max = slope_max

    def plot_ntracks_nhits(self):
        # check how many particles and hits there are per event
        
        plt.figure(figsize=(8,6))
        plt.style.use("kit_hist")
        nparticles = self.events.groupby('event_id')['particle_id'].nunique()
        nhits = self.events.groupby('event_id')['hit_id'].nunique()
        nevents = len(self.events.index.unique(level=0))

        hits = nhits.to_numpy().T
        particles = nparticles.to_numpy().T

        plt.figure(figsize=(8,6))
        hist = plt.hist(particles, histtype='stepfilled', facecolor=(0,0,0,0))
        plt.yscale("log")
        plt.xlabel(r"log($N_{particles}$)")
        binwidth = np.mean(np.diff(hist[1]))
        plt.ylabel(f'Entries / ({binwidth:.2f})')
        infos = r'$N_{events}=$'+ f'{nevents}'
        watermark(scale=1.3, information=infos)
        plt.savefig("img/3_Nparticles.pdf")
        plt.show()
        

        hist = plt.hist(hits, histtype='stepfilled', facecolor=(0,0,0,0))
        plt.yscale("log")
        binwidth = np.mean(np.diff(hist[1]))
        plt.ylabel(f'Entries / ({binwidth:.2f})')
        plt.xlabel(r"log($N_{hits}$)")
        infos = r'$N_{events}=$'+ f'{nevents}'
        watermark(scale=1.3, information=infos)
        plt.savefig("img/3_Nhits.pdf")
        plt.show()
        
        print(f'mean number of particles: {np.mean(particles):.2f}, mean number of hits: {np.mean(hits):.2f}')
       
    
    def merge_graphs(self):
        
        nodes, edges = np.zeros(self.graphs[0].x.shape), np.zeros(self.graphs[0].edge_attr.shape)
        for g in tqdm(self.graphs):
            nodes = np.concatenate((nodes, np.fliplr(g.x)))
            edges = np.concatenate((edges, g.edge_attr), axis=1)
        
        return nodes, edges
    
        
    def plot_graph_data_info(self):
        plt.style.use("kit_hist")
        
        nodes, edges = self.merge_graphs()
        nevents = len(self.graphs)
        
        hist1 = plt.hist(nodes, label=['x','z','theta'], histtype='stepfilled', facecolor=(0,0,0,0), stacked=True, hatch='///')
        plt.style.use('kit')
        plt.hist(nodes, histtype='step', stacked=True, facecolor=(0,0,0,0), edgecolor='black', linestyle='-', hatch=None)
        plt.yscale("log")
        plt.xlabel("log(node attributes)")
        binwidth = np.mean(np.diff(hist1[1]))
        plt.ylabel(f'Entries / ({binwidth:.2f})')
        plt.legend(loc='upper right', frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white')
        infos = r'$N_{events}=$'+ f'{nevents},  ' + r'$p_z^{min}= $'+f'{self.pz_min}'
        watermark(scale=1.3, information=infos)
        plt.savefig('img/3_node_attr.pdf')
        plt.show()
        

        plt.style.use('kit_hist')
        hist2 = plt.hist(edges.T, label=['dx','dz','dtheta'], histtype='stepfilled', stacked=True, facecolor=(0,0,0,0) , hatch='///')
        plt.style.use('kit')
        plt.hist(edges.T, histtype='step', stacked=True, facecolor=(0,0,0,0), edgecolor='black', linestyle='-', hatch=None)
        plt.yscale("log")
        plt.xlabel("log(edge attributes)")
        binwidth = np.mean(np.diff(hist2[1]))
        plt.ylabel(f'Entries / ({binwidth:.2f})')
        plt.legend(loc='center right', frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white')
        infos = r'$N_{events}=$'+ f'{nevents},  ' + r'$p_z^{min}= $'+f'{self.pz_min}' + r',  $s^{max}= $'+f'{self.slope_max}'
        watermark(scale=1.3, information=infos)
        plt.subplots_adjust(wspace=0.3)
        plt.savefig("img/3_edge_attr.pdf")
        plt.show()
        return(hist1)
        
        
    def plot_graph_dimensions(self, nnodes, nedges, slope, ntestevents):
        plt.style.use("kit_hist")
    
        plt.style.use('kit')
        plt.errorbar(slope,nedges[:,0],nedges[:,1], linestyle='')
        plt.xlabel(r'$s^{max}$')
        plt.ylabel(r'$N_{edges}$')
        infos = r'$N_{events}=$'+ f'{ntestevents},  ' + r'$p_z^{min}= $'+f'{self.pz_min}'
        watermark(scale=1.1, information=infos )
        plt.savefig("img/3_Nedges_afterfiltering.pdf")
        plt.show()
        
    
    def plot_purity_efficiency(self, cuts, cut_pos, purity, efficiency, TNR, FNR, nevents, variable=None, xname='threshold', yname=None, save_name=None, add_inf=''):           
        plt.style.use("kit")      
        
        plt.figure(figsize=(9,6))
        plt.plot(cuts, purity, label='purity', marker='None')
        plt.plot(cuts, efficiency, label='efficiency', marker='None')

        if cut_pos:
            plt.axvline(cuts[cut_pos], ymax=0.8, linestyle=':', color='black',label=f'best {variable} = {cuts[cut_pos]:.3f}')
            plt.plot([], [], ' ', label=f'pur = {purity[cut_pos]:.3f}')
            plt.plot([], [], ' ', label=f'eff = {efficiency[cut_pos]:.3f}')
        
        infos = r'$N_{events}=$'+ f'{nevents}' + add_inf
        watermark(scale=1.7, information=infos, shift=0.14)
        plt.xlabel(xname)
        if yname:
            plt.ylabel(yname)
        plt.legend(loc='upper right', frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white')
        plt.savefig(save_name, bbox_inches='tight')
        plt.show() 
        
        print(f'best pz threshold at {cuts[cut_pos]:.4f}, removed bad (TNR): {TNR[cut_pos]:.3f}, lost good (FNR): {FNR[cut_pos]:.3f}')

        
        
class plot_event:
    
    def __init__(self, event=None, graph=None, scale=1.4, shift=0.13):
        self.event = event
        self.graph = graph
        self.scale = scale
        self.shift = shift
        
    def __plot_display(self, name, title=None, xlabel='z (cm)', ylabel='x (cm)', py=0.9, fontsize=18):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='upper right', frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white', fontsize=12)
        watermark(py=py, fontsize=fontsize,  shift=self.shift, scale=self.scale, information=title)
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
        X_particle = X[self.graph.pid==pid].T
        
        x = X_particle[0]
        z = X_particle[1]
        
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
        plt.style.use("kit")
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