import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

from .plot import watermark
from ..data.graphdata import GraphDataset

class plot_information:
    
    def __init__(self, events=None, graphs=None, nevents=None, pz_min=None, slope_max=None, name=''):
        self.name = name
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
        if self.nevents==None:
            self.nevents = len(self.events.index.unique(level=0))

        hits = nhits.to_numpy().T
        particles = nparticles.to_numpy().T

        plt.figure(figsize=(8,6))
        hist = plt.hist(particles, histtype='stepfilled', facecolor=(0,0,0,0))
        plt.yscale("log")
        plt.xlabel(r"$N_{particles}$")
        binwidth = np.mean(np.diff(hist[1]))
        plt.ylabel(f'Entries / ({binwidth:.2f})')
        infos = r'$N_{events}=$'+ f'{self.nevents}'
        watermark(scale=1.3, information=infos)
        plt.savefig(f"img/3_Nparticles_{self.name}.pdf")
        plt.show()

        hist = plt.hist(hits, histtype='stepfilled', facecolor=(0,0,0,0))
        plt.yscale("log")
        binwidth = np.mean(np.diff(hist[1]))
        plt.ylabel(f'Entries / ({binwidth:.2f})')
        plt.xlabel(r"$N_{hits}$")
        infos = r'$N_{events}=$'+ f'{self.nevents}'
        watermark(scale=1.3, information=infos)
        plt.savefig(f"img/3_Nhits_{self.name}.pdf")
        plt.show()
        
        print(f'mean number of particles: {np.mean(particles):.2f}, mean number of hits: {np.mean(hits):.2f}')
       
    
    def merge_graphs(self):
        
        nodes, edges = np.zeros(self.graphs[0].x.shape), np.zeros(self.graphs[0].edge_attr.shape)
        for g in tqdm(self.graphs):
            nodes = np.concatenate((nodes, g.x))
            edges = np.concatenate((edges, g.edge_attr), axis=1)
        
        return nodes, edges
    
        
    def plot_graph_data_info(self, bins=20, log=True):
        
        plt.style.use('kit')
        nodes, edges = self.merge_graphs()
        nevents = len(self.graphs)
        
        hist1 = plt.hist(nodes, bins=bins, label=['x','z','theta'], histtype='step', facecolor=(0,0,0,0), linewidth=1.5)
        
        if log: plt.yscale("log")
        plt.xlabel("node attributes")
        binwidth = np.mean(np.diff(hist1[1]))
        plt.ylabel(f'Entries / ({binwidth:.2f})')
        plt.legend(loc='upper right', frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white')
        infos = r'$N_{events}=$'+ f'{nevents},  ' + r'$p_z^{min}= $'+f'{self.pz_min} GeV/c'
        watermark(scale=1.3, information=infos)
        plt.savefig(f'img/3_node_attr_{self.name}.pdf')
        plt.show()
        
        plt.style.use('kit')
        hist2 = plt.hist(edges.T, bins=bins, label=['dx','dz','dtheta'], histtype='step', facecolor=(0,0,0,0), linewidth=1.5)

        if log: plt.yscale("log")
        plt.xlabel("edge attributes")
        binwidth = np.mean(np.diff(hist2[1]))
        plt.ylabel(f'Entries / ({binwidth:.2f})')
        plt.legend(loc='center right', frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white')
        infos = r'$N_{events}=$'+ f'{nevents},  ' + r'$p_z^{min}= $'+f'{self.pz_min} GeV/c' + r',  $s^{max}= $'+f'{self.slope_max}'
        watermark(scale=1.3, information=infos)
        plt.subplots_adjust(wspace=0.3)
        plt.savefig(f"img/3_edge_attr_{self.name}.pdf")
        plt.show()
        
        
    def plot_graph_dimensions(self, nnodes, nedges, slope, ntestevents):
    
        plt.style.use('kit')
        plt.errorbar(slope,nedges[:,0],nedges[:,1], linestyle='')
        plt.xlabel(r'$s^{max}$')
        plt.ylabel(r'$N_{edges}$')
        infos = r'$N_{events}=$'+ f'{ntestevents},  ' + r'$p_z^{min}= $'+f'{self.pz_min} GeV/c'
        watermark(scale=1.2, information=infos )
        plt.savefig(f"img/3_Nedges_afterfiltering_{self.name}.pdf")
        plt.show()
        
    
    def plot_purity_efficiency(self, cuts, cut_pos, purity, efficiency, TNR, FNR, nevents, variable=None, unit='', xname='threshold', yname=None, save_name=None, add_inf='', legloc='best', scale=1.3):           
        plt.style.use("kit")      
        
        plt.figure(figsize=(8,6))
        plt.plot(cuts, purity, label='purity', marker='None')
        plt.plot(cuts, efficiency, label='efficiency', marker='None')

        if cut_pos:
            plt.axvline(cuts[cut_pos], ymax=0.8, linestyle=':', color='black',label=f'{variable} = {cuts[cut_pos]:.3f} {unit}')
            plt.plot([], [], ' ', label=f'pur = {purity[cut_pos]:.3f}')
            plt.plot([], [], ' ', label=f'eff = {efficiency[cut_pos]:.3f}')
        
        infos = r'$N_{events}=$'+ f'{nevents}' + add_inf
        watermark(scale=scale, information=infos)
        plt.xlabel(xname)
        if yname:
            plt.ylabel(yname)
        plt.legend(loc=legloc, frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white')
        plt.savefig(save_name, bbox_inches='tight')
        plt.show() 
        
        print(f'best pz threshold at {cuts[cut_pos]:.4f}, removed bad (TNR): {TNR[cut_pos]:.3f}, lost good (FNR): {FNR[cut_pos]:.3f}')

        
        
class plot_event:
    
    def __init__(self, event=None, graph=None, scale=1.4, shift=0.13, name=''):
        self.name = name
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
        
    def get_edges(self, ids, edge_id=0):
        
        id1, id2 = ids
        X = np.array(self.graph.x)
        X_seg = np.vstack((X[int(id1)], X[int(id2)])).T
        
        x = X_seg[0]
        z = X_seg[1]
        
        dx, dz = self.graph.edge_attr.T[edge_id]
        slope = np.abs((dx*20)/(dz*100))
            
        return x, z, slope
    
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
        self.__plot_display(f'img/3_rawdata_event_{self.name}.pdf', f'event ID = {evID}')
        
        
    def plot_graphdisplay(self, pz_min, slope_max, show_slope=False, colormap='viridis', reverse_cm=True):
        '''
        A method to plot one graph in the 'x-z' projection for event ID: evID
        '''
        plt.style.use("kit")
        
        evID = self.graph.pid.index.unique()[0]
        infos = f'event ID = {evID},  ' + r'$p_z^{min}= $'+f'{pz_min} GeV/c' + r',  $s^{max}= $'+f'{slope_max}'
        segments = self.graph.edge_index
        segments = np.stack((segments[0], segments[1]), axis=1)

        y = np.array(self.graph.y)
                
        cmap = plt.get_cmap(colormap)
        if reverse_cm: cmap = cmap.reversed()
        norm = mpl.colors.Normalize(vmin=0, vmax=slope_max)
        
        plt.style.use("kit")
        

        for edge_id, seg in enumerate(segments):
            x, z, slope = self.get_edges(seg, edge_id) 
            
            if show_slope:
                plt.rcParams["figure.figsize"] = (12.5,6)                
                plt.plot(z*100, x*20, linewidth=1.0, linestyle='-', marker='None', c=cmap(norm(slope)))

                
            else:
                plt.rcParams["figure.figsize"] = (10,6)
                plt.plot(z*100, x*20, linewidth=1.0, linestyle='-', marker='None', color='black') 


        ids = np.unique(self.graph.pid)    
        for pid in ids:
            x, z = self.get_hits(pid)               
            plt.plot(z*100, x*20, linestyle='None', label=f'MC particle {pid:.0f}')
            
        if show_slope:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#             sm.set_array([0,slope_max]) 
            plt.colorbar(sm, label='slope')       
            
            
        self.__plot_display(f'img/3_graph_event_{self.name}.pdf', f'event ID = {evID}')
        
    def plot_traineddisplay(self, model, pz_min, slope_max, disc=0, device='cpu'):       
        '''
        A method to plot one graph of the trained model in the 'x-z' projection for event ID: evID
        '''
        
        plt.style.use("kit")
        
        evID = self.graph.pid.index.unique()[0]
        infos = f'event ID = {evID},  ' + r'$p_z^{min}= $'+f'{pz_min} GeV/c' + r',  $s^{max}= $'+f'{slope_max}'
        data = GraphDataset(self.graph)[0]
        output = model(data)

        X = np.array(data.x)
        segments = data.edge_index.T
        p_t = np.hstack((segments, np.vstack(output.detach().numpy())))
        
        plt.figure(figsize=(12.5,6))
        cmap = plt.get_cmap('viridis')

        for row in p_t:
            id1, id2, output = row
            if output < disc: continue
                
            x, z, slope = self.get_edges([id1, id2]) 
            plt.plot(z*100, x*20, linewidth=1.0, linestyle='-', marker='None', c=cmap.reversed()(output)) 

        ids = np.unique(data.pid.T[1])    
        for pid in ids:
            x, z = self.get_hits(pid)
            
            plt.plot(z*100, x*20, linestyle='None', label=f'MC particle {pid:.0f}')

        sm = plt.cm.ScalarMappable(cmap=cmap.reversed())
        sm.set_array([]) 
        plt.colorbar(sm, label='GNN output')
        
        self.__plot_display(f'img/3_trained_event_{self.name}.pdf', infos)