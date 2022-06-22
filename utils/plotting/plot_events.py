import numpy as np
import matplotlib.pyplot as plt
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

        plt.figure(figsize=(12,5))
        plt.subplot(121)
        hist = plt.hist(particles, histtype='stepfilled', facecolor=(0,0,0,0))
        plt.yscale("log")
        plt.xlim(hist[1][0], hist[1][-1])
        plt.xlabel("particles per event", fontsize=15)
        plt.ylabel("counts", fontsize=15)
        watermark(py=0.9, fontsize=15, shift=0.2)
        plt.subplot(122)
        hist = plt.hist(hits, histtype='stepfilled', facecolor=(0,0,0,0))
        plt.yscale("log")
        plt.xlim(hist[1][0], hist[1][-1])
        plt.xlabel("hits per event", fontsize=15)
        plt.ylabel("counts", fontsize=15)
        watermark(py=0.9, fontsize=15, shift=0.2)
        plt.subplots_adjust(wspace=0.3)
        plt.savefig("img/NtracksAndHits.pdf")
        plt.show()
        
    def plot_graph_information(self):
        plt.style.use("kit_hist")
        
        nodes, edges, _, _, _ = self.graphs[0]
        nedges, nnodes = [], []



        for g in self.graphs:
            nodes = np.concatenate((nodes, g.x))
            edges = np.concatenate((edges, g.edge_attr), axis=1)
            nedges.append(len(g.y))
            nnodes.append(len(g.x))
        
        plt.figure()
        plt.subplot(121)
        hist1 = plt.hist(nodes, label=['x','z','theta','isochrone'], histtype='stepfilled', facecolor=(0,0,0,0), stacked=True)
        plt.yscale("log")
        plt.xlabel("node attributes")
        plt.ylabel("counts")
        plt.legend(loc='upper right', frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white', fontsize=12)
        watermark(shift=0.2)
        plt.subplot(122)
        hist2 = plt.hist(edges.T, label=['dx','dz','dtheta'], histtype='stepfilled', facecolor=(0,0,0,0), stacked=True)
        plt.yscale("log")
        plt.xlabel("edge attributes")
        plt.ylabel("counts")
        plt.legend(loc='upper right', frameon = True, framealpha = 0.8, facecolor = 'white', edgecolor = 'white', fontsize=12)
        watermark(shift=0.2)
        plt.subplots_adjust(wspace=0.3)
        plt.savefig("img/graphdata.pdf")
        plt.show()
    
        plt.figure()
        plt.subplot(121)
        hist3 = plt.hist(nnodes, histtype='stepfilled', facecolor=(0,0,0,0))
        plt.yscale("log")
        plt.xlabel("# nodes")
        plt.ylabel("counts")
        watermark(shift=0.2)
        plt.subplot(122)
        hist4 = plt.hist(nedges, histtype='stepfilled', facecolor=(0,0,0,0))
        plt.yscale("log")
        plt.xlabel("# edges")
        plt.ylabel("counts")
        watermark(shift=0.2)
        plt.subplots_adjust(wspace=0.3)
        plt.savefig("img/NnodesAndEdges.pdf")
        plt.show()      
        
        

class plot_event:
    
    def __init__(self, event=None, graph=None, scale=1.4, shift=0.16):
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
        plt.savefig('img/rawdata_event.pdf')
        plt.show()
        

    def plot_eventdisplay(self):
        '''
        A method to plot one event in the 'x-z' projection for event ID: evID
        '''
        plt.style.use("kit")
        event = self.event
        ids = np.unique(event.particle_id)
        evID = event.index.unique()[0]

        plt.figure(figsize=(10,6))
        for ID in ids:
            df = event.loc[event.particle_id==ID]
            plt.scatter(df.z, df.x, s=df.iso*200, marker=r'$\odot$', linestyle='None', label=f'MC particle {ID:.0f}')
        self.__plot_display('img/rawdata_event.pdf', f'event ID = {evID}')
        
        
    def plot_graphdisplay(self):
        '''
        A method to plot one graph in the 'x-z' projection for event ID: evID
        '''
        segments = self.graph.edge_index
        segments = np.stack((segments[0], segments[1]), axis=1)

        X = np.array(self.graph.x)
        y = np.array(self.graph.y)
        evID = self.graph.pid.index.unique()[0]
        plt.style.use("kit")
        plt.figure(figsize=(10,6))

        for seg in segments:
            id1, id2 = seg
            x,z,theta,iso = np.vstack((X[int(id1)], X[int(id2)])).T
            plt.plot(z*100, x*10, linewidth=1.0, linestyle='-', marker='None', color='black') 

        ids = np.unique(self.graph.pid)    
        for ID in ids:
            x,z,_, iso = X[self.graph.pid==ID].T
            plt.plot(z*100, x*10, linestyle='None', label=f'MC particle {ID:.0f}')
        self.__plot_display('img/graph_event.pdf', f'event ID = {evID}')
        
    def plot_traineddisplay(self, model, disc=0, device='cpu'):       
        '''
        A method to plot one graph of the trained model in the 'x-z' projection for event ID: evID
        '''
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
            if output < 0.: continue
            x,z,theta, iso = np.vstack((X[int(id1)], X[int(id2)])).T
            plt.plot(z*100, x*10, linewidth=1.0, linestyle='-', marker='None', c=cmap.reversed()(output),) 

        ids = np.unique(data.pid.T[1])    
        for ID in ids:
            x,z,_, iso = X[data.pid.T[1]==ID].T
            plt.plot(z*100, x*10, linestyle='None', label=f'MC particle {ID:.0f}')

        sm = plt.cm.ScalarMappable(cmap=cmap.reversed())
        sm.set_array([]) 
        plt.colorbar(sm, label='GNN output')       

        self.__plot_display('img/trained_event.pdf', f'event ID = {evID}')