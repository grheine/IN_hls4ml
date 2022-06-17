import os
import torch
import numpy as np
from torchvision import transforms
from torch_geometric.data import Data, Dataset

class GraphDataset(Dataset):
    def __init__(self, graphs, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(None, transform, pre_transform)
        self.graphs = graphs
    
    def len(self):
        return len(self.graphs)
        
    def get(self, idx):
        if np.array(self.graphs, dtype=object).ndim==1:
            f = self.graphs
        else:
            f = self.graphs[idx]
            
        x = torch.from_numpy(f.x)#, dtype=torch.float32)
        edge_attr = torch.from_numpy(f.edge_attr)#, dtype=torch.float32)
        edge_index = torch.from_numpy(f.edge_index)
        y = torch.from_numpy(f.y)#, dtype=torch.uint8)
        pid = torch.from_numpy(np.array(f.pid.reset_index('event_id')))#, dtype=torch.uint8)
#             print(x.shape, edge_attr.shape, edge_index.shape, y.shape)
        data = Data(x=x.float(), edge_index=edge_index, edge_attr=torch.transpose(edge_attr.float(), 0, 1),pid=pid, y=y.float())
        data.num_nodes = len(x)

        return data