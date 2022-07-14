import torch
import torch_geometric
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, Sigmoid

from prettytable import PrettyTable


class RelationalModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RelationalModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, m):
        return self.layers(m)

class ObjectModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ObjectModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, C):
        return self.layers(C)


class InteractionNetwork(MessagePassing):
    def __init__(self, hidden_size=30):
        super(InteractionNetwork, self).__init__(aggr='add', flow='source_to_target')
        
        self.nodeD = 3
        self.edgeD = 4
        self.hidden_size = hidden_size
                
        self.R1 = RelationalModel(10, 4, hidden_size)
        self.O = ObjectModel(7, 3, hidden_size)
        self.R2 = RelationalModel(10, 1, hidden_size)
        self.E: Tensor = Tensor()

    def forward(self, data: Tensor) -> Tensor:
        x = data.x
        edge_index, edge_attr = data.edge_index, data.edge_attr
        
        x_tilde = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=None)
        m2 = torch.cat([x_tilde[edge_index[1]],
                        x_tilde[edge_index[0]],
                        self.E], dim=1)
        return torch.sigmoid(self.R2(m2))

    def message(self, x_i, x_j, edge_attr):
        # x_i --> incoming
        # x_j --> outgoing    
        m1 = torch.cat([x_i, x_j, edge_attr], dim=1)
        self.E = self.R1(m1)
        return self.E

    def update(self, aggr_out, x):
        c = torch.cat([x, aggr_out], dim=1)
        return self.O(c)
    
    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params+=params
        print(self)
        print(table)
        print(f"Total Trainable Params: {total_params}")
        
    def __str__(self):
        return f'InteractionNetwork(node_dim: {self.nodeD}, edge_dim: {self.edgeD}, hidden_size: {self.hidden_size})'
    
    