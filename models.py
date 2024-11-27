import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv

class BasicMPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(BasicMPNN, self).__init__(aggr='mean')
        self.mlp = Sequential(Linear(in_channels+3, out_channels), # 3 is the number of edge features
                              ReLU(),
                              Linear(out_channels, out_channels),
                             # ReLU()
                              )

    def forward(self, h,  edge_index, edge_attr):
        # Start propagating messages.
        accumulated_message= self.propagate(h=h, edge_index=edge_index, edge_attr=edge_attr)
        return accumulated_message

    def message(self, h_j, edge_attr):
        # x_j has shape [E, out_channels]
        # edge_attr has shape [E, edge_features]
        input = torch.cat([h_j, edge_attr], dim=-1)
        return self.mlp(input)

class BasicMessagePassingNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        N_features=20
        torch.manual_seed(12345)
        self.pass1 = BasicMPNN(4, N_features)
        self.pass2 = BasicMPNN(N_features, N_features)
        self.classifier = Linear(N_features, 3)

    def forward(self, data):
        edge_index=data.edge_index
        edge_attr=data.edge_attr
        x=data.x

        h = self.pass1(h=x , edge_index=edge_index, edge_attr=edge_attr)
        h = h.relu()
        h = self.pass2(h=h,edge_index=edge_index, edge_attr=edge_attr)

        h = torch.sigmoid(self.classifier(h))
        return h

class NodeEncoding(torch.nn.Module):
    def __init__(self, node_dim, encoding_dim):
        super().__init__()
        self.layer1 = Linear(node_dim, 10)
        self.layer2 = Linear(10, encoding_dim)

    def forward(self, data):
        h = self.layer1(data.x)
        h = ReLU(h)
        h = self.layer2(h)
        data.x = h
        return data

class GATNetwork(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        N_features=20
        torch.manual_seed(12345)
        self.conv1 = GATConv(in_channels, N_features, edge_dim=7, heads=4)
        self.conv2 = GATConv(N_features, N_features, edge_dim=7)
        self.decoder = Sequential(Linear(N_features, 10),
                                  ReLU(),
                                  Linear(10, out_channels))
        
    def forward(self, data):
        data.x = F.dropout(data.x, p=0.6, training=self.training)
        h = self.conv1(data.x, data.edge_index, data.edge_attr)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.pass2(h, data.edge_index, data.edge_attr)
        h = F.elu(h)
        h = self.decoder(h)
        return h


        