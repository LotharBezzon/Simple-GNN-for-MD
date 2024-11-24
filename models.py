import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing, GCNConv

class BasicMPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(BasicMPNN, self).__init__(aggr='mean')
        self.mlp = Sequential(Linear(in_channels, out_channels),
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
        return self.mlp(h_j + edge_attr)

class MessagePassingNetwork(torch.nn.Module):
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
        h = h.relu()
        h = self.pass3(h=h, edge_index=edge_index, edge_attr=edge_attr)

        h = torch.sigmoid(self.classifier(h))
        return h