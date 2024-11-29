import torch
from torch.nn import Sequential, Linear, GELU, BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv

class MPLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPLayer, self).__init__(aggr='sum')
        self.mlp = mlp(in_channels, out_channels)

    def forward(self, v,  edge_index, e):
        # Start propagating messages.
        accumulated_message= self.propagate(v=v, edge_index=edge_index, e=e)
        return accumulated_message

    def message(self, v_i, v_j, e):
        return self.mlp(v_i + v_j + e)

class mlp(torch.nn.Module):
    def __init__(self, in_channels, out_channel, hidden_dim=16, hidden_num=2, activation=GELU()):
        super().__init__()
        layers = [Linear(in_channels, hidden_dim), activation]
        for _ in range(hidden_num):
            layers.append(Linear(hidden_dim, hidden_dim))
            layers.append(activation)
        layers.append(Linear(hidden_dim, out_channel))
        self.mlp = Sequential(*layers)
        self._init_parameters()

    def _init_parameters(self):
         for layer in self.mlp:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
            return self.mlp(x)

class GNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, embedding_dim, out_dim, mp_num=3, activation=GELU()):
        super().__init__()
        torch.manual_seed(12345)
        self.node_encoder = mlp(node_dim, embedding_dim)
        self.edge_encoder = mlp(edge_dim, embedding_dim)
        self.message_passing = []
        norm_layer = BatchNorm1d(embedding_dim)
        for _ in range(mp_num):
            self.message_passing.append(norm_layer, MPLayer(embedding_dim, embedding_dim))
        self.decoder = mlp(embedding_dim, out_dim)
        
        
    def forward(self, data):
        v = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        for layer in self.message_passing:
            if isinstance(layer, MPLayer):
                v = v + layer(v, data.edge_index, e)    # Residual connection
            elif isinstance(layer, BatchNorm1d):
                v = layer(v)
        f = self.decoder(v)
        return f


        