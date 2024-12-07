import torch
from torch.nn import Sequential, Linear, GELU, BatchNorm1d, Dropout, LayerNorm, ReLU, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv

class mlp(torch.nn.Module):
    def __init__(self, in_channels, out_channel, hidden_dim=128, hidden_num=1, activation=ReLU()):
        super().__init__()
        #normalization = BatchNorm1d(in_channels)
        self.layers = [Linear(in_channels, hidden_dim), activation]
        for _ in range(hidden_num):
            self.layers.append(Dropout(0.1))
            self.layers.append(Linear(hidden_dim, hidden_dim))
            self.layers.append(activation)
        self.layers.append(Linear(hidden_dim, out_channel))
        self.mlp = Sequential(*self.layers)
        self._init_parameters()

    def _init_parameters(self):
         for layer in self.mlp:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
            return self.mlp(x)
    
class MPLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.mlp = mlp(2*in_channels, out_channels)
        self.norm_layer = BatchNorm1d(in_channels)

    def forward(self, edge_index, v,  e):
        # Start propagating messages.
        accumulated_message= self.propagate(edge_index, v=v, e=e)
        return accumulated_message

    def message(self, v_i, v_j, e):
        return self.mlp(torch.cat([v_i + v_j, e], dim=-1))

class GNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, embedding_dim=128, mp_num=4):
        super().__init__()
        torch.manual_seed(12345)
        self.node_encoder = mlp(node_dim, embedding_dim)
        self.edge_encoder = mlp(edge_dim, embedding_dim)
        self.message_passing_layers = ModuleList([])
        self.norm_layer = BatchNorm1d(embedding_dim)
        for _ in range(mp_num):
            self.message_passing_layers.append(BatchNorm1d(embedding_dim))
            self.message_passing_layers.append(MPLayer(embedding_dim, embedding_dim))
        self.decoder = mlp(embedding_dim, out_dim)
        
        
    def forward(self, data):
        v = self.node_encoder(data.x)
        #print("After node_encoder:", v)
        
        e = self.edge_encoder(data.edge_attr)
        #print("After edge_encoder:", e)
        
        e = self.norm_layer(e)
        #print("After norm_layer:", e)
        
        for layer in self.message_passing:
            if isinstance(layer, MPLayer):
                v = v + layer(data.edge_index, v, e)  # Residual connection
                #print("After MPLayer:", v)
            else:
                v = layer(v)
                #print("After BatchNorm1d:", v)
        
        f = self.decoder(v)
        #print("After decoder:", f)
        
        return f

class SimpleMPLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.mlp = mlp(in_channels, out_channels)
        self.norm_layer = BatchNorm1d(in_channels)

    def forward(self, edge_index, v,  e):
        # Start propagating messages.
        #v = self.norm_layer(v)
        accumulated_message= self.propagate(edge_index, v=v, e=e)
        return accumulated_message

    def message(self, v_i, v_j, e):
        return self.mlp(torch.cat([v_i + v_j, e], dim=-1))

class SimpleGNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, out_dim, embedding_dim=128, mp_num=4):
        super().__init__()
        torch.manual_seed(12345)
        self.message_passing_layers = ModuleList([SimpleMPLayer(node_dim + edge_dim, embedding_dim)])
        for _ in range(mp_num-2):
            self.message_passing_layers.append(BatchNorm1d(embedding_dim))
            self.message_passing_layers.append(SimpleMPLayer(embedding_dim+edge_dim, embedding_dim))
        self.message_passing_layers.append(SimpleMPLayer(embedding_dim+edge_dim, out_dim))
        #self.message_passing = Sequential(*self.message_passing_layers)
        
    def forward(self, data):
        v = self.message_passing_layers[0](data.edge_index, data.x, data.edge_attr)
        for layer in self.message_passing_layers[1:-1]:
            if isinstance(layer, SimpleMPLayer):
                v = v + layer(data.edge_index, v, data.edge_attr)
            else:
                v = layer(v)
        f = self. message_passing_layers[-1](data.edge_index, v, data.edge_attr)

        return f
    