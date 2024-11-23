import torch
from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.data import DataLoader
from DataProcessing import read_data, make_basic_graphs

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
        accumulated_message= self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return accumulated_message

    def message(self, h_j, edge_attr):
        # x_j has shape [E, out_channels]
        # edge_attr has shape [E, edge_features]
        return h_j + edge_attr

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out

# Example usage
if __name__ == "__main__":
    data = read_data('data/dump.1.lammpstrj')
    graphs = make_basic_graphs(data)
    '''loader = DataLoader(graphs, batch_size=32, shuffle=True)

    model = BasicMPNN(in_channels=4, out_channels=3)  # Adjust in_channels and out_channels as needed
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = F.mse_loss(out, batch.y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item()}')'''