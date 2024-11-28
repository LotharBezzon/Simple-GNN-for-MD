from torch_geometric.loader import DataLoader
import torch
from torch_geometric.loader import DataLoader
from DataProcessing import read_data, make_basic_graphs, make_SchNetlike_graphs
from models import BasicMessagePassingNetwork, GATNetwork
from sklearn.model_selection import train_test_split


def train(model, optimizer, loader, lossFunc):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Forward pass.
        loss = lossFunc(out, data.y)  # Loss computation.
        loss.backward()  # Backward pass.
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader, lossFunc):
    model.eval()
    total_loss = 0
    for data in loader:
        pred = model(data)
        print(pred, data.y)
        loss = lossFunc(pred, data.y)
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

if __name__ == '__main__':
    data = read_data('data/dump.1.lammpstrj')
    print('Data read')
    graphs = make_SchNetlike_graphs(data[:10])
    #graphs = make_basic_graphs(data)
    print('Graphs made')
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.1, random_state=42)

    train_loader = DataLoader(train_graphs, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=10)
    print('Data loaded')

    model = GATNetwork(2, 3)
    #model = BasicMessagePassingNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    lossFunc = torch.nn.MSELoss(reduction='sum')

    test_losses = []
    train_losses = []
    for epoch in range(1, 100 ):
        loss = train(model, optimizer, train_loader, lossFunc)
        test_loss = test(model, test_loader, lossFunc)
        test_losses.append(test_loss)
        train_losses.append(loss)
        print(f'Epoch: {epoch:02d}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}')