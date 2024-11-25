from torch_geometric.loader import DataLoader
import torch
from torch_geometric.data import DataLoader
from DataProcessing import read_data, make_basic_graphs
from models import MessagePassingNetwork
from sklearn.model_selection import train_test_split

data = read_data('data/dump.1.lammpstrj')
print('Data read')
graphs = make_basic_graphs(data)
print('Graphs made')
train_graphs, test_graphs = train_test_split(graphs, test_size=0.1, random_state=42)

train_loader = DataLoader(train_graphs, batch_size=10, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=10)
print('Data loaded')

model = MessagePassingNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lossFunc = torch.nn.MSELoss()

def train(model, optimizer, loader):
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
def test(model, loader):
    model.eval()
    total_correct = 0
    total_loss = 0
    for data in loader:
        pred = model(data)
        loss = lossFunc(pred, data.y)
        total_loss += loss.item() * data.num_graphs
        total_correct += int(( (pred > 0.5)== data.y).sum())
    return total_correct / len(loader.dataset), total_loss / len(train_loader.dataset)

test_losses = []
train_losses = []
for epoch in range(1, 100 ):
    loss = train(model, optimizer, train_loader)
    test_acc,test_loss = test(model, test_loader)
    test_losses.append(test_loss)
    train_losses.append(loss)
    print(f'Epoch: {epoch:02d}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')