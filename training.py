import torch
from torch_geometric.loader import DataLoader
from DataProcessing import read_data, make_SchNetlike_graphs
from models import GNN
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, loader, lossFunc, clip_value=1.0):
    model.train()
    total_loss = 0
    global batch_size
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Forward pass.
        loss = lossFunc(out, data.y)  # Loss computation.
        loss.backward()  # Backward pass.
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()  # Update model parameters.
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset) / batch_size


@torch.no_grad()
def test(model, loader, lossFunc):
    model.eval()
    total_loss = 0
    count = 0
    global batch_size
    for data in loader:
        data = data.to(device)
        pred = model(data)
        loss = lossFunc(pred, data.y)
        count += 1
        if count % 1024 == 0:
            print(pred[:5], data.y[:5])
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset) / batch_size

def save_checkpoint(model, optimizer, epoch, checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch}')

if __name__ == '__main__':
    files = [f'data/short.{i}.lammpstrj' for i in range(1, 21)]
    data = read_data(files)
    print('Data read')
    
    graphs = make_SchNetlike_graphs(data)
    print('Graphs made')

    graphs = shuffle(graphs, random_state=42)
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.1, random_state=42)
    batch_size = 1
    train_loader = DataLoader(train_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)
    print('Data loaded')

    model = GNN(1, 4, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    lossFunc = torch.nn.L1Loss(reduction='sum')

    test_losses = []
    train_losses = []
    for epoch in range(1, 100 ):
        loss = train(model, optimizer, train_loader, lossFunc, clip_value=1.0)
        test_loss = test(model, test_loader, lossFunc)
        test_losses.append(test_loss)
        train_losses.append(loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch:02d}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr*10**7:.2f}*10^(-7)')
        scheduler.step()

        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch)