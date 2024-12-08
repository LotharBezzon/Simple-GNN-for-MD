import torch
from torch_geometric.loader import DataLoader
from DataProcessing import read_data, make_graphs
from models import GNN, GATModel
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, optimizer, loader, lossFunc, clip_value=1.0):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Forward pass.
        loss = lossFunc(out, data.y)  # Loss computation.
        loss.backward()  # Backward pass.
        
        # Gradient clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()  # Update model parameters.
        total_loss += loss.item()
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader, lossFunc):
    model.eval()
    total_loss = 0
    count = 0
    for data in loader:
        data = data.to(device)
        pred = model(data)
        loss = lossFunc(pred, data.y)
        count += 1
        if count % 32 == 0:
            print(pred[:5], data.y[:5])
        total_loss += loss.item()
    return total_loss / len(loader.dataset)

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

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Checkpoint loaded from epoch {epoch}')
        return epoch
    else:
        print(f'No checkpoint found at {checkpoint_path}')
        return 0

if __name__ == '__main__':
    files = [f'data/N216.{i}.lammpstrj' for i in range(1, 101)]
    data = read_data(files)
    print('Data read')
    
    graphs = make_graphs(data)
    print(len(graphs))
    print('Graphs made')

    np.random.shuffle(graphs)
    test_length = int(len(graphs) / 10)
    train_graphs, test_graphs = graphs[:-test_length], graphs[-test_length:]
    batch_size = 32
    train_loader = DataLoader(train_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)
    print('Data loaded')

    model = GNN(1, 6, 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    lossFunc = torch.nn.L1Loss(reduction='sum')

    # Load from checkpoint if available
    start_epoch = load_checkpoint(model, optimizer, 'checkpoints/checkpoint_epoch_24.pth')
    # Delete next line
    for g in optimizer.param_groups:
        g['lr'] = 0.001

    test_losses = []
    train_losses = []
    for epoch in range(start_epoch + 1, 50):
        loss = train(model, optimizer, train_loader, lossFunc, clip_value=1.0)
        test_loss = test(model, test_loader, lossFunc)
        test_losses.append(test_loss)
        train_losses.append(loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch:02d}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, LR: {current_lr*10**7:.2f}*10^(-7)')
        scheduler.step()

        if epoch % 6 == 0:
            save_checkpoint(model, optimizer, epoch)