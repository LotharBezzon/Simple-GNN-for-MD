import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
import torch
import torch.nn.functional as Func

def read_data(file):
    with open(file, 'r') as f:
        lines = iter(f.readlines())
        data = []
        for line in lines:
            if line.startswith('ITEM: TIMESTEP'):
                timestep = int(next(lines))
                next(lines)  # Skip ITEM: NUMBER OF ATOMS
                atom_number = int(next(lines))
                next(lines)  # Skip ITEM: BOX BOUNDS
                next(lines)
                next(lines)
                next(lines)
                next(lines)  # Skip ITEM: ATOMS id mol type x y z fx fy fz
                atoms = []
                for _ in range(atom_number):
                    atom_data = next(lines).split()
                    atom = {
                        'id': int(atom_data[0]) - 1, # 0-based indexing
                        'mol': int(atom_data[1]),
                        'type': int(atom_data[2]),
                        'x': float(atom_data[3]),
                        'y': float(atom_data[4]),
                        'z': float(atom_data[5]),
                        'fx': float(atom_data[6]),
                        'fy': float(atom_data[7]),
                        'fz': float(atom_data[8])
                    }
                    atoms.append(atom)
                data.append({'timestep': timestep, 'atoms': atoms})
        return data

# The most essential model I could think about
def make_basic_graphs(data, k=5):
    graphs = []
    for frame in data:
        atoms = frame['atoms']
        # Position and type should be enough to understand Coulomb and LJ interactions
        x = torch.tensor([[atom['x'], atom['y'], atom['z'], 
                           atom['type']-1] for atom in atoms], dtype=torch.float)
        
        # Create k-nearest neighbors graph
        edge_index = knn_graph(x[:, :3], k=k, batch=None, loop=False)
        
        # This one-hot vectors should distinguish the three different types of interactions
        atom_mols = torch.tensor([atom['mol'] for atom in atoms])
        atom_types = torch.tensor([atom['type'] for atom in atoms])
        
        edge_attr = torch.zeros((edge_index.size(1), 3), dtype=torch.float)
        mol_diff = atom_mols[edge_index[0]] != atom_mols[edge_index[1]]
        type_diff = atom_types[edge_index[0]] != atom_types[edge_index[1]]
        
        edge_attr[mol_diff, 0] = 1  # LJ + Coul for distant atoms
        edge_attr[~mol_diff & type_diff, 1] = 1  # OH bonds
        edge_attr[~mol_diff & ~type_diff, 2] = 1  # HOH angle rigidity

        y = torch.tensor([(atom['fx'], atom['fy'], atom['fz']) for atom in atoms], dtype=torch.float)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return graphs

# Build graphs for a SchNet architecture
def make_SchNet_graphs(data):
    pass

# Build graphs with encoded node features
def make_encoded_graphs(data):
    pass