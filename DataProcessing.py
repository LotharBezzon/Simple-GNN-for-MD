import numpy as np
from torch_geometric.data import Data
import torch
import torch.nn.functional as Func

def read_data(files):
    data = []
    for file in files:
        with open(file, 'r') as f:
            lines = iter(f.readlines())
            file_data = []
            for line in lines:
                if line.startswith('ITEM: TIMESTEP'):
                    timestep = int(next(lines))
                    next(lines)  # Skip ITEM: NUMBER OF ATOMS
                    atom_number = int(next(lines))
                    next(lines)  # Skip ITEM: BOX BOUNDS
                    bound_x = next(lines).split()
                    size_x = - float(bound_x[0]) + float(bound_x[1])
                    bound_y = next(lines).split()
                    size_y = - float(bound_y[0]) + float(bound_y[1])
                    bound_z = next(lines).split()
                    size_z = - float(bound_z[0]) + float(bound_z[1])
                    box_size = torch.tensor([size_x, size_y, size_z], dtype=torch.float)
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
                    file_data.append({'timestep': timestep, 'num_atoms': atom_number, 'box_size': box_size, 'atoms': atoms})
        data += file_data
    return data

# To account for periodic boundary conditions
def minimum_image_distance(coords1, coords2, box_size):

  delta = coords1 - coords2
  delta -= torch.round(delta / box_size) * box_size
  distance = torch.norm(delta, dim=-1)
  return distance

# Build graphs for a SchNet architecture
def make_SchNetlike_graphs(data):
    graphs = []
    for frame in data:
        atoms = frame['atoms']
        x = torch.tensor([[(atom['type'] - 1.5)*2] for atom in atoms], dtype=torch.float)

        # Create a fully connected graph
        edge_index = torch.combinations(torch.arange(frame['num_atoms']), r=2).t()
        
        # Duplicate edges in the opposite direction to make it undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        positions = torch.tensor([[atom['x'], atom['y'], atom['z']] for atom in atoms], dtype=torch.float)
        start_nodes, end_nodes = edge_index
        distances = minimum_image_distance(positions[start_nodes], positions[end_nodes], frame['box_size'])
        
        # Filter out edges with distances greater than 2.3
        mask = distances <= 2.3
        edge_index = edge_index[:, mask]
        distances = distances[mask]

        # Edge features are smooth 'one-hot vectors' to represent distances  
        #dist_one_hot = [0.5, 1.02, 1.69, 1.97, 3, 4]    # 0.96 for O-H, 1.69 for H-H, 1.97 for H bonds
        #edge_attr = torch.zeros((edge_index.size(1), len(dist_one_hot)), dtype=torch.float)
        #for i, dist in enumerate(dist_one_hot):
        #    edge_attr[:, i] = np.e ** (-(distances - dist)**2 / 0.15)    # 0.4 is quite arbitrary
        #edge_attr = distances

        atom_mols = torch.tensor([atom['mol'] for atom in atoms])
        atom_types = torch.tensor([atom['type'] for atom in atoms])
        
        edge_attr = torch.zeros((edge_index.size(1), 4), dtype=torch.float)
        mol_diff = atom_mols[edge_index[0]] != atom_mols[edge_index[1]]
        type_diff = atom_types[edge_index[0]] != atom_types[edge_index[1]]
        
        edge_attr[mol_diff, 0] = 1  # LJ + Coul for distant atoms
        edge_attr[~mol_diff & type_diff, 1] = 1  # OH bonds
        edge_attr[~mol_diff & ~type_diff, 2] = 1  # HOH angle rigidity
        edge_attr[:, 3] = distances

        y = torch.tensor([[atom['fx'], atom['fy'], atom['fz']] for atom in atoms], dtype=torch.float)
        y_mean = y.mean(dim=0, keepdim=True)
        y_std = y.std(dim=0, keepdim=True)
        y = (y - y_mean) / y_std
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return graphs

# Build graphs with encoded node features
def make_encoded_graphs(data):
    pass