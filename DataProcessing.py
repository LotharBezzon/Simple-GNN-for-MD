import numpy as np
from torch_geometric.data import Data
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
def make_basic_graphs(data):
    graphs = []
    for frame in data:
        atoms = frame['atoms']
        # Position and type should be enough to understand Coulomb and LJ interactions
        x = np.array([[atom['x'], atom['y'], atom['z'], 
                       atom['type']-1] for atom in atoms])
        # For the moment, I keep the graph fully connected
        edge_index = torch.tensor([[i, j] for i in range(len(atoms)) for j in range(len(atoms)) if i != j], dtype=torch.long)
        # This one-hot vectors should distinguish the three different types of interactions
        edge_attr = torch.tensor([[1,0,0] if atoms[edge[0]]['mol'] != atoms[edge[1]]['mol']            # for LJ + Coul for distant atoms
                              else ([0,1,0] if atoms[edge[0]]['type'] != atoms[edge[1]]['type']    # for OH bonds
                                    else [0,0,1]) for edge in edge_index])                         # for HOH angle rigidity
        y = np.array([(atom['fx'], atom['fy'], atom['fz']) for atom in atoms])
        graphs.append(Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y))
    return graphs

# Build graphs for a SchNet architecture
def make_SchNet_graphs(data):
    pass

# Build graphs with encoded node features
def make_encoded_graphs(data):
    pass