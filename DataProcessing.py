import numpy as np
from torch_geometric.data import Data

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
                        'id': int(atom_data[0]),
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

def make_basic_graphs(data):
    graphs = []
    for frame in data:
        atoms = frame['atoms']
        x = np.array([[atom['x'], atom['y'], atom['z']] for atom in atoms])
        edge_index = np.array([i, j] for i in range(len(atoms)) for j in range(len(atoms)) if i != j)
        edge_attr = np.array([[atom['fx'], atom['fy'], atom['fz']] for atom in atoms])
        y = np.array([atom['type'] for atom in atoms])
        graphs.append(Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y))
    return graphs