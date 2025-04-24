import os
import torch
import meshio
from torch.utils.data import Dataset, DataLoader


def create_mesh_dataloader(mesh_dir, batch_size=1, shuffle=True, num_workers=1, transform=None):
    """
    Create a DataLoader for mesh data
    
    Args:
        mesh_dir (str): Directory containing .obj files
        batch_size (int): Batch size for DataLoader
        shuffle (bool): Whether to shuffle the dataset
        num_workers (int): Number of worker threads for loading data
        transform (callable, optional): Optional transform to be applied on meshes
        
    Returns:
        DataLoader: PyTorch DataLoader for mesh data
    """
    dataset = MeshPathDataset(mesh_dir=mesh_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        #collate_fn=mesh_collate_fn,
    )
    return dataloader

class MeshPathDataset(Dataset):
    def __init__(self, mesh_dir):
        """
        Dataset for loading .obj mesh files as paths
        
        Args:
            mesh_dir (str): Directory containing .obj files
        """
        self.mesh_dir = mesh_dir
        
        self.mesh_paths = []
        for filename in os.listdir(mesh_dir):
            if filename.endswith('.obj'):
                self.mesh_paths.append(os.path.join(mesh_dir, filename))
                
        self.mesh_paths.sort()

    def __len__(self):
        return len(self.mesh_paths)
    
    def __getitem__(self, idx):
        mesh_path = self.mesh_paths[idx]
        return mesh_path


class MeshDataset(Dataset):
    def __init__(self, mesh_dir, transform=None):
        """
        Dataset for loading .obj mesh files
        
        Args:
            mesh_dir (str): Directory containing .obj files
            transform (callable, optional): Optional transform to be applied on a mesh
        """
        self.mesh_dir = mesh_dir
        self.transform = transform
        
        # Hans: Extend to multiple meshes types. Including tetrahedral meshes
        self.mesh_paths = []
        for filename in os.listdir(mesh_dir):
            if filename.endswith('.obj'):
                self.mesh_paths.append(os.path.join(mesh_dir, filename))
                
        self.mesh_paths.sort()

    def is_surface_mesh(self):
        if len(self.mesh_paths) == 0:
            raise ValueError("There are no meshes to check!")
        sample_mesh_data = self[0]
        return sample_mesh_data["elements"].shape[1] == 3
        
    def is_volume_mesh(self):
        if len(self.mesh_paths) == 0:
            raise ValueError("There are no meshes to check!")
        sample_mesh_data = self[0]
        return sample_mesh_data["elements"].shape[1] == 4

    def __len__(self):
        return len(self.mesh_paths)
    
    def __getitem__(self, idx):
        mesh_path = self.mesh_paths[idx]
        mesh = meshio.read(mesh_path)
        vertices = torch.from_numpy(mesh.points).float()
        
        elements = None
        if hasattr(mesh, 'cells') and len(mesh.cells) > 0:
            for cell_block in mesh.cells:
                if cell_block.type == "triangle":
                    elements = torch.from_numpy(cell_block.data).long()
                    break
        
        # Create a dictionary with mesh data
        mesh_data = {
            'vertices': vertices,
            'elements': elements,
            'path': mesh_path,
            'name': os.path.basename(mesh_path)
        }
        
        # Add any point data if available
        if hasattr(mesh, 'point_data') and mesh.point_data:
            for key, value in mesh.point_data.items():
                mesh_data[f'point_data_{key}'] = torch.from_numpy(value)
        
        # Apply transforms if any
        if self.transform:
            mesh_data = self.transform(mesh_data)
            
        return mesh_data


def mesh_collate_fn(batch):
    """
    Custom collate function for mesh batches
    
    Args:
        batch (list): List of mesh dictionaries
        
    Returns:
        dict: Batched mesh data
    """
    # Initialize the batched data dictionary
    batched_data = {}
    
    # Get all keys from the first element
    keys = batch[0].keys()
    
    for key in keys:
        if key in ['path', 'name']:
            # For string attributes, just collect them in a list
            batched_data[key] = [item[key] for item in batch]
        elif key == 'elements':
            # Faces need special handling because when batched,
            # face indices need to be offset based on the number of vertices
            batched_faces = []
            vertex_offset = 0
            
            for item in batch:
                if item[key] is not None:
                    faces = item[key].clone()
                    faces += vertex_offset
                    batched_faces.append(faces)
                    # Update vertex offset for the next mesh
                    vertex_offset += item['vertices'].shape[0]
            
            if batched_faces:
                batched_data[key] = torch.cat(batched_faces, dim=0)
            else:
                batched_data[key] = None
        else:
            # For tensor attributes, concatenate along the first dimension
            tensors = [item[key] for item in batch if item[key] is not None]
            if tensors:
                batched_data[key] = torch.cat(tensors, dim=0)
            else:
                batched_data[key] = None
    
    return batched_data


