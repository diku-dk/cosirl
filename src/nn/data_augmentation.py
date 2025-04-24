import torch            

class IdentityTransform:
    def __call__(self, mesh_data):
        vertices = mesh_data['vertices']
        vertices += 0.0 # Just as an example
        mesh_data['vertices'] = vertices
        return mesh_data
