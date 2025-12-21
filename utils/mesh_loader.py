# utils/mesh_loader.py
import trimesh
import numpy as np
from PIL import Image

class MeshData:
    """Simple container for raw mesh data"""
    def __init__(self, vertices, norms, uvs, faces, texture_data=None):
        self.vertices = vertices
        self.norms = norms
        self.uvs = uvs
        self.faces = faces
        self.texture_data = texture_data # RGBA bytes

def load_mesh_data(path, is_occluder=False):
    """Parses 3D file and returns MeshData object."""
    try:
        mesh = trimesh.load(path, force='mesh')
        
        # 1. Centering and Scaling
        mesh.vertices -= mesh.centroid 
        max_span = np.max(mesh.extents)
        if max_span > 0: 
            mesh.vertices /= max_span
            mesh.vertices *= 5.0 # Standard size normalization

        # 2. Extract Data
        verts = np.array(mesh.vertices, dtype=np.float32)
        
        if hasattr(mesh, 'vertex_normals'):
            norms = np.array(mesh.vertex_normals, dtype=np.float32)
        else:
            norms = np.zeros_like(verts)

        if hasattr(mesh.visual, 'uv') and not is_occluder:
             uvs = np.array(mesh.visual.uv, dtype=np.float32)
        else:
             uvs = np.zeros((len(verts), 2), dtype=np.float32)
             
        faces = np.array(mesh.faces, dtype=np.uint32).flatten()

        # 3. Process Texture (If any)
        texture_bytes = None
        if not is_occluder:
            tex = getattr(mesh.visual.material, 'image', None) or getattr(mesh.visual, 'image', None)
            if tex:
                if not isinstance(tex, Image.Image): 
                    tex = Image.fromarray(tex)
                # Convert to RGBA and Flip Upside Down for OpenGL
                texture_bytes = np.flipud(np.array(tex.convert("RGBA")))

        return MeshData(verts, norms, uvs, faces, texture_bytes)

    except Exception as e:
        print(f"Error loading mesh {path}: {e}")
        return None