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
        
        # Calculate the center of mass (centroid)
        center = mesh.centroid
        
        # Move all vertices so the center becomes (0,0,0)
        mesh.vertices -= center
        
        # --- SCALING FIX ---
        # Normalize size so it fits in our view (approx 5 units wide)
        max_span = np.max(mesh.extents)
        if max_span > 0:
            mesh.vertices /= max_span  # Make it size 1.0
            mesh.vertices *= 5.0       # Scale up to size 5.0

        print(f"Loaded {path}")
        print(f"Original Bounds: {mesh.bounds}")
        mesh.vertices -= mesh.centroid
        print(f"New Bounds (Should be centered around 0): {mesh.bounds}")
        
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