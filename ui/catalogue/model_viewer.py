# ui/catalogue/model_viewer.py
import numpy as np
import trimesh
from PIL import Image
import ctypes
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

# --- SHADERS ---
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_nrm;
layout(location = 2) in vec2 a_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_normal;
out vec2 v_uv;

void main() {
    gl_Position = u_proj * u_view * u_model * vec4(a_pos, 1.0);
    v_normal = a_nrm;
    v_uv = a_uv;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform bool u_has_tex;

void main() {
    if (u_has_tex) {
        fragColor = texture(u_tex, v_uv);
    } else {
        // Fallback color (Golden)
        vec3 light = vec3(0.5, 0.5, 1.0);
        float diff = max(dot(normalize(v_normal), normalize(light)), 0.2);
        fragColor = vec4(vec3(0.8, 0.7, 0.2) * diff, 1.0);
    }
}
"""

class ModernMeshWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.mesh_ready = False
        self.texture_id = None
        self.has_texture = False
        self.index_count = 0
        
        # Matrices
        self.view_mat = np.eye(4, dtype=np.float32)
        self.proj_mat = np.eye(4, dtype=np.float32)
        self.model_mat = np.eye(4, dtype=np.float32)
        
        # Rotation state
        self.rot_x = 0
        self.rot_y = 0
        self.last_pos = None

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # Dark Grey Background to match App Theme
        glClearColor(0.12, 0.12, 0.12, 1.0)

        try:
            self.shader_program = compileProgram(
                compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            )
            self._init_uniforms()
        except Exception as e:
            print(f"Shader compilation failed: {e}")

    def _init_uniforms(self):
        self.u_model = glGetUniformLocation(self.shader_program, "u_model")
        self.u_view = glGetUniformLocation(self.shader_program, "u_view")
        self.u_proj = glGetUniformLocation(self.shader_program, "u_proj")
        self.u_tex = glGetUniformLocation(self.shader_program, "u_tex")
        self.u_has_tex = glGetUniformLocation(self.shader_program, "u_has_tex")

    def load_mesh(self, file_path):
        try:
            mesh = trimesh.load(file_path, force='mesh')
            
            # Auto-Center & Scale
            mesh.vertices -= mesh.centroid 
            max_dim = np.max(mesh.bounding_box.extents)
            if max_dim > 0:
                mesh.apply_scale(1.5 / max_dim) 
            
            self.makeCurrent() 
            self.texture_id, self.has_texture = self._load_texture_data(mesh)
            self._upload_geometry(mesh)
            self.doneCurrent()
            
            self.mesh_ready = True
            self.update()
            
        except Exception as e:
            print(f"Error loading mesh: {e}")

    def _load_texture_data(self, mesh):
        tex_img = None
        if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image'):
            tex_img = mesh.visual.material.image
        elif hasattr(mesh.visual, 'image'):
            tex_img = mesh.visual.image
            
        if tex_img:
            if not isinstance(tex_img, Image.Image):
                tex_img = Image.fromarray(tex_img)
            
            img_data = np.array(tex_img.convert("RGBA"))
            img_data = np.flipud(img_data) # Flip for GL
            
            tex_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img_data.shape[1], img_data.shape[0], 
                         0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
            return tex_id, True
            
        return None, False

    def _upload_geometry(self, mesh):
        verts = np.array(mesh.vertices, dtype=np.float32)
        
        if hasattr(mesh, 'vertex_normals'):
            norms = np.array(mesh.vertex_normals, dtype=np.float32)
        else:
            norms = np.zeros_like(verts)

        if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            uvs = np.array(mesh.visual.uv, dtype=np.float32)
        else:
            uvs = np.zeros((len(verts), 2), dtype=np.float32)

        data = np.hstack((verts, norms, uvs)).astype(np.float32)
        faces = np.array(mesh.faces, dtype=np.uint32).flatten()
        self.index_count = len(faces)

        if hasattr(self, 'vao'): glDeleteVertexArrays(1, [self.vao])
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)
        
        stride = data.strides[0]
        glEnableVertexAttribArray(0); glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1); glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(2); glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        
        glBindVertexArray(0)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        aspect = w / h if h > 0 else 1
        self.proj_mat = self._perspective(45, aspect, 0.1, 100.0)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if not self.mesh_ready: return
        
        glUseProgram(self.shader_program)
        
        self.model_mat = self._create_rotation_matrix(self.rot_x, self.rot_y)
        view = np.eye(4, dtype=np.float32)
        view[3, 2] = -3.0
        
        glUniformMatrix4fv(self.u_proj, 1, GL_FALSE, self.proj_mat)
        glUniformMatrix4fv(self.u_view, 1, GL_FALSE, view)
        glUniformMatrix4fv(self.u_model, 1, GL_FALSE, self.model_mat)

        if self.has_texture:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glUniform1i(self.u_tex, 0)
            glUniform1i(self.u_has_tex, 1)
        else:
            glUniform1i(self.u_has_tex, 0)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def _perspective(self, fov, aspect, near, far):
        f = 1.0 / np.tan(np.radians(fov) / 2)
        mat = np.zeros((4, 4), dtype=np.float32)
        mat[0, 0] = f / aspect; mat[1, 1] = f
        mat[2, 2] = (far + near) / (near - far)
        mat[2, 3] = -1; mat[3, 2] = (2 * far * near) / (near - far)
        return mat

    def _create_rotation_matrix(self, rx, ry):
        rx, ry = np.radians(rx), np.radians(ry)
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        
        mat_x = np.eye(4, dtype=np.float32)
        mat_x[1,1], mat_x[1,2] = cos_x, -sin_x
        mat_x[2,1], mat_x[2,2] = sin_x, cos_x
        
        mat_y = np.eye(4, dtype=np.float32)
        mat_y[0,0], mat_y[0,2] = cos_y, sin_y
        mat_y[2,0], mat_y[2,2] = -sin_y, cos_y
        
        return np.dot(mat_y, mat_x)

    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.last_pos:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            self.rot_x += dy
            self.rot_y += dx
            self.last_pos = event.pos()
            self.update()