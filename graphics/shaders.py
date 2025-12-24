# graphics/shaders.py

MESH_VS = """
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
    v_normal = mat3(transpose(inverse(u_model))) * a_nrm;
    v_uv = a_uv;
}
"""

MESH_FS = """
#version 330 core
in vec3 v_normal;
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
uniform bool u_has_tex;
uniform vec3 u_light_pos;
uniform float u_ambient_str;
uniform float u_diffuse_str;
uniform vec4 u_color_override; 

void main() {
    vec3 light_dir = normalize(u_light_pos);
    vec3 norm = normalize(v_normal);
    float diff = max(dot(norm, light_dir), 0.0) * u_diffuse_str;
    vec4 base_color = u_color_override; 
    if (u_has_tex) { base_color = texture(u_tex, v_uv); }
    vec3 ambient = base_color.rgb * u_ambient_str;
    vec3 diffuse = base_color.rgb * diff;
    fragColor = vec4(ambient + diffuse, base_color.a);
}
"""

# Background (Camera Feed) Shaders
BG_VS = """#version 330 core
layout(location = 0) in vec2 a_pos; layout(location = 1) in vec2 a_uv;
out vec2 v_uv; void main() { v_uv = a_uv; gl_Position = vec4(a_pos, 0.99, 1.0); }"""

BG_FS = """#version 330 core
in vec2 v_uv; out vec4 color; uniform sampler2D u_tex;
void main() { color = texture(u_tex, v_uv); }"""

DEBUG_VS = """
#version 330 core
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_color; 
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_model; // Added Model Matrix so axes can move with object!
out vec3 v_color;
void main() {
    gl_Position = u_proj * u_view * u_model * vec4(a_pos, 1.0);
    v_color = a_color;
}
"""

DEBUG_FS = """
#version 330 core
in vec3 v_color;
out vec4 color;
void main() {
    color = vec4(v_color, 1.0);
}
"""