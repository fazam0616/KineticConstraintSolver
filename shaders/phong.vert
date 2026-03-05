#version 130
in vec3 a_position;
in vec3 a_normal;
uniform mat4 u_MVP;
out vec3 v_normal;
void main() {
    v_normal = normalize(a_normal);
    gl_Position = u_MVP * vec4(a_position, 1.0);
}
