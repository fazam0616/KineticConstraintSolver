#version 430

in vec3 v_normal_ws;
out vec4 frag_color;

const vec3 LIGHT_DIR = vec3(0.40825, 0.81650, 0.40825); // normalize(1,2,1)

void main() {
    vec3 n = normalize(v_normal_ws);
    // Two-sided diffuse
    float d = max(dot(n, LIGHT_DIR), max(dot(-n, LIGHT_DIR), 0.0));
    frag_color = vec4(vec3(0.22 + 0.60 * d), 1.0);
}
