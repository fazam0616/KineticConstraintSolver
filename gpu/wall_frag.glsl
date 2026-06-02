#version 430

in vec3 v_normal_ws;
flat in int v_translucent;
out vec4 frag_color;

const vec3 LIGHT_DIR = vec3(0.40825, 0.81650, 0.40825); // normalize(1,2,1)

void main() {
    vec3 n = normalize(v_normal_ws);
    // Two-sided diffuse
    float d = max(dot(n, LIGHT_DIR), max(dot(-n, LIGHT_DIR), 0.0));
    if (v_translucent != 0) {
        // Light-blue semi-transparent (piston lid)
        vec3 blue = vec3(0.3 + 0.4 * d, 0.55 + 0.25 * d, 0.9);
        frag_color = vec4(blue, 0.35);
    } else {
        frag_color = vec4(vec3(0.22 + 0.60 * d), 1.0);
    }
}
