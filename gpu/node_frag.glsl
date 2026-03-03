#version 430

flat in uint v_flags;
in   vec3  v_normal_ws;

out vec4 frag_color;

const uint ANCHORED_BIT = 1u;
// normalize(1,2,1)
const vec3 LIGHT_DIR    = vec3(0.40825, 0.81650, 0.40825);

void main() {
    vec3  n     = normalize(v_normal_ws);
    // Two-sided diffuse so back-faces aren't black
    float d     = max(dot(n, LIGHT_DIR), max(dot(-n, LIGHT_DIR), 0.0));
    bool  anc   = (v_flags & ANCHORED_BIT) != 0u;
    vec3  base  = anc ? vec3(0.15, 0.15, 0.15) : vec3(0.90, 0.18, 0.12);
    frag_color  = vec4(base * (0.25 + 0.75 * d), 1.0);
}
