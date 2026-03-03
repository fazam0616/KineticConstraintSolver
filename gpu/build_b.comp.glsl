// build_b.comp.glsl
// One thread per non-spring constraint.  Writes b[c] = -C(q) (constraint error negated).
// Mirrors dist_err() and anchor_err() from Constraint.c.
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer PosSSBO { vec4 pos4[]; };
layout(std430, binding = 5) readonly buffer ConBuf  { int  con_data[]; }; // m_sparse*8
layout(std430, binding = 9) writeonly buffer BVec   { float b[]; };

uniform int u_m;

#define CT_DIST   1
#define CT_ANCHOR 3

void main() {
    int c = int(gl_GlobalInvocationID.x);
    if (c >= u_m) return;

    int   ctype = con_data[c*8 + 0];
    int   a_idx = con_data[c*8 + 1];
    int   b_idx = con_data[c*8 + 2];
    float rest  = intBitsToFloat(con_data[c*8 + 3]);
    float ax    = intBitsToFloat(con_data[c*8 + 4]);
    float ay    = intBitsToFloat(con_data[c*8 + 5]);
    float az    = intBitsToFloat(con_data[c*8 + 6]);

    float err = 0.0;

    if (ctype == CT_DIST && a_idx >= 0 && b_idx >= 0) {
        // dist_err: |a - b| - rest_length
        vec3 d = pos4[a_idx].xyz - pos4[b_idx].xyz;
        err = length(d) - rest;

    } else if (ctype == CT_ANCHOR && a_idx >= 0) {
        // anchor_err: dist(node, anchor)
        vec3 d = pos4[a_idx].xyz - vec3(ax, ay, az);
        err = length(d);
    }

    b[c] = -err; // RHS = -C(q)
}
