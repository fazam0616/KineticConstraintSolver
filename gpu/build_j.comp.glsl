// build_j.comp.glsl
// One thread per non-spring constraint (0..m_sparse-1).
// Fills J_cols[c*6+k] and J_vals[c*6+k] with the at-most-6 nonzeros
// of Jacobian row c.  Unused slots are filled with col=-1, val=0.
//
// Constraint SSBO layout (8 ints per entry, non-spring constraints only):
//   [0] ctype  [1] a_idx  [2] b_idx
//   [3] floatBitsToInt(rest)
//   [4..6] floatBitsToInt(anchor xyz)
//   [7] floatBitsToInt(stiffness)
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer PosSSBO   { vec4  pos4[];       }; // n nodes
layout(std430, binding = 4) readonly buffer FlagsSSBO { uint  node_flags[]; }; // n nodes
layout(std430, binding = 5) readonly buffer ConBuf    { int   con_data[];   }; // m_sparse*8

layout(std430, binding = 6) writeonly buffer JColsBuf { int   J_cols[];     }; // m_sparse*6
layout(std430, binding = 7) writeonly buffer JValsBuf { float J_vals[];     }; // m_sparse*6

uniform int u_m; // number of non-spring constraints

#define CT_DIST   1
#define CT_ANCHOR 3
#define ANCHORED_BIT 1u

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

    // clear all 6 slots
    for (int k = 0; k < 6; k++) {
        J_cols[c*6 + k] = -1;
        J_vals[c*6 + k] = 0.0;
    }

    if (ctype == CT_DIST && a_idx >= 0 && b_idx >= 0) {
        vec3 pa = pos4[a_idx].xyz;
        vec3 pb = pos4[b_idx].xyz;
        vec3 d  = pb - pa;
        float len = length(d);
        vec3  n = (len > 1e-9) ? d / len : vec3(0.0, 1.0, 0.0);

        bool b_anchored = (node_flags[b_idx] & ANCHORED_BIT) != 0u;

        // node a: +n  (same sign as dist_dc_triplet: +nx for base node)
        J_cols[c*6 + 0] = 3*a_idx + 0;  J_vals[c*6 + 0] = n.x;
        J_cols[c*6 + 1] = 3*a_idx + 1;  J_vals[c*6 + 1] = n.y;
        J_cols[c*6 + 2] = 3*a_idx + 2;  J_vals[c*6 + 2] = n.z;

        // node b: -n  (only if not anchored)
        if (!b_anchored) {
            J_cols[c*6 + 3] = 3*b_idx + 0;  J_vals[c*6 + 3] = -n.x;
            J_cols[c*6 + 4] = 3*b_idx + 1;  J_vals[c*6 + 4] = -n.y;
            J_cols[c*6 + 5] = 3*b_idx + 2;  J_vals[c*6 + 5] = -n.z;
        }

    } else if (ctype == CT_ANCHOR && a_idx >= 0) {
        vec3 pa     = pos4[a_idx].xyz;
        vec3 anchor = vec3(ax, ay, az);
        vec3 d      = anchor - pa;
        float len   = length(d);
        vec3  n     = (len > 1e-9) ? d / len : vec3(0.0, 1.0, 0.0);

        // node a only: +n  (same sign as anchor_dc_triplet)
        J_cols[c*6 + 0] = 3*a_idx + 0;  J_vals[c*6 + 0] = n.x;
        J_cols[c*6 + 1] = 3*a_idx + 1;  J_vals[c*6 + 1] = n.y;
        J_cols[c*6 + 2] = 3*a_idx + 2;  J_vals[c*6 + 2] = n.z;
    }
}
