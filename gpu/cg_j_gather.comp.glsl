// cg_j_gather.comp.glsl
// Sparse J × (M^{-1} J^T p)  →  Ap_vec.
// Also accumulates partial sums of p · Ap into reduce_buf.
//
// Called after cg_jt_scatter + cg_minv_scale so jt_vec holds M^{-1} J^T p.
// One thread per constraint c.
//
// Dispatch: ceil(m / 64) workgroups × 1 × 1
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 6)  readonly buffer JColsBuf  { int   J_cols[];   }; // m*6
layout(std430, binding = 7)  readonly buffer JValsBuf  { float J_vals[];   }; // m*6
layout(std430, binding = 13) readonly buffer PVecBuf   { float p_data[];   }; // m
layout(std430, binding = 14)          buffer ApVecBuf  { float Ap_data[];  }; // m
layout(std430, binding = 23) readonly buffer JtVecBuf  { float jt_vec[];   }; // 3n
layout(std430, binding = 25)          buffer ReduceBuf { float reduce_buf[];};  // ceil(m/64)

uniform int   u_m;
uniform float u_dt; // = sub_dt; scales Ap by sub_dt^2 to match A = sub_dt^2 J M^{-1} J^T

shared float s_partial[64];

void main() {
    uint c   = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;

    float pAp_partial = 0.0;
    if (c < uint(u_m)) {
        float Ap_c = 0.0;
        for (int k = 0; k < 6; k++) {
            int col = J_cols[int(c) * 6 + k];
            if (col < 0) continue;
            Ap_c += J_vals[int(c) * 6 + k] * jt_vec[uint(col)];
        }
        // Scale by sub_dt^2 to match build_A:  A = sub_dt^2 * J M^{-1} J^T
        // Add diagonal regularization (1e-10) to keep A numerically PD
        Ap_c = Ap_c * u_dt * u_dt + 1e-10 * p_data[c];
        Ap_data[c]   = Ap_c;
        pAp_partial  = p_data[c] * Ap_c;
    }

    // Partial p·Ap reduction within this workgroup
    s_partial[lid] = pAp_partial;
    barrier();
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid < stride) s_partial[lid] += s_partial[lid + stride];
        barrier();
    }
    if (lid == 0u) reduce_buf[gl_WorkGroupID.x] = s_partial[0];
}
