// cg_jt_gather.comp.glsl
// CSR-based J^T gather: one thread per DOF, NO atomics.
// Replaces the three-dispatch chain: cg_zero_v + cg_jt_scatter + cg_minv_scale.
//
// Modes (set by CPU per dispatch):
//   u_src=0, u_apply_minv=1  → jt_vec[d] = M^{-1} J^T p   (CG iterations)
//   u_src=1, u_apply_minv=0  → jt_vec[d] = J^T λ           (post-solve, for apply_corr)
//
// CSR layout: for DOF d, the contributing (c,k) pairs are
//   csr_data[ csr_offsets[d] .. csr_offsets[d+1]-1 ]
//   each entry is two consecutive ints: { constraint_idx, k_slot }
//
// Dispatch: ceil(3*n / 64) workgroups × 1 × 1
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 2)  readonly buffer InvMassSSBO { float inv_mass[];   }; // n
layout(std430, binding = 7)  readonly buffer JValsBuf    { float J_vals[];     }; // m*6
layout(std430, binding = 10) readonly buffer LVecBuf     { float l_data[];     }; // m  (λ)
layout(std430, binding = 13) readonly buffer PVecBuf     { float p_data[];     }; // m  (p)
layout(std430, binding = 23)          buffer JtVecBuf    { float jt_vec[];     }; // 3n
layout(std430, binding = 26) readonly buffer CsrOffBuf   { int   csr_offsets[];};  // 3n+1
layout(std430, binding = 27) readonly buffer CsrDatBuf   { int   csr_data[];   };  // n_entries*2

uniform int u_n3;         // = 3 * n
uniform int u_src;        // 0 = read p_data (binding 13), 1 = read l_data (binding 10)
uniform int u_apply_minv; // 1 = multiply result by inv_mass[d/3], 0 = skip

void main() {
    uint d = gl_GlobalInvocationID.x;
    if (d >= uint(u_n3)) return;

    int start = csr_offsets[int(d)];
    int end   = csr_offsets[int(d) + 1];

    float sum = 0.0;
    for (int e = start; e < end; e++) {
        int c = csr_data[e * 2 + 0];
        int k = csr_data[e * 2 + 1];
        float src_val = (u_src == 0) ? p_data[c] : l_data[c];
        sum += J_vals[c * 6 + k] * src_val;
    }

    if (u_apply_minv != 0) {
        sum *= inv_mass[d / 3u];
    }

    jt_vec[d] = sum;
}
