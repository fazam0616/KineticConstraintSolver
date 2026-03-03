// build_A.comp.glsl
// Computes the system matrix A = dt^2 * J * diag(inv_mass) * J^T.
// Dispatch: ceil(m/8) x ceil(m/8) workgroups of 8x8 threads.
// Thread (i,j) computes A[i,j].
// Mirrors: build_sparse_A_from_constraints()  (Jc_scaled * Jct path in Simulator.c).
#version 430 core
layout(local_size_x = 8, local_size_y = 8) in;

layout(std430, binding = 2) readonly buffer InvMassSSBO { float inv_mass[]; }; // n floats (per node)
layout(std430, binding = 6) readonly buffer JColsBuf    { int   J_cols[];   }; // m_sparse*6
layout(std430, binding = 7) readonly buffer JValsBuf    { float J_vals[];   }; // m_sparse*6
layout(std430, binding = 8) writeonly buffer AMatBuf    { float A_data[];   }; // m_sparse*m_sparse

uniform int   u_m;  // m_sparse
uniform float u_dt; // full dt (not sub_dt), used for dt^2 scaling

void main() {
    int i = int(gl_GlobalInvocationID.x);
    int j = int(gl_GlobalInvocationID.y);
    if (i >= u_m || j >= u_m) return;

    // A[i,j] = dt^2 * sum_{k,l: J_cols[i*6+k]==J_cols[j*6+l]} J_vals[i*6+k] * inv_mass[col/3] * J_vals[j*6+l]
    float sum = 0.0;
    for (int ki = 0; ki < 6; ki++) {
        int col_i = J_cols[i*6 + ki];
        if (col_i < 0) continue;
        float vi  = J_vals[i*6 + ki];
        float im  = inv_mass[col_i / 3]; // shared per node across x,y,z DOFs

        for (int kj = 0; kj < 6; kj++) {
            if (J_cols[j*6 + kj] == col_i) {
                sum += vi * im * J_vals[j*6 + kj];
            }
        }
    }

    // also add tiny regularization on the diagonal to keep A PD
    float diag_reg = (i == j) ? 1e-10 : 0.0;
    A_data[i * u_m + j] = sum * u_dt * u_dt + diag_reg;
}
