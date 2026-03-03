// cg_solve.comp.glsl
// Full Conjugate-Gradient solve for A * l = b in a single workgroup.
// Supports up to 512 constraints (m_sparse <= 512).
// Mirrors cg_solve_sparse() in Simulator.c but in single-precision on device.
//
// Launch: 1 workgroup with local_size_x = 512 threads.
// Unused threads (i >= m) participate in barriers but contribute 0.
#version 430 core
layout(local_size_x = 512) in;

layout(std430, binding = 8)  readonly  buffer AMatBuf { float A_data[]; }; // m*m
layout(std430, binding = 9)  readonly  buffer BVecBuf { float b_data[]; }; // m
layout(std430, binding = 10) writeonly buffer LVecBuf { float l_data[]; }; // m  (output)
// temporaries (written/read by this invocation only, so no hazards across frames)
layout(std430, binding = 12) buffer RVecBuf  { float r_data[]; };
layout(std430, binding = 13) buffer PVecBuf  { float p_data[]; };
layout(std430, binding = 14) buffer ApVecBuf { float Ap_data[]; };

uniform int   u_m;        // number of constraints (<= 512)
uniform int   u_max_iter; // CG iteration cap (= solver_iters)
uniform float u_tol;      // convergence criterion (on |r|^2)

shared float s_p[512];
shared float s_reduce[512];

// Tree-reduce sum of val across all active threads.
// Returns total sum (same value for all threads).
// Ends with a groupMemoryBarrier+barrier so shared state is consistent after.
float tree_reduce_sum(uint lid, float val) {
    s_reduce[lid] = val;
    barrier();
    for (uint stride = 256u; stride > 0u; stride >>= 1u) {
        if (lid < stride) s_reduce[lid] += s_reduce[lid + stride];
        barrier();
    }
    return s_reduce[0]; // same for all threads after loop
}

void main() {
    uint i = gl_LocalInvocationIndex;

    // --- Initialise: x=0, r=b, p=b ---
    float xi = 0.0;
    float ri = (i < uint(u_m)) ? b_data[i] : 0.0;
    float pi = ri;

    s_p[i] = pi;   // load p into shared for fast Ap multiply
    barrier();

    // rr0 = r · r
    float rr = tree_reduce_sum(i, (i < uint(u_m)) ? ri * ri : 0.0);

    if (rr < u_tol) {
        if (i < uint(u_m)) l_data[i] = 0.0;
        return;
    }

    // --- CG iterations ---
    for (int iter = 0; iter < u_max_iter; iter++) {

        // Ap[i] = sum_j A[i,j] * s_p[j]   (thread i computes its own row dot product)
        float Ap_i = 0.0;
        if (i < uint(u_m)) {
            int base = int(i) * u_m;
            for (int j = 0; j < u_m; j++) {
                Ap_i += A_data[base + j] * s_p[j];
            }
        }

        // pAp = p · Ap  (same value for all threads after tree reduce)
        float pAp = tree_reduce_sum(i, (i < uint(u_m)) ? pi * Ap_i : 0.0);

        if (abs(pAp) < 1e-16) break; // all threads see same pAp → safe break

        float alpha = rr / pAp;

        if (i < uint(u_m)) {
            xi += alpha * pi;
            ri  -= alpha * Ap_i;
        }

        // rr_new = r · r
        float rr_new = tree_reduce_sum(i, (i < uint(u_m)) ? ri * ri : 0.0);

        bool converged = (rr_new < u_tol); // same for all threads after broadcast

        float beta = rr_new / max(rr, 1e-16);
        rr = rr_new;

        // p = r + beta * p
        pi = (i < uint(u_m)) ? (ri + beta * pi) : 0.0;
        s_p[i] = pi;
        barrier(); // make updated s_p visible for next iteration

        if (converged) break; // all threads set converged identically → safe break
    }

    // --- Write solution ---
    if (i < uint(u_m)) l_data[i] = xi;
}
