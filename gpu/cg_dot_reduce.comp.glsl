// cg_dot_reduce.comp.glsl
// Final reduction of reduce_buf partial sums → one scalar.
// Runs as a single workgroup of 512 threads (handles up to 512 partial sums,
// i.e. m <= 32768 constraints).
//
// Three modes (uniform u_mode):
//   0 = INIT  : sum → cg_scalars[0]  (rr = r·r initial)
//   1 = ALPHA : sum → pAp  = cg_scalars[1];  alpha = cg_scalars[0] / pAp → cg_scalars[2]
//   2 = BETA  : sum → rr_new = cg_scalars[0];  beta = rr_new / rr_old → cg_scalars[3]
//
// Dispatch: (1, 1, 1)
#version 430 core
layout(local_size_x = 512) in;

layout(std430, binding = 24) buffer CgScalars { float cg_scalars[8]; };
// cg_scalars layout: [0]=rr [1]=pAp [2]=alpha [3]=beta [4..7]=spare
layout(std430, binding = 25) readonly buffer ReduceBuf { float reduce_buf[]; };

uniform int u_n_partials; // = ceil(m/64), number of valid entries in reduce_buf
uniform int u_mode;       // 0=INIT, 1=ALPHA, 2=BETA

shared float s_sum[512];

void main() {
    uint lid = gl_LocalInvocationID.x;

    // Load: pad with 0 for threads beyond n_partials
    s_sum[lid] = (lid < uint(u_n_partials)) ? reduce_buf[lid] : 0.0;
    barrier();

    // Tree-reduce within this workgroup
    for (uint stride = 256u; stride > 0u; stride >>= 1u) {
        if (lid < stride) s_sum[lid] += s_sum[lid + stride];
        barrier();
    }

    if (lid == 0u) {
        float total = s_sum[0];
        if (u_mode == 0) {
            // INIT: store initial rr
            cg_scalars[0] = total;
        } else if (u_mode == 1) {
            // ALPHA: total = p·Ap; alpha = rr / p·Ap
            cg_scalars[1] = total;
            float denom   = (abs(total) < 1e-30) ? 1e-30 : total;
            cg_scalars[2] = cg_scalars[0] / denom;
        } else {
            // BETA: total = rr_new; beta = rr_new / rr_old
            float rr_old  = cg_scalars[0];
            float denom   = (abs(rr_old) < 1e-30) ? 1e-30 : rr_old;
            cg_scalars[3] = total / denom;
            cg_scalars[0] = total; // advance rr
        }
    }
}
