// cg_update_xr.comp.glsl
// x  +=  alpha * p
// r  -=  alpha * Ap
// Writes partial r·r sums into reduce_buf, then the last workgroup
// finalizes cg_scalars[3] = beta and advances cg_scalars[0] = rr_new
// inline, eliminating the separate dr_beta dispatch.
//
// One thread per constraint c.
// Dispatch: ceil(m / 64) workgroups × 1 × 1
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 10)          buffer LVecBuf  { float l_data[];    }; // m  (x)
layout(std430, binding = 12)          buffer RVecBuf  { float r_data[];    }; // m
layout(std430, binding = 13) readonly buffer PVecBuf  { float p_data[];    }; // m
layout(std430, binding = 14) readonly buffer ApVecBuf { float Ap_data[];   }; // m
layout(std430, binding = 24)          buffer CgScalars { float cg_scalars[8]; uint reduce_ctr; };
layout(std430, binding = 25)          buffer ReduceBuf{ float reduce_buf[];};  // ceil(m/64)

uniform int u_m;
uniform int u_n_partials; // = ceil(m/64)

shared float s_rr[64];

void main() {
    uint c   = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;

    float r_val = 0.0;
    if (c < uint(u_m)) {
        float alpha = cg_scalars[2];
        r_val       = r_data[c] - alpha * Ap_data[c];
        l_data[c]  += alpha * p_data[c];
        r_data[c]   = r_val;
    }

    // Partial r·r reduction within this workgroup
    s_rr[lid] = r_val * r_val;
    barrier();
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid < stride) s_rr[lid] += s_rr[lid + stride];
        barrier();
    }
    if (lid == 0u) {
        reduce_buf[gl_WorkGroupID.x] = s_rr[0];
        memoryBarrier();
        uint prior = atomicAdd(reduce_ctr, 1u);
        if (prior == uint(u_n_partials) - 1u) {
            // Last workgroup: finalize beta = rr_new / rr_old
            float total  = 0.0;
            for (uint i = 0u; i < uint(u_n_partials); i++) total += reduce_buf[i];
            float rr_old = cg_scalars[0];
            float denom  = (abs(rr_old) < 1e-30) ? 1e-30 : rr_old;
            cg_scalars[3] = total / denom;
            cg_scalars[0] = total; // advance rr
            atomicExchange(reduce_ctr, 0u);
        }
    }
