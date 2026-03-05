// cg_update_xr.comp.glsl
// x  +=  alpha * p
// r  -=  alpha * Ap
// Writes partial r·r sums into reduce_buf for the BETA dot-reduce pass.
//
// One thread per constraint c.
// Dispatch: ceil(m / 64) workgroups × 1 × 1
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 10)          buffer LVecBuf  { float l_data[];    }; // m  (x)
layout(std430, binding = 12)          buffer RVecBuf  { float r_data[];    }; // m
layout(std430, binding = 13) readonly buffer PVecBuf  { float p_data[];    }; // m
layout(std430, binding = 14) readonly buffer ApVecBuf { float Ap_data[];   }; // m
layout(std430, binding = 24) readonly buffer CgScalars{ float cg_scalars[];};
layout(std430, binding = 25)          buffer ReduceBuf{ float reduce_buf[];};  // ceil(m/64)

uniform int u_m;

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
    if (lid == 0u) reduce_buf[gl_WorkGroupID.x] = s_rr[0];
}
