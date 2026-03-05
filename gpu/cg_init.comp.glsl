// cg_init.comp.glsl
// Sparse CG initialisation: x = 0, r = b, p = b.
// Writes partial r·r sums into reduce_buf for the first dot-reduce pass.
//
// Dispatch: ceil(m / 64) workgroups × 1 × 1
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 9)  readonly buffer BVecBuf  { float b_data[];    }; // m
layout(std430, binding = 10)          buffer LVecBuf  { float l_data[];    }; // m  (x)
layout(std430, binding = 12)          buffer RVecBuf  { float r_data[];    }; // m
layout(std430, binding = 13)          buffer PVecBuf  { float p_data[];    }; // m
layout(std430, binding = 25)          buffer ReduceBuf{ float reduce_buf[];};  // ceil(m/64)

uniform int u_m;

shared float s_rr[64];

void main() {
    uint c   = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;

    float rc = 0.0;
    if (c < uint(u_m)) {
        float bc = b_data[c];
        l_data[c] = 0.0;
        r_data[c] = bc;
        p_data[c] = bc;
        rc = bc;
    }

    // Partial r·r reduction within this workgroup
    s_rr[lid] = rc * rc;
    barrier();
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (lid < stride) s_rr[lid] += s_rr[lid + stride];
        barrier();
    }
    if (lid == 0u) reduce_buf[gl_WorkGroupID.x] = s_rr[0];
}
