// cg_update_p.comp.glsl
// p  =  r  +  beta * p
//
// One thread per constraint c.
// Dispatch: ceil(m / 64) workgroups × 1 × 1
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 12) readonly buffer RVecBuf  { float r_data[];    }; // m
layout(std430, binding = 13)          buffer PVecBuf  { float p_data[];    }; // m
layout(std430, binding = 24) readonly buffer CgScalars{ float cg_scalars[];};

uniform int u_m;

void main() {
    uint c = gl_GlobalInvocationID.x;
    if (c >= uint(u_m)) return;
    float beta = cg_scalars[3];
    p_data[c]  = r_data[c] + beta * p_data[c];
}
