// cg_zero_v.comp.glsl
// Zeroes jt_vec (stored as float-as-uint) before each J^T scatter pass.
// Must run before cg_jt_scatter each CG iteration.
//
// Dispatch: ceil(3*n / 64) workgroups × 1 × 1
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 23) buffer JtVecBuf { uint jt_vec[]; }; // 3*n uints (float-as-uint)

uniform int u_n3; // = 3 * n

void main() {
    uint k = gl_GlobalInvocationID.x;
    if (k < uint(u_n3)) jt_vec[k] = 0u;
}
