// cg_minv_scale.comp.glsl
// Scales jt_vec in-place by M^{-1}: jt_vec[3i+d] *= inv_mass[i].
// Converts float-as-uint → float, multiplies, writes back as uint.
// Safe: one thread per node, each thread owns its 3 DOFs exclusively.
//
// Dispatch: ceil(n / 64) workgroups × 1 × 1
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 2)  readonly buffer InvMassSSBO { float inv_mass[]; };
layout(std430, binding = 23)          buffer JtVecBuf    { uint  jt_vec[];   }; // 3n

uniform int u_n;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_n)) return;

    float im = inv_mass[i];
    uint base = i * 3u;
    for (uint d = 0u; d < 3u; d++) {
        float v = uintBitsToFloat(jt_vec[base + d]);
        jt_vec[base + d] = floatBitsToUint(v * im);
    }
}
