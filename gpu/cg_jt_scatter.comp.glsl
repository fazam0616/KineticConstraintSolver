// cg_jt_scatter.comp.glsl
// Sparse J^T × p  →  jt_vec  using CAS-based atomic float add.
//
// One thread per constraint c.
// Reads p_vec (binding 13) as the source vector.
// For the post-solve scatter of J^T λ, the caller temporarily rebinds
// l_vec (binding 10) into slot 13 before dispatching this shader.
//
// Dispatch: ceil(m / 64) workgroups × 1 × 1
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 6)  readonly buffer JColsBuf { int   J_cols[]; }; // m*6
layout(std430, binding = 7)  readonly buffer JValsBuf { float J_vals[]; }; // m*6
layout(std430, binding = 13) readonly buffer PVecBuf  { float p_data[]; }; // source vec
layout(std430, binding = 23)          buffer JtVecBuf { uint  jt_vec[]; }; // 3n (float-as-uint)

uniform int u_m;

// CAS-based float atomic add (core OpenGL 4.3 — no NV extension needed)
void atomicAddF(uint idx, float val) {
    uint assumed, next;
    uint old = jt_vec[idx];
    do {
        assumed = old;
        next    = floatBitsToUint(uintBitsToFloat(assumed) + val);
        old     = atomicCompSwap(jt_vec[idx], assumed, next);
    } while (old != assumed);
}

void main() {
    uint c = gl_GlobalInvocationID.x;
    if (c >= uint(u_m)) return;

    float pc = p_data[c];
    for (int k = 0; k < 6; k++) {
        int col = J_cols[int(c) * 6 + k];
        if (col < 0) continue;
        atomicAddF(uint(col), J_vals[int(c) * 6 + k] * pc);
    }
}
