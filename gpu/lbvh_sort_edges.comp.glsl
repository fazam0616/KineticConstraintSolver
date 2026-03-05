// lbvh_sort_edges.comp.glsl
// LBVH Pass 2 (edges): in-place bitonic sort of escratch by Morton code (slot 7).
// Identical logic to lbvh_sort_walls.comp.glsl; only the SSBO binding differs.
//
// Uniforms: u_n_prims, u_pass_len, u_sub_step, u_local (same semantics as walls).
#version 430 core
layout(local_size_x = 1024) in;

layout(std430, binding = 21) buffer EScratch { int escratch[]; };
// n_prims is written GPU-side by lbvh_prepare_edge_dispatch into edge_meta[0]
// so this shader works correctly with glDispatchComputeIndirect.
layout(std430, binding = 29) readonly buffer EdgeMeta { int edge_meta[]; };

// u_n_prims kept as a fallback uniform (set to -1 when indirect path is active)
uniform int u_n_prims;
uniform int u_pass_len;
uniform int u_sub_step;
uniform int u_local;

shared int sh[1024 * 8]; // 32 KB — fits in GL 4.3 minimum 48 KB per WG

void swap_prim_local(int a, int b) {
    for (int k = 0; k < 8; k++) {
        int tmp   = sh[a*8+k];
        sh[a*8+k] = sh[b*8+k];
        sh[b*8+k] = tmp;
    }
}

void swap_prim_global(int a, int b) {
    for (int k = 0; k < 8; k++) {
        int tmp          = escratch[a*8+k];
        escratch[a*8+k]  = escratch[b*8+k];
        escratch[b*8+k]  = tmp;
    }
}

void main() {
    int tid = int(gl_LocalInvocationID.x);
    int gid = int(gl_GlobalInvocationID.x);
    int n   = (u_n_prims >= 0) ? u_n_prims : edge_meta[0];

    if (u_local == 1) {
        if (tid < n) {
            for (int k = 0; k < 8; k++)
                sh[tid*8+k] = escratch[tid*8+k];
        } else {
            for (int k = 0; k < 7; k++) sh[tid*8+k] = 0;
            sh[tid*8+7] = int(0xFFFFFFFFu);
        }
        barrier();

        int n2 = 1024;
        for (int len = 2; len <= n2; len <<= 1) {
            for (int step = len >> 1; step > 0; step >>= 1) {
                int i  = tid;
                int j  = i ^ step;
                if (j > i) {
                    bool asc = ((i & len) == 0);
                    uint ki  = uint(sh[i*8+7]);
                    uint kj  = uint(sh[j*8+7]);
                    if ((asc && ki > kj) || (!asc && ki < kj))
                        swap_prim_local(i, j);
                }
                barrier();
            }
        }

        if (tid < n) {
            for (int k = 0; k < 8; k++)
                escratch[tid*8+k] = sh[tid*8+k];
        }
        return;
    }

    // Global pass
    int n2 = 1; while (n2 < n) n2 <<= 1;
    int half_n2 = n2 >> 1;
    if (gid >= half_n2) return;

    int i = gid;
    int j = i ^ u_sub_step;
    if (j <= i) return;
    if (i >= n || j >= n) return;

    bool asc = ((i & u_pass_len) == 0);
    uint ki  = uint(escratch[i*8+7]);
    uint kj  = uint(escratch[j*8+7]);
    if ((asc && ki > kj) || (!asc && ki < kj))
        swap_prim_global(i, j);
}
