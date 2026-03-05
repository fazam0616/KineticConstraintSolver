// lbvh_sort_walls.comp.glsl
// LBVH Pass 2 (walls): in-place bitonic sort of wscratch by Morton code (slot 7).
//
// Threshold: if n <= SORT_LOCAL_MAX we use a single-workgroup shared-memory
// sort (local_size_x=512, dispatch(1,1,1)).
// Otherwise (n > 512) the CPU dispatches this shader multiple times with
// different (u_pass_len, u_sub_step) values for a global bitonic network.
//
// Uniforms:
//   u_n_prims   – number of primitives to sort
//   u_pass_len  – current "outer" distance (power of 2: 2,4,8,...,n)
//   u_sub_step  – current "inner" distance (power of 2: pass_len/2,...,1)
//   u_local     – 1 = local shared-memory sort (n<=512), 0 = global pass
//
// Each thread handles one compare-and-swap pair spaced u_sub_step apart.
// Dispatch for global pass: ceil(n/2 / 64) workgroups.
//
// Scratch layout (8 ints/prim):
//   [7] = Morton code (uint key), sort ascending.
#version 430 core
layout(local_size_x = 1024) in;

layout(std430, binding = 22) buffer ScratchBuf { int wscratch[]; };
// n_prims written GPU-side into edge_meta[1] for walls (walls are static,
// so this is populated once at scene upload; indirect path not used for walls).
layout(std430, binding = 29) readonly buffer EdgeMeta { int edge_meta[]; };

// u_n_prims kept as primary; -1 signals "read from edge_meta[1]"
uniform int u_n_prims;
uniform int u_pass_len;   // used only when u_local == 0
uniform int u_sub_step;   // used only when u_local == 0
uniform int u_local;      // 1 = shared-memory sort, 0 = one global step

// ── Shared memory for local sort ─────────────────────────────────────────────
// 1024 prims * 8 ints = 8192 ints = 32 KB shared mem
shared int sh[1024 * 8];

void swap_prim_local(int a, int b) {
    for (int k = 0; k < 8; k++) {
        int tmp    = sh[a*8+k];
        sh[a*8+k]  = sh[b*8+k];
        sh[b*8+k]  = tmp;
    }
}

void swap_prim_global(int a, int b) {
    for (int k = 0; k < 8; k++) {
        int tmp          = wscratch[a*8+k];
        wscratch[a*8+k]  = wscratch[b*8+k];
        wscratch[b*8+k]  = tmp;
    }
}

void main() {
    int tid = int(gl_LocalInvocationID.x);
    int gid = int(gl_GlobalInvocationID.x);
    int n   = (u_n_prims >= 0) ? u_n_prims : edge_meta[1];

    // ── Local shared-memory bitonic sort (n <= 512) ───────────────────────
    if (u_local == 1) {
        // Load up to 512 prims into shared memory
        if (tid < n) {
            for (int k = 0; k < 8; k++)
                sh[tid*8+k] = wscratch[tid*8+k];
        } else {
            // Pad with max value so they sink to the end
            for (int k = 0; k < 7; k++) sh[tid*8+k] = 0;
            sh[tid*8+7] = int(0xFFFFFFFFu);
        }
        barrier();

        // Round n up to next power of two for clean bitonic
        int n2 = 1;
        while (n2 < 1024) n2 <<= 1;  // always 1024 since local_size_x=1024

        for (int len = 2; len <= n2; len <<= 1) {
            for (int step = len >> 1; step > 0; step >>= 1) {
                int i   = tid;
                int j   = i ^ step;
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

        // Write back
        if (tid < n) {
            for (int k = 0; k < 8; k++)
                wscratch[tid*8+k] = sh[tid*8+k];
        }
        return;
    }

    // ── Global bitonic sort pass ──────────────────────────────────────────
    // Each thread handles one compare-and-swap pair.
    // Pair: (i, j) where j = i ^ u_sub_step.
    int n2 = 1; while (n2 < n) n2 <<= 1;
    int half_n2 = n2 >> 1;
    if (gid >= half_n2) return;

    int i = gid;
    int j = i ^ u_sub_step;
    if (j <= i) return;
    if (i >= n || j >= n) return;

    bool asc = ((i & u_pass_len) == 0);
    uint ki  = uint(wscratch[i*8+7]);
    uint kj  = uint(wscratch[j*8+7]);
    if ((asc && ki > kj) || (!asc && ki < kj))
        swap_prim_global(i, j);
}
