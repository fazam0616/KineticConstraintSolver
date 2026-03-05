// lbvh_build_walls.comp.glsl
// LBVH Pass 3 (walls): Karras 2012-style parallel BVH construction.
// Given n sorted primitives in wscratch (sorted by Morton code, slot [7]),
// builds a complete BVH with (n-1) internal nodes + n leaf nodes = 2n-1 total.
//
// Node index layout (compatible with collision.comp.glsl traversal):
//   Internal node i  →  bvh slot i         (i = 0..n-2)
//   Leaf node i      →  bvh slot (n-1 + i) (i = 0..n-1)
//
// BVH node layout (10 ints/node — unchanged from old serial builder):
//   [0..2] = AABB min.xyz  (floatBitsToInt)
//   [3]    = left child index  (-1 if leaf — compatible sentinel)
//   [4..6] = AABB max.xyz  (floatBitsToInt)
//   [7]    = right child index  OR  first_prim index in scratch (leaf)
//   [8]    = prim_count (0 = internal, 1 = leaf — always 1 for LBVH leaves)
//   [9]    = 0 (padding)
//
// AABB stored directly in node — no bottom-up propagation SSBO needed.
// For internal nodes the AABB is computed as the union of all prim AABBs
// in [first..last] (O(n) per thread in worst case, O(n log n) total).
// For n ≤ a few thousand this is fast enough per-frame.
//
// Dispatch: ceil((2*u_n_prims - 1) / 64) workgroups.
//   Thread gid < u_n_prims - 1  → internal node
//   Thread gid < 2*u_n_prims - 1 (and gid >= u_n_prims - 1) → leaf node
//
// Special case n == 1: dispatch 1 thread → single leaf as root (slot 0).
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 15) buffer BvhBuf     { int bvh_nodes[]; };
layout(std430, binding = 22) readonly buffer ScratchBuf { int wscratch[]; };

uniform int u_n_prims;

// ── Morton delta: number of common high-order bits between keys i and j ──────
// Returns -1 if i or j is out of range (handles boundary conditions).
int delta(int i, int j, int n) {
    if (j < 0 || j >= n) return -1;
    uint ki = uint(wscratch[i*8+7]);
    uint kj = uint(wscratch[j*8+7]);
    // If keys are identical, use index as tiebreaker (guarantees unique ordering).
    if (ki == kj) return 32 + (31 - findMSB(uint(i ^ j)));
    return 31 - findMSB(ki ^ kj);
}

// ── Karras 2012: determine range [first, last] for internal node idx ─────────
ivec2 determineRange(int idx, int n) {
    // Direction: +1 or -1
    int d = (delta(idx, idx + 1, n) >= delta(idx, idx - 1, n)) ? 1 : -1;

    // Upper bound on range length
    int delta_min = delta(idx, idx - d, n);
    int l_max = 2;
    while (delta(idx, idx + l_max * d, n) > delta_min)
        l_max <<= 1;

    // Binary search for the other end
    int l = 0;
    for (int t = l_max >> 1; t >= 1; t >>= 1)
        if (delta(idx, idx + (l + t) * d, n) > delta_min)
            l += t;

    int j = idx + l * d;
    return ivec2(min(idx, j), max(idx, j));
}

// ── Karras 2012: find split position within [first, last] ────────────────────
int findSplit(int first, int last, int n) {
    int delta_node = delta(first, last, n);

    int split = first;
    int step  = last - first;
    do {
        step = (step + 1) >> 1;
        int new_split = split + step;
        if (new_split < last && delta(first, new_split, n) > delta_node)
            split = new_split;
    } while (step > 1);
    return split;
}

// ── Compute tight AABB over sorted prim range [first..last] ──────────────────
void range_aabb(int first, int last, out vec3 mn, out vec3 mx) {
    mn = vec3( 1e30);
    mx = vec3(-1e30);
    for (int k = first; k <= last; k++) {
        vec3 lo = vec3(intBitsToFloat(wscratch[k*8+0]),
                       intBitsToFloat(wscratch[k*8+1]),
                       intBitsToFloat(wscratch[k*8+2]));
        vec3 hi = vec3(intBitsToFloat(wscratch[k*8+3]),
                       intBitsToFloat(wscratch[k*8+4]),
                       intBitsToFloat(wscratch[k*8+5]));
        mn = min(mn, lo);
        mx = max(mx, hi);
    }
}

// ── Write a BVH node ──────────────────────────────────────────────────────────
void write_node(int slot, vec3 mn, vec3 mx, int left, int right, int count) {
    bvh_nodes[slot*10+0] = floatBitsToInt(mn.x);
    bvh_nodes[slot*10+1] = floatBitsToInt(mn.y);
    bvh_nodes[slot*10+2] = floatBitsToInt(mn.z);
    bvh_nodes[slot*10+3] = left;
    bvh_nodes[slot*10+4] = floatBitsToInt(mx.x);
    bvh_nodes[slot*10+5] = floatBitsToInt(mx.y);
    bvh_nodes[slot*10+6] = floatBitsToInt(mx.z);
    bvh_nodes[slot*10+7] = right;
    bvh_nodes[slot*10+8] = count;
    bvh_nodes[slot*10+9] = 0;
}

void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int n   = u_n_prims;

    // ── Special case: single primitive ───────────────────────────────────────
    if (n == 1) {
        if (gid == 0) {
            vec3 mn = vec3(intBitsToFloat(wscratch[0]),
                           intBitsToFloat(wscratch[1]),
                           intBitsToFloat(wscratch[2]));
            vec3 mx = vec3(intBitsToFloat(wscratch[3]),
                           intBitsToFloat(wscratch[4]),
                           intBitsToFloat(wscratch[5]));
            write_node(0, mn, mx, -1, 0, 1);
        }
        return;
    }

    int total = 2 * n - 1;
    if (gid >= total) return;

    if (gid < n - 1) {
        // ── Internal node ─────────────────────────────────────────────────
        int idx   = gid;
        ivec2 rng = determineRange(idx, n);
        int first = rng.x, last = rng.y;
        int split = findSplit(first, last, n);

        // Children: if a child range is a single prim, point to leaf slot (n-1+prim)
        int left  = (split     == first) ? (n - 1 + split)     : split;
        int right = (split + 1 == last)  ? (n - 1 + split + 1) : split + 1;

        vec3 mn, mx;
        range_aabb(first, last, mn, mx);
        write_node(idx, mn, mx, left, right, 0);
    } else {
        // ── Leaf node ─────────────────────────────────────────────────────
        int prim_idx = gid - (n - 1);  // index into sorted scratch
        vec3 mn = vec3(intBitsToFloat(wscratch[prim_idx*8+0]),
                       intBitsToFloat(wscratch[prim_idx*8+1]),
                       intBitsToFloat(wscratch[prim_idx*8+2]));
        vec3 mx = vec3(intBitsToFloat(wscratch[prim_idx*8+3]),
                       intBitsToFloat(wscratch[prim_idx*8+4]),
                       intBitsToFloat(wscratch[prim_idx*8+5]));
        // Slot = n-1+prim_idx.  left=-1 marks leaf; right=scratch index.
        write_node(gid, mn, mx, -1, prim_idx, 1);
    }
}
