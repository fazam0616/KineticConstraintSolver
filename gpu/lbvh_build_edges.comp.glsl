// lbvh_build_edges.comp.glsl
// LBVH Pass 3 (edges): Karras 2012-style parallel BVH construction.
// Identical to lbvh_build_walls.comp.glsl; only the SSBO bindings differ.
//
// Reads  escratch (binding 21) for sorted prim AABBs + Morton codes.
// Writes ebvh     (binding 20) for BVH nodes.
//
// Node index layout:
//   Internal node i  →  ebvh slot i         (i = 0..n-2)
//   Leaf node i      →  ebvh slot (n-1 + i) (i = 0..n-1)
//
// Dispatch: ceil((2*u_n_prims - 1) / 64) workgroups.
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 20) buffer EBvhBuf  { int ebvh[]; };
layout(std430, binding = 21) readonly buffer EScratch { int escratch[]; };
// n_prims is written GPU-side by lbvh_prepare_edge_dispatch into edge_meta[0]
layout(std430, binding = 29) readonly buffer EdgeMeta { int edge_meta[]; };

// u_n_prims: -1 means read from edge_meta[0] (indirect path)
uniform int u_n_prims;

int delta(int i, int j, int n) {
    if (j < 0 || j >= n) return -1;
    uint ki = uint(escratch[i*8+7]);
    uint kj = uint(escratch[j*8+7]);
    if (ki == kj) return 32 + (31 - findMSB(uint(i ^ j)));
    return 31 - findMSB(ki ^ kj);
}

ivec2 determineRange(int idx, int n) {
    int d = (delta(idx, idx + 1, n) >= delta(idx, idx - 1, n)) ? 1 : -1;
    int delta_min = delta(idx, idx - d, n);
    int l_max = 2;
    while (delta(idx, idx + l_max * d, n) > delta_min)
        l_max <<= 1;
    int l = 0;
    for (int t = l_max >> 1; t >= 1; t >>= 1)
        if (delta(idx, idx + (l + t) * d, n) > delta_min)
            l += t;
    int j = idx + l * d;
    return ivec2(min(idx, j), max(idx, j));
}

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

void range_aabb(int first, int last, out vec3 mn, out vec3 mx) {
    mn = vec3( 1e30);
    mx = vec3(-1e30);
    for (int k = first; k <= last; k++) {
        vec3 lo = vec3(intBitsToFloat(escratch[k*8+0]),
                       intBitsToFloat(escratch[k*8+1]),
                       intBitsToFloat(escratch[k*8+2]));
        vec3 hi = vec3(intBitsToFloat(escratch[k*8+3]),
                       intBitsToFloat(escratch[k*8+4]),
                       intBitsToFloat(escratch[k*8+5]));
        mn = min(mn, lo);
        mx = max(mx, hi);
    }
}

void write_node(int slot, vec3 mn, vec3 mx, int left, int right, int count) {
    ebvh[slot*10+0] = floatBitsToInt(mn.x);
    ebvh[slot*10+1] = floatBitsToInt(mn.y);
    ebvh[slot*10+2] = floatBitsToInt(mn.z);
    ebvh[slot*10+3] = left;
    ebvh[slot*10+4] = floatBitsToInt(mx.x);
    ebvh[slot*10+5] = floatBitsToInt(mx.y);
    ebvh[slot*10+6] = floatBitsToInt(mx.z);
    ebvh[slot*10+7] = right;
    ebvh[slot*10+8] = count;
    ebvh[slot*10+9] = 0;
}

void main() {
    int gid = int(gl_GlobalInvocationID.x);
    int n   = (u_n_prims >= 0) ? u_n_prims : edge_meta[0];
    if (n <= 0) return;

    if (n == 1) {
        if (gid == 0) {
            vec3 mn = vec3(intBitsToFloat(escratch[0]),
                           intBitsToFloat(escratch[1]),
                           intBitsToFloat(escratch[2]));
            vec3 mx = vec3(intBitsToFloat(escratch[3]),
                           intBitsToFloat(escratch[4]),
                           intBitsToFloat(escratch[5]));
            write_node(0, mn, mx, -1, 0, 1);
        }
        return;
    }

    int total = 2 * n - 1;
    if (gid >= total) return;

    if (gid < n - 1) {
        int idx   = gid;
        ivec2 rng = determineRange(idx, n);
        int first = rng.x, last = rng.y;
        int split = findSplit(first, last, n);

        int left  = (split     == first) ? (n - 1 + split)     : split;
        int right = (split + 1 == last)  ? (n - 1 + split + 1) : split + 1;

        vec3 mn, mx;
        range_aabb(first, last, mn, mx);
        write_node(idx, mn, mx, left, right, 0);
    } else {
        int prim_idx = gid - (n - 1);
        vec3 mn = vec3(intBitsToFloat(escratch[prim_idx*8+0]),
                       intBitsToFloat(escratch[prim_idx*8+1]),
                       intBitsToFloat(escratch[prim_idx*8+2]));
        vec3 mx = vec3(intBitsToFloat(escratch[prim_idx*8+3]),
                       intBitsToFloat(escratch[prim_idx*8+4]),
                       intBitsToFloat(escratch[prim_idx*8+5]));
        write_node(gid, mn, mx, -1, prim_idx, 1);
    }
}
