// lbvh_morton_edges.comp.glsl
// LBVH Pass 1 (edges): one thread per constraint; skips non-CT_DIST.
// Atomically compacts CT_DIST edges into escratch (binding 21), 8 ints/prim:
//   [0..2] = AABB min.xyz (floatBitsToInt)
//   [3..5] = AABB max.xyz (floatBitsToInt)
//   [6]    = original constraint index
//   [7]    = Morton code (as int; treated as uint during sort)
//
// The atomic counter is stored at escratch[u_m_total * 8].
// CPU must allocate the SSBO with 1 extra int and zero that slot before dispatch.
// After the barrier, CPU reads back that counter to get the actual edge count
// (u_n_edges) needed for the sort and build passes.
//
// Dispatch: ceil(u_m_total / 64) workgroups.
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding =  0) readonly buffer PosBuf  { vec4 pos_r[]; };
layout(std430, binding =  5) readonly buffer ConBuf  { int  cdata[]; };
layout(std430, binding = 21)          buffer EScratch { int  escratch[]; };

uniform int  u_m_total;
uniform vec3 u_scene_min;
uniform vec3 u_scene_max;

const int CT_DIST = 0;

uint expand_bits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}
uint morton3D(vec3 p) {
    p = clamp(p, vec3(0.0), vec3(1.0));
    uint x = uint(p.x * 1023.0);
    uint y = uint(p.y * 1023.0);
    uint z = uint(p.z * 1023.0);
    return (expand_bits(x) << 2u) | (expand_bits(y) << 1u) | expand_bits(z);
}

void main() {
    int c = int(gl_GlobalInvocationID.x);
    if (c >= u_m_total) return;
    if (cdata[c*8 + 0] != CT_DIST) return;

    int ai = cdata[c*8 + 1];
    int bi = cdata[c*8 + 2];
    if (ai < 0 || bi < 0) return;

    // Compact into next free slot
    int slot = atomicAdd(escratch[u_m_total * 8], 1);

    vec3 A  = pos_r[ai].xyz;
    vec3 B  = pos_r[bi].xyz;
    float r = (pos_r[ai].w + pos_r[bi].w) * 0.5;
    vec3 mn = min(A, B) - r;
    vec3 mx = max(A, B) + r;

    escratch[slot*8+0] = floatBitsToInt(mn.x);
    escratch[slot*8+1] = floatBitsToInt(mn.y);
    escratch[slot*8+2] = floatBitsToInt(mn.z);
    escratch[slot*8+3] = floatBitsToInt(mx.x);
    escratch[slot*8+4] = floatBitsToInt(mx.y);
    escratch[slot*8+5] = floatBitsToInt(mx.z);
    escratch[slot*8+6] = c;

    vec3 cen  = (mn + mx) * 0.5;
    vec3 span = u_scene_max - u_scene_min;
    vec3 norm = (span.x > 1e-9 && span.y > 1e-9 && span.z > 1e-9)
                ? (cen - u_scene_min) / span : vec3(0.5);
    escratch[slot*8+7] = int(morton3D(norm));
}
