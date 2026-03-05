// lbvh_morton_walls.comp.glsl
// LBVH Pass 1 (walls): one thread per triangle.
// Computes triangle AABB + 30-bit Morton code from centroid.
// Writes to wscratch (binding 22), 8 ints/prim:
//   [0..2] = AABB min.xyz (floatBitsToInt)
//   [3..5] = AABB max.xyz (floatBitsToInt)
//   [6]    = original triangle index
//   [7]    = Morton code (as int; treated as uint during sort)
//
// Dispatch: ceil(u_n_tris / 64) workgroups.
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding =  0) readonly buffer PosBuf     { vec4 pos_r[]; };
layout(std430, binding = 17) readonly buffer WallIdxBuf { int  widx[]; };
layout(std430, binding = 22)          buffer ScratchBuf { int  wscratch[]; };

uniform int  u_n_tris;
uniform vec3 u_scene_min;
uniform vec3 u_scene_max;

const float PAD = 0.01;

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
    int i = int(gl_GlobalInvocationID.x);
    if (i >= u_n_tris) return;

    int ai = widx[i*4+0], bi = widx[i*4+1], ci = widx[i*4+2];
    vec3 A = pos_r[ai].xyz, B = pos_r[bi].xyz, C = pos_r[ci].xyz;
    vec3 mn = min(A, min(B, C)) - PAD;
    vec3 mx = max(A, max(B, C)) + PAD;

    wscratch[i*8+0] = floatBitsToInt(mn.x);
    wscratch[i*8+1] = floatBitsToInt(mn.y);
    wscratch[i*8+2] = floatBitsToInt(mn.z);
    wscratch[i*8+3] = floatBitsToInt(mx.x);
    wscratch[i*8+4] = floatBitsToInt(mx.y);
    wscratch[i*8+5] = floatBitsToInt(mx.z);
    wscratch[i*8+6] = i;

    vec3 cen  = (mn + mx) * 0.5;
    vec3 span = u_scene_max - u_scene_min;
    vec3 norm = (span.x > 1e-9 && span.y > 1e-9 && span.z > 1e-9)
                ? (cen - u_scene_min) / span : vec3(0.5);
    wscratch[i*8+7] = int(morton3D(norm));
}
