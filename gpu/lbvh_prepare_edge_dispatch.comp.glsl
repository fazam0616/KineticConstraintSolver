// lbvh_prepare_edge_dispatch.comp.glsl
// 1-thread shader: reads GPU-side edge count written by lbvh_morton_edges
// (atomicAdd into escratch[u_m_total * 8]) and writes:
//   ssbo_edge_meta[0]   = n_edges          (binding 29)
//   ssbo_indirect_args  = {sort_x, 1, 1,   (binding 28, byte offset 0)
//                           build_x, 1, 1}  (binding 28, byte offset 12)
// Dispatch: (1, 1, 1) — single thread.
// Must be followed by glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT |
//                                     GL_COMMAND_BARRIER_BIT)
// before the indirect sort and build dispatches.
#version 430 core
layout(local_size_x = 1) in;

// escratch: counter stored at element [u_m_total * 8]
layout(std430, binding = 21) readonly buffer EScratch  { int escratch[]; };

// indirect_args: two DispatchIndirectCommand structs (3 uints each = 12 bytes each)
//   [0..2]  = sort  {groups_x, 1, 1}
//   [3..5]  = build {groups_x, 1, 1}
layout(std430, binding = 28) writeonly buffer IndirectArgs { uint indirect_args[]; };

// edge_meta: [0] = n_edges (read by sort and build shaders as u_n_prims replacement)
layout(std430, binding = 29) writeonly buffer EdgeMeta { int edge_meta[]; };

uniform int u_m_total;  // used to locate the atomic counter in escratch

void main() {
    int n_edges = escratch[u_m_total * 8];

    // clamp to valid range: 0 means nothing to do (dispatch 0 groups = GPU no-op)
    if (n_edges < 0) n_edges = 0;

    edge_meta[0] = n_edges;

    // Sort dispatch: ceil(n_edges / 2 / 64) groups for global path,
    // OR 1 group for local path.  The sort shader reads edge_meta[0]
    // and checks n_prims itself, so we always write ceil(n_edges/2/64)
    // clamped to at least 1 so the shader can early-exit cleanly.
    // For the local path the sort shader uses u_local=1 → 1 WG covers all.
    // We over-dispatch conservatively: sort_x = max(1, ceil(n_edges / 2 / 64)).
    // Extra threads in the global path check (gid >= half_n2) and return.
    uint sort_x  = (n_edges <= 0) ? 0u : uint((n_edges / 2 + 63) / 64);
    if (sort_x < 1u && n_edges > 0) sort_x = 1u;

    // Build dispatch: ceil((2*n_edges - 1) / 64) — for n_edges==1: 1 group.
    uint total   = (n_edges <= 0) ? 0u :
                   (n_edges == 1) ? 1u : uint(2 * n_edges - 1);
    uint build_x = (total == 0u) ? 0u : (total + 63u) / 64u;

    // Write sort args (offset 0)
    indirect_args[0] = sort_x;
    indirect_args[1] = 1u;
    indirect_args[2] = 1u;

    // Write build args (offset 12 bytes = element 3)
    indirect_args[3] = build_x;
    indirect_args[4] = 1u;
    indirect_args[5] = 1u;
}
