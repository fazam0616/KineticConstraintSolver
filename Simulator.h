#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "datastructures.h"
#include "Node.h"
#include "Constraint.h"
#include "TriangleBVH.h"
#include "EdgeBVH.h"
#include "ParticleSim.h"

typedef struct Simulator Simulator;

struct Simulator {
    DynArray *nodes; // Node*
    DynArray *constraints; // Constraint*
    DynArray *walls; // TriangleWall*
    float gravity[3];
    float dt;
    int solver_iters;
    float damping;
    TriangleBVH *triangle_bvh; // BVH for triangle walls
    EdgeBVH *edge_bvh;         // BVH for edges (constraints)
    // GPU integration toggle: when enabled the simulator will attempt to run
    // the integration/collision kernels on device. GPU resources are opaque
    // and managed inside Simulator.c via simulator_init_gpu()/simulator_free_gpu().
    int use_gpu;
    void *gpu_ctx; // opaque pointer to GPU resources (managed in Simulator.c)
    int gpu_debug; // when 1: also run CPU solver and print comparison each substep
    ParticleSim *particles; // optional particle fluid sim coupled to constraint mesh
};

// ── Per-frame timing breakdown (updated each frame when use_gpu==1) ──────────
// CPU times are wall-clock milliseconds measured around each dispatch group.
// GPU times are elapsed nanoseconds from GL_TIME_ELAPSED queries (1-frame delayed
// to avoid pipeline stalls); divide by 1e6 to get ms.
typedef struct {
    // CPU wall-clock (ms) for each major phase
    double cpu_lbvh_ms;       // edge LBVH build (all substeps)
    double cpu_wall_bvh_ms;   // wall LBVH rebuild (all substeps)
    double cpu_collision_ms;  // sphere-tri + edge-edge (all substeps)
    double cpu_cg_ms;         // full CG block (all substeps)
    double cpu_apply_ms;      // apply_corr (all substeps)
    double cpu_particles_ms;  // particle_sim_step (all substeps)
    double cpu_total_ms;      // total simulator_gpu_step wall time
    // GPU elapsed time (ms) for each phase — 1-frame delayed
    double gpu_lbvh_ms;
    double gpu_wall_bvh_ms;
    double gpu_collision_ms;
    double gpu_cg_ms;
    double gpu_apply_ms;
    double gpu_particles_ms;
} SimTimings;

// Returns a pointer to the most recently completed SimTimings (read-only).
// Valid after the first call to simulator_gpu_step().
const SimTimings* simulator_get_timings(const Simulator *s);

Simulator* simulator_create(float dt);
void simulator_free(Simulator *s);
void simulator_add_node(Simulator *s, Node *n);
void simulator_add_constraint(Simulator *s, Constraint *c);
void simulator_add_wall(Simulator *s, TriangleWall *w);
void simulator_step(Simulator *s);
void simulator_draw(Simulator *s, float cam_yaw, float cam_pitch);

// Packed scene layout suitable for GPU upload (SoA buffers)
typedef struct PackedScene {
    int node_count;
    float *positions;   // len = node_count * 3
    float *velocities;  // len = node_count * 3
    float *inv_mass;    // len = node_count
    float *radius;      // len = node_count
    unsigned int *flags; // len = node_count (bitflags: anchored/collide/sim_ignore)

    int constraint_count;
    int *ctype;   // len = constraint_count
    int *a_idx;   // len = constraint_count
    int *b_idx;   // len = constraint_count (or -1)
    float *rest;  // len = constraint_count
    float *stiffness; // len = constraint_count
    float *anchor_pos; // len = constraint_count * 3 (non-zero only for CT_ANCHOR)
    int m_sparse;      // number of non-spring constraints (packed first in all arrays)

    int triangle_count;
    float *triangle_vertices; // len = triangle_count * 9 (3 verts * 3 components)
} PackedScene;

// Create a PackedScene from the pointer-based Simulator layout.
// Caller must free the returned PackedScene with simulator_free_packed_scene().
PackedScene* simulator_pack_scene(Simulator *s);
void simulator_free_packed_scene(PackedScene *p);

// GPU lifecycle and upload helpers
int simulator_init_gpu(Simulator *s);
void simulator_free_gpu(Simulator *s);
void simulator_upload_scene_to_gpu(Simulator *s, PackedScene *p);
void simulator_enable_gpu(Simulator *s, int enable);

// Perform one simulation step on GPU (requires init + uploaded scene)
void simulator_gpu_step(Simulator *s);

// Sync GPU positions/velocities back to CPU Node structs (call once per render frame for UI)
void simulator_sync_positions(Simulator *s);

// Enable/disable dynamic wall BVH rebuild every substep (needed when wall nodes are mobile)
void simulator_set_rebuild_wall_bvh(Simulator *s, int enable);
void simulator_set_edge_edge(Simulator *s, int enable);

// Re-upload only the translucency flags (widx[i*4+3]) of all walls to the GPU.
// Call this whenever wall->translucent is changed at runtime.
void simulator_upload_wall_flags(Simulator *s);


// Query triangles overlapping an AABB (for node→face queries)
void simulator_query_triangles(Simulator *s, const AABB *query, DynArray *out_indices);

// Query edges overlapping an AABB (for edge→edge broadphase)
void simulator_query_edges(Simulator *s, const AABB *query, DynArray *out_indices);

// Self-traverse edge BVH for edge-edge broadphase
void simulator_edge_bvh_self_traverse(Simulator *s, void (*callback)(int, int, void*), void *userdata);

// Generate a triangular mesh from a polygon defined by a DynArray of Node*
// - sim: simulator to which new nodes/constraints will be added
// - poly_nodes: DynArray of Node* defining polygon vertices (in order)
// - k: subdivision factor; number of samples per edge will be k (k>=1)
// Returns a DynArray* of Triangle* where Triangle is an allocated struct { Node *a,*b,*c }.
// Caller is responsible for freeing the returned DynArray and triangle structs.
DynArray* simulator_generate_mesh_from_nodes(Simulator *sim, DynArray *poly_nodes, int k);

#endif // SIMULATOR_H
