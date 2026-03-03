#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "datastructures.h"
#include "Node.h"
#include "Constraint.h"
#include "TriangleBVH.h"
#include "EdgeBVH.h"

typedef struct Simulator Simulator;

struct Simulator {
    DynArray *nodes; // Node*
    DynArray *constraints; // Constraint*
    DynArray *walls; // TriangleWall*
    float gravity[3];
    float dt;
    int solver_iters;
    float damping;
    float velocity_blend; // how much to blend projected velocity into previous velocity (0..1)
    TriangleBVH *triangle_bvh; // BVH for triangle walls
    EdgeBVH *edge_bvh;         // BVH for edges (constraints)
    // GPU integration toggle: when enabled the simulator will attempt to run
    // the integration/collision kernels on device. GPU resources are opaque
    // and managed inside Simulator.c via simulator_init_gpu()/simulator_free_gpu().
    int use_gpu;
    void *gpu_ctx; // opaque pointer to GPU resources (managed in Simulator.c)
    int gpu_debug; // when 1: also run CPU solver and print comparison each substep
};

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
