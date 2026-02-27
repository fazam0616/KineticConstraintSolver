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
};

Simulator* simulator_create(float dt);
void simulator_free(Simulator *s);
void simulator_add_node(Simulator *s, Node *n);
void simulator_add_constraint(Simulator *s, Constraint *c);
void simulator_add_wall(Simulator *s, TriangleWall *w);
void simulator_step(Simulator *s);
void simulator_draw(Simulator *s, float cam_yaw, float cam_pitch);


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
