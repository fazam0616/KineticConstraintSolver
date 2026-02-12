#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "datastructures.h"
#include "Node.h"
#include "Constraint.h"
#include "Octree.h"

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
    OctreeNode *octree; // spatial acceleration for collision detection
};

Simulator* simulator_create(float dt);
void simulator_free(Simulator *s);
void simulator_add_node(Simulator *s, Node *n);
void simulator_add_constraint(Simulator *s, Constraint *c);
void simulator_add_wall(Simulator *s, TriangleWall *w);
void simulator_step(Simulator *s);
void simulator_draw(Simulator *s, float cam_yaw, float cam_pitch);

// Find the closest node to a given position within max_distance
// Returns node index or -1 if none found within range
// Uses octree for efficient spatial lookup
int simulator_find_closest_node(Simulator *s, float pos[3], float max_distance);

// Generate a triangular mesh from a polygon defined by a DynArray of Node*
// - sim: simulator to which new nodes/constraints will be added
// - poly_nodes: DynArray of Node* defining polygon vertices (in order)
// - k: subdivision factor; number of samples per edge will be k (k>=1)
// Returns a DynArray* of Triangle* where Triangle is an allocated struct { Node *a,*b,*c }.
// Caller is responsible for freeing the returned DynArray and triangle structs.
DynArray* simulator_generate_mesh_from_nodes(Simulator *sim, DynArray *poly_nodes, int k);

#endif // SIMULATOR_H
