#ifndef OCTREE_H
#define OCTREE_H

#include "datastructures.h"
#include "Node.h"

// Forward declarations
typedef struct OctreeNode OctreeNode;
typedef struct TriangleWall TriangleWall;

#include "BVH.h"

// Node entry that stores both index and position for redistribution
typedef struct {
    int node_idx;
    float pos[3];
} NodeEntry;

// Octree node for spatial partitioning
struct OctreeNode {
    AABB bounds;
    OctreeNode *children[8];  // NULL if leaf
    DynArray *node_entries;   // NodeEntry* in this cell (leaf only)
    DynArray *triangle_indices; // Indices of TriangleWalls stored at this level
    DynArray *constraint_indices; // Indices of constraints (for edge-edge collisions)
    int depth;
    int is_leaf;
};

// Register edge in all cubes it intersects
void octree_insert_constraint_all_cubes(OctreeNode *root, int constraint_idx, float pos_a[3], float pos_b[3]);

// Create and destroy octree
OctreeNode* octree_create(float min_x, float min_y, float min_z, 
                          float max_x, float max_y, float max_z, int max_depth);
void octree_free(OctreeNode *node);

// Insert a node (by index) into the octree at the appropriate leaf
void octree_insert_node(OctreeNode *root, int node_idx, float pos[3]);

// Remove a node (by index) from the octree using its previous position
void octree_remove(OctreeNode *root, int node_idx, float pos[3]);

// Insert a constraint (by index) into ancestor node containing both endpoints
void octree_insert_constraint(OctreeNode *root, int constraint_idx,
                             float pos_a[3], float pos_b[3]);

// Insert a constraint (by index) into the ancestor node containing both endpoints
void octree_insert_constraint(OctreeNode *root, int constraint_idx, 
                             float pos_a[3], float pos_b[3]);

// Insert a triangle (by index) into the octree at the minimal common ancestor
// of all three vertices. Returns the depth at which it was stored.
int octree_insert_triangle(OctreeNode *root, int tri_idx, 
                           float a[3], float b[3], float c[3]);

// Query triangles that could collide with a point at given position.
// Returns a DynArray* of int* (triangle indices). Caller must free.
DynArray* octree_query_triangles(OctreeNode *root, float pos[3]);

// Query nodes in the leaf cell containing the given position.
// Returns a DynArray* of int* (node indices). Caller must free.
DynArray* octree_query_nodes(OctreeNode *root, float pos[3]);

// Query constraints in region containing both points (finds ancestor node).
// Returns a DynArray* of int* (constraint indices). Caller must free.
DynArray* octree_query_constraints(OctreeNode *root, float pos_a[3], float pos_b[3]);

// Find the closest node to a given position within max_distance.
// Returns node index or -1 if none found.
int octree_find_closest_node(OctreeNode *root, Node **all_nodes, size_t num_nodes,
                             float pos[3], float max_distance);

// Clear all node and triangle indices from the octree (for rebuilding)
void octree_clear(OctreeNode *root);

// Helper: get which octant (0-7) a point belongs to relative to center
int octree_get_octant(const float center[3], const float point[3]);

// Helper: check if AABB contains point
int aabb_contains_point(const AABB *box, const float point[3]);

// Helper: compute AABB that contains three points
AABB aabb_from_triangle(const float a[3], const float b[3], const float c[3]);

// Helper: check if two AABBs intersect
int aabb_intersects(const AABB *a, const AABB *b);

#endif // OCTREE_H
