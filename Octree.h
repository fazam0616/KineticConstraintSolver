#ifndef OCTREE_H
#define OCTREE_H

#include "datastructures.h"
#include "Node.h"

// Forward declarations
typedef struct OctreeNode OctreeNode;
typedef struct TriangleWall TriangleWall;

// Axis-aligned bounding box
typedef struct {
    float min[3];
    float max[3];
} AABB;

// Octree node for spatial partitioning
struct OctreeNode {
    AABB bounds;
    OctreeNode *children[8];  // NULL if leaf
    DynArray *node_indices;   // Indices of Nodes in this cell (leaf only)
    DynArray *triangle_indices; // Indices of TriangleWalls stored at this level
    int depth;
    int is_leaf;
};

// Create and destroy octree
OctreeNode* octree_create(float min_x, float min_y, float min_z, 
                          float max_x, float max_y, float max_z, int max_depth);
void octree_free(OctreeNode *node);

// Insert a node (by index) into the octree at the appropriate leaf
void octree_insert_node(OctreeNode *root, int node_idx, float pos[3]);

// Insert a triangle (by index) into the octree at the minimal common ancestor
// of all three vertices. Returns the depth at which it was stored.
int octree_insert_triangle(OctreeNode *root, int tri_idx, 
                           float a[3], float b[3], float c[3]);

// Query triangles that could collide with a point at given position.
// Returns a DynArray* of int* (triangle indices). Caller must free.
DynArray* octree_query_triangles(OctreeNode *root, float pos[3]);

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
