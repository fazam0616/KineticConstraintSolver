#ifndef TRIANGLE_BVH_H
#define TRIANGLE_BVH_H

#include "datastructures.h"
#include "Node.h"
#include <stddef.h>

#include "AABB.h"

// Triangle BVH node
typedef struct TriangleBVHNode {
    AABB bounds;
    int is_leaf;
    union {
        struct {
            struct TriangleBVHNode *left;
            struct TriangleBVHNode *right;
        } children;
        struct {
            int tri_idx; // index into triangle wall array
        } leaf;
    } data;
} TriangleBVHNode;

// BVH root structure
typedef struct {
    TriangleBVHNode *root;
    size_t n_triangles;
    // Optionally, store triangle indices for refit
    int *triangle_indices;
} TriangleBVH;

TriangleBVH *triangle_bvh_build(DynArray *walls);
void triangle_bvh_free(TriangleBVH *bvh);
void triangle_bvh_refit(TriangleBVH *bvh, DynArray *walls);
void triangle_bvh_query(const TriangleBVH *bvh, const AABB *query, DynArray *out_indices);
void triangle_bvh_debug_draw(const TriangleBVH *bvh, int depth_limit);

#endif // TRIANGLE_BVH_H
