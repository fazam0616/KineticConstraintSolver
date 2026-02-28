#ifndef TRIANGLE_BVH_H
#define TRIANGLE_BVH_H

#include "datastructures.h"
#include "Node.h"
#include <stddef.h>

#include "BVH.h"

// BVH wrapper for triangles. Leaves store void* leaf_data that wrappers set to int* indices.
typedef struct {
    BVHNode *root;
    size_t n_triangles;
    int *triangle_indices;
} TriangleBVH;

TriangleBVH *triangle_bvh_build(DynArray *walls);
void triangle_bvh_free(TriangleBVH *bvh);
void triangle_bvh_refit(TriangleBVH *bvh, DynArray *walls);
void triangle_bvh_query(const TriangleBVH *bvh, const AABB *query, DynArray *out_indices);
void triangle_bvh_debug_draw(const TriangleBVH *bvh, int depth_limit);

#endif // TRIANGLE_BVH_H
