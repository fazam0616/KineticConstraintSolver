#ifndef EDGE_BVH_H
#define EDGE_BVH_H

#include "datastructures.h"
#include "Constraint.h"
#include <stddef.h>

#include "BVH.h"

// BVH wrapper for edges. Leaves store void* leaf_data that wrappers set to int* indices.
typedef struct {
    BVHNode *root;
    size_t n_edges;
    int *edge_indices;
} EdgeBVH;

EdgeBVH *edge_bvh_build(DynArray *edges, float capsule_radius);
void edge_bvh_free(EdgeBVH *bvh);
void edge_bvh_refit(EdgeBVH *bvh, DynArray *edges, float capsule_radius);
void edge_bvh_query(const EdgeBVH *bvh, const AABB *query, DynArray *out_indices);
void edge_bvh_self_traverse(const EdgeBVH *bvh, void (*callback)(int, int, void*), void *userdata);
void edge_bvh_debug_draw(const EdgeBVH *bvh, int depth_limit);

#endif // EDGE_BVH_H
