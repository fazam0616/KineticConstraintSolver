#ifndef BVH_H
#define BVH_H

#include <float.h>

typedef struct {
    float min[3];
    float max[3];
} AABB;

static inline void aabb_init_empty(AABB *a) {
    for (int i = 0; i < 3; ++i) {
        a->min[i] = FLT_MAX;
        a->max[i] = -FLT_MAX;
    }
}

static inline void aabb_expand(AABB *a, const AABB *b) {
    for (int i = 0; i < 3; ++i) {
        if (b->min[i] < a->min[i]) a->min[i] = b->min[i];
        if (b->max[i] > a->max[i]) a->max[i] = b->max[i];
    }
}

static inline int axis_of_max_extent(const AABB *a) {
    float ext[3] = {a->max[0]-a->min[0], a->max[1]-a->min[1], a->max[2]-a->min[2]};
    int axis = 0;
    if (ext[1] > ext[axis]) axis = 1;
    if (ext[2] > ext[axis]) axis = 2;
    return axis;
}

// Generic BVH node. Leaf nodes store `leaf_data` (void*), wrappers decide what it points to.
typedef struct BVHNode {
    AABB bounds;
    int is_leaf;
    struct BVHNode *left;
    struct BVHNode *right;
    void *leaf_data; // user-defined pointer (e.g., int* index)
} BVHNode;

// Free a BVHNode tree. Frees nodes themselves but does NOT free leaf_data.
// Wrapper code should free any allocated leaf_data if needed before calling this.
void bvh_free(BVHNode *root);

// Traverse two BVHNode trees and call callback for leaf pairs whose AABBs overlap.
// callback receives (leafA, leafB, userdata).
void bvh_traverse_pairs(const BVHNode *a, const BVHNode *b, void (*callback)(void*, void*, void*), void *userdata);

// Convenience: self-traverse a single tree (pairs within same tree). Callback will be called with each pair (leafA, leafB) where leafA < leafB according to pointer address to avoid duplicates.
void bvh_self_traverse(const BVHNode *root, void (*callback)(void*, void*, void*), void *userdata);

#endif // BVH_H
