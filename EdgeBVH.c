#include "EdgeBVH.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>

static void compute_edge_aabb(const Constraint *e, float capsule_radius, AABB *out_box) {
    const float *a = e->node->pos;
    const float *b = e->other->pos;
    for (int i = 0; i < 3; ++i) {
        float minv = (a[i] < b[i] ? a[i] : b[i]) - capsule_radius;
        float maxv = (a[i] > b[i] ? a[i] : b[i]) + capsule_radius;
        out_box->min[i] = minv;
        out_box->max[i] = maxv;
    }
}

static void aabb_expand(AABB *a, const AABB *b) {
    for (int i = 0; i < 3; ++i) {
        if (b->min[i] < a->min[i]) a->min[i] = b->min[i];
        if (b->max[i] > a->max[i]) a->max[i] = b->max[i];
    }
}

static void aabb_init_empty(AABB *a) {
    for (int i = 0; i < 3; ++i) {
        a->min[i] = FLT_MAX;
        a->max[i] = -FLT_MAX;
    }
}

static int axis_of_max_extent(const AABB *a) {
    float ext[3] = {a->max[0]-a->min[0], a->max[1]-a->min[1], a->max[2]-a->min[2]};
    int axis = 0;
    if (ext[1] > ext[axis]) axis = 1;
    if (ext[2] > ext[axis]) axis = 2;
    return axis;
}

static int compare_edge_centroid(const void *a, const void *b, void *ctx) {
    const Constraint **edges = (const Constraint**)ctx;
    int ia = *(const int*)a, ib = *(const int*)b;
    const Constraint *ea = edges[ia], *eb = edges[ib];
    float ca = (ea->node->pos[0] + ea->other->pos[0]) * 0.5f;
    float cb = (eb->node->pos[0] + eb->other->pos[0]) * 0.5f;
    return (ca > cb) - (ca < cb);
}

static EdgeBVHNode *build_node(const Constraint **edges, int *indices, size_t start, size_t end, float capsule_radius) {
    EdgeBVHNode *node = (EdgeBVHNode*)malloc(sizeof(EdgeBVHNode));
    aabb_init_empty(&node->bounds);
    for (size_t i = start; i < end; ++i) {
        AABB edge_box;
        compute_edge_aabb(edges[indices[i]], capsule_radius, &edge_box);
        aabb_expand(&node->bounds, &edge_box);
    }
    size_t n = end - start;
    if (n == 1) {
        node->is_leaf = 1;
        node->data.leaf.edge_idx = indices[start];
        return node;
    }
    int axis = axis_of_max_extent(&node->bounds);
    qsort_r(indices+start, n, sizeof(int), compare_edge_centroid, (void*)edges);
    size_t mid = start + n/2;
    node->is_leaf = 0;
    node->data.children.left = build_node(edges, indices, start, mid, capsule_radius);
    node->data.children.right = build_node(edges, indices, mid, end, capsule_radius);
    return node;
}

EdgeBVH *edge_bvh_build(DynArray *edges, float capsule_radius) {
    size_t n = dynarray_size(edges);
    if (n == 0) return NULL;
    EdgeBVH *bvh = (EdgeBVH*)malloc(sizeof(EdgeBVH));
    bvh->n_edges = n;
    bvh->edge_indices = (int*)malloc(sizeof(int)*n);
    const Constraint **edge_ptrs = (const Constraint**)malloc(sizeof(Constraint*)*n);
    for (size_t i = 0; i < n; ++i) {
        bvh->edge_indices[i] = (int)i;
        edge_ptrs[i] = (Constraint*)dynarray_get(edges, i);
    }
    bvh->root = build_node(edge_ptrs, bvh->edge_indices, 0, n, capsule_radius);
    free(edge_ptrs);
    return bvh;
}

void free_edge_node(EdgeBVHNode *node) {
    if (!node) return;
    if (!node->is_leaf) {
        free_edge_node(node->data.children.left);
        free_edge_node(node->data.children.right);
    }
    free(node);
}

void edge_bvh_free(EdgeBVH *bvh) {
    if (!bvh) return;
    // Helper: recursively free nodes
    free_edge_node(bvh->root);
    free(bvh->edge_indices);
    free(bvh);
}

void traverse(const EdgeBVHNode *a, const EdgeBVHNode *b, void (*callback)(int, int, void*), void *userdata) {
    // Prune if AABBs do not overlap
    int overlap = 1;
    for (int i = 0; i < 3; ++i) {
        if (a->bounds.max[i] < b->bounds.min[i] || a->bounds.min[i] > b->bounds.max[i]) {
            overlap = 0;
            break;
        }
    }
    if (!overlap) return;
    if (a->is_leaf && b->is_leaf) {
        int idxA = a->data.leaf.edge_idx;
        int idxB = b->data.leaf.edge_idx;
        if (idxA < idxB) callback(idxA, idxB, userdata);
        return;
    }
    if (a->is_leaf) {
        traverse(a, b->data.children.left, callback, userdata);
        traverse(a, b->data.children.right, callback, userdata);
    } else if (b->is_leaf) {
        traverse(a->data.children.left, b, callback, userdata);
        traverse(a->data.children.right, b, callback, userdata);
    } else {
        traverse(a->data.children.left, b->data.children.left, callback, userdata);
        traverse(a->data.children.left, b->data.children.right, callback, userdata);
        traverse(a->data.children.right, b->data.children.left, callback, userdata);
        traverse(a->data.children.right, b->data.children.right, callback, userdata);
    }
}

void edge_bvh_self_traverse(const EdgeBVH *bvh, void (*callback)(int, int, void*), void *userdata) {
    if (!bvh || !bvh->root) return;
    // Helper: recursively traverse pairs
    
    traverse(bvh->root, bvh->root, callback, userdata);
}

void edge_bvh_debug_draw(const EdgeBVH *bvh, int depth_limit) {
    // ...draw AABBs for debugging...
}
