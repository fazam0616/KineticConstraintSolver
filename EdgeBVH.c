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

static const Constraint **g_edge_ptrs_for_sort = NULL;
static int compare_edge_centroid_qsort(const void *a, const void *b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    const Constraint *ea = g_edge_ptrs_for_sort[ia], *eb = g_edge_ptrs_for_sort[ib];
    float ca = (ea->node->pos[0] + ea->other->pos[0]) * 0.5f;
    float cb = (eb->node->pos[0] + eb->other->pos[0]) * 0.5f;
    return (ca > cb) - (ca < cb);
}

static BVHNode *build_node(const Constraint **edges, int *indices, size_t start, size_t end, float capsule_radius) {
    BVHNode *node = (BVHNode*)malloc(sizeof(BVHNode));
    aabb_init_empty(&node->bounds);
    node->is_leaf = 0;
    node->left = node->right = NULL;
    node->leaf_data = NULL;
    for (size_t i = start; i < end; ++i) {
        AABB edge_box;
        compute_edge_aabb(edges[indices[i]], capsule_radius, &edge_box);
        aabb_expand(&node->bounds, &edge_box);
    }
    size_t n = end - start;
    if (n == 1) {
        node->is_leaf = 1;
        int *leaf_idx = (int*)malloc(sizeof(int));
        *leaf_idx = indices[start];
        node->leaf_data = leaf_idx;
        return node;
    }
    g_edge_ptrs_for_sort = edges;
    qsort(indices+start, n, sizeof(int), compare_edge_centroid_qsort);
    g_edge_ptrs_for_sort = NULL;
    size_t mid = start + n/2;
    node->left = build_node(edges, indices, start, mid, capsule_radius);
    node->right = build_node(edges, indices, mid, end, capsule_radius);
    return node;
}

// Helper: refit node bounds (top-level to avoid nested functions)
void edge_refit_node(BVHNode *node, DynArray *edges, float capsule_radius) {
    if (!node) return;
    if (node->is_leaf) {
        int idx = *(int*)node->leaf_data;
        Constraint *e = (Constraint*)dynarray_get(edges, idx);
        compute_edge_aabb(e, capsule_radius, &node->bounds);
        return;
    }
    edge_refit_node(node->left, edges, capsule_radius);
    edge_refit_node(node->right, edges, capsule_radius);
    aabb_init_empty(&node->bounds);
    aabb_expand(&node->bounds, &node->left->bounds);
    aabb_expand(&node->bounds, &node->right->bounds);
}

// Helper: query traversal
void edge_query_node(const BVHNode *node, const AABB *query, DynArray *out_indices) {
    if (!node) return;
    int overlap = 1;
    for (int i = 0; i < 3; ++i) {
        if (node->bounds.max[i] < query->min[i] || node->bounds.min[i] > query->max[i]) { overlap = 0; break; }
    }
    if (!overlap) return;
    if (node->is_leaf) {
        int *idx = (int*)malloc(sizeof(int));
        *idx = *(int*)node->leaf_data;
        dynarray_append(out_indices, idx);
        return;
    }
    edge_query_node(node->left, query, out_indices);
    edge_query_node(node->right, query, out_indices);
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

void edge_bvh_refit(EdgeBVH *bvh, DynArray *edges, float capsule_radius) {
    if (!bvh || !bvh->root || !edges) return;
    // call external static helper
    extern void edge_refit_node(BVHNode *node, DynArray *edges, float capsule_radius);
    edge_refit_node(bvh->root, edges, capsule_radius);
}

void edge_bvh_query(const EdgeBVH *bvh, const AABB *query, DynArray *out_indices) {
    if (!bvh || !bvh->root || !query || !out_indices) return;
    extern void edge_query_node(const BVHNode *node, const AABB *query, DynArray *out_indices);
    edge_query_node(bvh->root, query, out_indices);
}

static void free_leaf_data_nodes(BVHNode *node) {
    if (!node) return;
    if (node->is_leaf) {
        if (node->leaf_data) free(node->leaf_data);
        node->leaf_data = NULL;
        return;
    }
    free_leaf_data_nodes(node->left);
    free_leaf_data_nodes(node->right);
}

void edge_bvh_free(EdgeBVH *bvh) {
    if (!bvh) return;
    free_leaf_data_nodes(bvh->root);
    bvh_free(bvh->root);
    free(bvh->edge_indices);
    free(bvh);
}

// Adapter to convert leaf_data (int*) to original callback signature
typedef struct { void (*cb)(int,int,void*); void *userdata; } EdgeAdapter;
static void edge_adapter_cb(void *la, void *lb, void *ud) {
    EdgeAdapter *ad = (EdgeAdapter*)ud;
    int ia = *(int*)la;
    int ib = *(int*)lb;
    if (ia < ib) ad->cb(ia, ib, ad->userdata);
}

void edge_bvh_self_traverse(const EdgeBVH *bvh, void (*callback)(int, int, void*), void *userdata) {
    if (!bvh || !bvh->root) return;
    EdgeAdapter ad = { callback, userdata };
    bvh_self_traverse(bvh->root, edge_adapter_cb, &ad);
}

void edge_bvh_debug_draw(const EdgeBVH *bvh, int depth_limit) {
    if (!bvh || !bvh->root) return;
    extern void bvh_debug_draw(const BVHNode *root, int depth_limit);
    bvh_debug_draw(bvh->root, depth_limit);
}
