#include "TriangleBVH.h"
#include "Constraint.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>

static void compute_triangle_aabb(const TriangleWall *w, AABB *out_box) {
    const float *a = w->A->pos;
    const float *b = w->B->pos;
    const float *c = w->C->pos;
    for (int i = 0; i < 3; ++i) {
        float minv = a[i];
        if (b[i] < minv) minv = b[i];
        if (c[i] < minv) minv = c[i];
        float maxv = a[i];
        if (b[i] > maxv) maxv = b[i];
        if (c[i] > maxv) maxv = c[i];
        out_box->min[i] = minv;
        out_box->max[i] = maxv;
    }
}
// Median split on largest axis
typedef struct {
    int *indices;
    size_t count;
} IndexList;

static const TriangleWall **g_wall_ptrs_for_sort = NULL;
static int compare_tri_centroid_qsort(const void *a, const void *b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    const TriangleWall *wa = g_wall_ptrs_for_sort[ia], *wb = g_wall_ptrs_for_sort[ib];
    float ca = (wa->A->pos[0] + wa->B->pos[0] + wa->C->pos[0]) / 3.0f;
    float cb = (wb->A->pos[0] + wb->B->pos[0] + wb->C->pos[0]) / 3.0f;
    return (ca > cb) - (ca < cb);
}

static BVHNode *build_node(const TriangleWall **walls, int *indices, size_t start, size_t end) {
    BVHNode *node = (BVHNode*)malloc(sizeof(BVHNode));
    aabb_init_empty(&node->bounds);
    node->is_leaf = 0;
    node->left = node->right = NULL;
    node->leaf_data = NULL;
    for (size_t i = start; i < end; ++i) {
        AABB tri_box;
        compute_triangle_aabb(walls[indices[i]], &tri_box);
        aabb_expand(&node->bounds, &tri_box);
    }
    size_t n = end - start;
    if (n == 1) {
        node->is_leaf = 1;
        int *leaf_idx = (int*)malloc(sizeof(int));
        *leaf_idx = indices[start];
        node->leaf_data = leaf_idx;
        return node;
    }
    g_wall_ptrs_for_sort = walls;
    qsort(indices+start, n, sizeof(int), compare_tri_centroid_qsort);
    g_wall_ptrs_for_sort = NULL;
    size_t mid = start + n/2;
    node->left = build_node(walls, indices, start, mid);
    node->right = build_node(walls, indices, mid, end);
    return node;
}

// Helper: refit node
void tri_refit_node(BVHNode *node, DynArray *walls) {
    if (!node) return;
    if (node->is_leaf) {
        int idx = *(int*)node->leaf_data;
        TriangleWall *w = (TriangleWall*)dynarray_get(walls, idx);
        compute_triangle_aabb(w, &node->bounds);
        return;
    }
    tri_refit_node(node->left, walls);
    tri_refit_node(node->right, walls);
    aabb_init_empty(&node->bounds);
    aabb_expand(&node->bounds, &node->left->bounds);
    aabb_expand(&node->bounds, &node->right->bounds);
}

// Helper: query traversal
void tri_query_node(const BVHNode *node, const AABB *query, DynArray *out_indices) {
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
    tri_query_node(node->left, query, out_indices);
    tri_query_node(node->right, query, out_indices);
}

TriangleBVH *triangle_bvh_build(DynArray *walls) {
    size_t n = dynarray_size(walls);
    if (n == 0) return NULL;
    TriangleBVH *bvh = (TriangleBVH*)malloc(sizeof(TriangleBVH));
    bvh->n_triangles = n;
    bvh->triangle_indices = (int*)malloc(sizeof(int)*n);
    const TriangleWall **wall_ptrs = (const TriangleWall**)malloc(sizeof(TriangleWall*)*n);
    for (size_t i = 0; i < n; ++i) {
        bvh->triangle_indices[i] = (int)i;
        wall_ptrs[i] = (TriangleWall*)dynarray_get(walls, i);
    }
    bvh->root = build_node(wall_ptrs, bvh->triangle_indices, 0, n);
    free(wall_ptrs);
    return bvh;
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

void triangle_bvh_free(TriangleBVH *bvh) {
    if (!bvh) return;
    free_leaf_data_nodes(bvh->root);
    bvh_free(bvh->root);
    free(bvh->triangle_indices);
    free(bvh);
}

void triangle_bvh_refit(TriangleBVH *bvh, DynArray *walls) {
    if (!bvh || !bvh->root || !walls) return;
    extern void tri_refit_node(BVHNode *node, DynArray *walls);
    tri_refit_node(bvh->root, walls);
}

void triangle_bvh_query(const TriangleBVH *bvh, const AABB *query, DynArray *out_indices) {
    if (!bvh || !bvh->root || !query || !out_indices) return;
    extern void tri_query_node(const BVHNode *node, const AABB *query, DynArray *out_indices);
    tri_query_node(bvh->root, query, out_indices);
}

void triangle_bvh_debug_draw(const TriangleBVH *bvh, int depth_limit) {
    // ...draw AABBs for debugging...
}
