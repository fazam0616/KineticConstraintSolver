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

// Median split on largest axis
typedef struct {
    int *indices;
    size_t count;
} IndexList;

static int axis_of_max_extent(const AABB *a) {
    float ext[3] = {a->max[0]-a->min[0], a->max[1]-a->min[1], a->max[2]-a->min[2]};
    int axis = 0;
    if (ext[1] > ext[axis]) axis = 1;
    if (ext[2] > ext[axis]) axis = 2;
    return axis;
}

static int compare_tri_centroid(const void *a, const void *b, void *ctx) {
    const TriangleWall **walls = (const TriangleWall**)ctx;
    int ia = *(const int*)a, ib = *(const int*)b;
    const TriangleWall *wa = walls[ia], *wb = walls[ib];
    float ca = (wa->A->pos[0] + wa->B->pos[0] + wa->C->pos[0]) / 3.0f;
    float cb = (wb->A->pos[0] + wb->B->pos[0] + wb->C->pos[0]) / 3.0f;
    return (ca > cb) - (ca < cb);
}

static TriangleBVHNode *build_node(const TriangleWall **walls, int *indices, size_t start, size_t end) {
    TriangleBVHNode *node = (TriangleBVHNode*)malloc(sizeof(TriangleBVHNode));
    aabb_init_empty(&node->bounds);
    for (size_t i = start; i < end; ++i) {
        AABB tri_box;
        compute_triangle_aabb(walls[indices[i]], &tri_box);
        aabb_expand(&node->bounds, &tri_box);
    }
    size_t n = end - start;
    if (n == 1) {
        node->is_leaf = 1;
        node->data.leaf.tri_idx = indices[start];
        return node;
    }
    // Median split
    int axis = axis_of_max_extent(&node->bounds);
    // Sort indices by centroid along axis
    qsort_r(indices+start, n, sizeof(int), compare_tri_centroid, (void*)walls);
    size_t mid = start + n/2;
    node->is_leaf = 0;
    node->data.children.left = build_node(walls, indices, start, mid);
    node->data.children.right = build_node(walls, indices, mid, end);
    return node;
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

void free_node(TriangleBVHNode *node) {
    if (!node) return;
    if (!node->is_leaf) {
        free_node(node->data.children.left);
        free_node(node->data.children.right);
    }
    free(node);
}

void triangle_bvh_free(TriangleBVH *bvh) {
    if (!bvh) return;
    
    free_node(bvh->root);
    free(bvh->triangle_indices);
    free(bvh);
}

void triangle_bvh_refit(TriangleBVH *bvh, DynArray *walls) {
    if (!bvh || !bvh->root || !walls) return;
    void refit_node(TriangleBVHNode *node) {
        if (node->is_leaf) {
            TriangleWall *w = (TriangleWall*)dynarray_get(walls, node->data.leaf.tri_idx);
            compute_triangle_aabb(w, &node->bounds);
            return;
        }
        refit_node(node->data.children.left);
        refit_node(node->data.children.right);
        aabb_init_empty(&node->bounds);
        aabb_expand(&node->bounds, &node->data.children.left->bounds);
        aabb_expand(&node->bounds, &node->data.children.right->bounds);
    }
    refit_node(bvh->root);
}

void triangle_bvh_query(const TriangleBVH *bvh, const AABB *query, DynArray *out_indices) {
    if (!bvh || !bvh->root || !query || !out_indices) return;
    void query_node(const TriangleBVHNode *node) {
        // Prune if AABBs do not overlap
        int overlap = 1;
        for (int i = 0; i < 3; ++i) {
            if (node->bounds.max[i] < query->min[i] || node->bounds.min[i] > query->max[i]) {
                overlap = 0;
                break;
            }
        }
        if (!overlap) return;
        if (node->is_leaf) {
            int *idx = (int*)malloc(sizeof(int));
            *idx = node->data.leaf.tri_idx;
            dynarray_append(out_indices, idx);
            return;
        }
        query_node(node->data.children.left);
        query_node(node->data.children.right);
    }
    query_node(bvh->root);
}

void triangle_bvh_debug_draw(const TriangleBVH *bvh, int depth_limit) {
    // ...draw AABBs for debugging...
}
