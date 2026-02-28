#include "BVH.h"
#include <stdlib.h>

static int aabb_overlap(const AABB *a, const AABB *b) {
    for (int i = 0; i < 3; ++i) {
        if (a->max[i] < b->min[i] || a->min[i] > b->max[i]) return 0;
    }
    return 1;
}

void bvh_free(BVHNode *root) {
    if (!root) return;
    if (!root->is_leaf) {
        bvh_free(root->left);
        bvh_free(root->right);
    }
    free(root);
}

static void traverse_pairs_internal(const BVHNode *a, const BVHNode *b, void (*callback)(void*, void*, void*), void *userdata) {
    if (!a || !b) return;
    if (!aabb_overlap(&a->bounds, &b->bounds)) return;
    if (a->is_leaf && b->is_leaf) {
        // avoid duplicate or self-pair
        if (a != b) callback(a->leaf_data, b->leaf_data, userdata);
        return;
    }
    if (a->is_leaf) {
        traverse_pairs_internal(a, b->left, callback, userdata);
        traverse_pairs_internal(a, b->right, callback, userdata);
    } else if (b->is_leaf) {
        traverse_pairs_internal(a->left, b, callback, userdata);
        traverse_pairs_internal(a->right, b, callback, userdata);
    } else {
        traverse_pairs_internal(a->left, b->left, callback, userdata);
        traverse_pairs_internal(a->left, b->right, callback, userdata);
        traverse_pairs_internal(a->right, b->left, callback, userdata);
        traverse_pairs_internal(a->right, b->right, callback, userdata);
    }
}

void bvh_traverse_pairs(const BVHNode *a, const BVHNode *b, void (*callback)(void*, void*, void*), void *userdata) {
    if (!a || !b) return;
    traverse_pairs_internal(a, b, callback, userdata);
}

// For self-traverse, call traverse_pairs_internal on root vs root but ensure each pair called once.
static void self_traverse_internal(const BVHNode *a, const BVHNode *b, void (*callback)(void*, void*, void*), void *userdata) {
    if (!a || !b) return;
    if (!aabb_overlap(&a->bounds, &b->bounds)) return;
    if (a->is_leaf && b->is_leaf) {
        // only call when leaf pointer ordering to avoid duplicates or self-pair
        if (a < b) callback(a->leaf_data, b->leaf_data, userdata);
        return;
    }
    if (a->is_leaf) {
        self_traverse_internal(a, b->left, callback, userdata);
        self_traverse_internal(a, b->right, callback, userdata);
    } else if (b->is_leaf) {
        self_traverse_internal(a->left, b, callback, userdata);
        self_traverse_internal(a->right, b, callback, userdata);
    } else {
        self_traverse_internal(a->left, b->left, callback, userdata);
        self_traverse_internal(a->left, b->right, callback, userdata);
        self_traverse_internal(a->right, b->left, callback, userdata);
        self_traverse_internal(a->right, b->right, callback, userdata);
    }
}

void bvh_self_traverse(const BVHNode *root, void (*callback)(void*, void*, void*), void *userdata) {
    if (!root) return;
    self_traverse_internal(root, root, callback, userdata);
}
