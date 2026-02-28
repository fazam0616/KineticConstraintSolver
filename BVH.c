#include "BVH.h"
#include <stdlib.h>
// OpenGL immediate-mode calls for debug drawing
#include <GL/gl.h>

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

// Helper: draw an axis-aligned bounding box as wireframe
static void draw_aabb_wire(const AABB *b) {
    // 8 corners
    float x0 = b->min[0], y0 = b->min[1], z0 = b->min[2];
    float x1 = b->max[0], y1 = b->max[1], z1 = b->max[2];
    // bottom face
    glBegin(GL_LINE_LOOP);
    glVertex3f(x0,y0,z0);
    glVertex3f(x1,y0,z0);
    glVertex3f(x1,y0,z1);
    glVertex3f(x0,y0,z1);
    glEnd();
    // top face
    glBegin(GL_LINE_LOOP);
    glVertex3f(x0,y1,z0);
    glVertex3f(x1,y1,z0);
    glVertex3f(x1,y1,z1);
    glVertex3f(x0,y1,z1);
    glEnd();
    // vertical edges
    glBegin(GL_LINES);
    glVertex3f(x0,y0,z0); glVertex3f(x0,y1,z0);
    glVertex3f(x1,y0,z0); glVertex3f(x1,y1,z0);
    glVertex3f(x1,y0,z1); glVertex3f(x1,y1,z1);
    glVertex3f(x0,y0,z1); glVertex3f(x0,y1,z1);
    glEnd();
}

// Recursive BVH draw
static void bvh_draw_node_recursive(const BVHNode *node, int depth_limit) {
    if (!node) return;
    draw_aabb_wire(&node->bounds);
    if (depth_limit == 0) return;
    if (!node->is_leaf) {
        int next = (depth_limit > 0) ? depth_limit - 1 : depth_limit;
        bvh_draw_node_recursive(node->left, next);
        bvh_draw_node_recursive(node->right, next);
    }
}

void bvh_debug_draw(const BVHNode *root, int depth_limit) {
    if (!root) return;
    // GL state: assume caller sets color/line width as desired. Use a thin line by default.
    glLineWidth(1.0f);
    bvh_draw_node_recursive(root, depth_limit);
}
