#include "Octree.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define MAX_OCTREE_DEPTH 8

// Helper: get which octant (0-7) a point belongs to
int octree_get_octant(const float center[3], const float point[3]) {
    int oct = 0;
    if (point[0] >= center[0]) oct |= 1;
    if (point[1] >= center[1]) oct |= 2;
    if (point[2] >= center[2]) oct |= 4;
    return oct;
}

// Helper: check if AABB contains point
int aabb_contains_point(const AABB *box, const float point[3]) {
    return point[0] >= box->min[0] && point[0] <= box->max[0] &&
           point[1] >= box->min[1] && point[1] <= box->max[1] &&
           point[2] >= box->min[2] && point[2] <= box->max[2];
}

// Helper: compute AABB from three points
AABB aabb_from_triangle(const float a[3], const float b[3], const float c[3]) {
    AABB box;
    box.min[0] = fminf(fminf(a[0], b[0]), c[0]);
    box.min[1] = fminf(fminf(a[1], b[1]), c[1]);
    box.min[2] = fminf(fminf(a[2], b[2]), c[2]);
    box.max[0] = fmaxf(fmaxf(a[0], b[0]), c[0]);
    box.max[1] = fmaxf(fmaxf(a[1], b[1]), c[1]);
    box.max[2] = fmaxf(fmaxf(a[2], b[2]), c[2]);
    return box;
}

// Helper: check if two AABBs intersect
int aabb_intersects(const AABB *a, const AABB *b) {
    return !(a->max[0] < b->min[0] || a->min[0] > b->max[0] ||
             a->max[1] < b->min[1] || a->min[1] > b->max[1] ||
             a->max[2] < b->min[2] || a->min[2] > b->max[2]);
}

// Create octree node
OctreeNode* octree_create(float min_x, float min_y, float min_z,
                          float max_x, float max_y, float max_z, int max_depth) {
    OctreeNode *node = (OctreeNode*)malloc(sizeof(OctreeNode));
    memset(node, 0, sizeof(OctreeNode));
    
    node->bounds.min[0] = min_x;
    node->bounds.min[1] = min_y;
    node->bounds.min[2] = min_z;
    node->bounds.max[0] = max_x;
    node->bounds.max[1] = max_y;
    node->bounds.max[2] = max_z;
    
    node->depth = 0;
    node->is_leaf = 1;
    node->node_indices = dynarray_create(16);
    node->triangle_indices = dynarray_create(8);
    
    // Initialize children to NULL
    for (int i = 0; i < 8; ++i) {
        node->children[i] = NULL;
    }
    
    return node;
}

// Recursively free octree
void octree_free(OctreeNode *node) {
    if (!node) return;
    
    // Free children first
    for (int i = 0; i < 8; ++i) {
        if (node->children[i]) {
            octree_free(node->children[i]);
        }
    }
    
    // Free dynamic arrays (elements are just ints, no need to free individually)
    if (node->node_indices) dynarray_free(node->node_indices, NULL);
    if (node->triangle_indices) dynarray_free(node->triangle_indices, NULL);
    
    free(node);
}

// Subdivide an octree node into 8 children
static void octree_subdivide(OctreeNode *node) {
    if (!node->is_leaf) return; // Already subdivided
    
    float *min = node->bounds.min;
    float *max = node->bounds.max;
    float center[3] = {
        (min[0] + max[0]) * 0.5f,
        (min[1] + max[1]) * 0.5f,
        (min[2] + max[2]) * 0.5f
    };
    
    // Create 8 children
    for (int i = 0; i < 8; ++i) {
        OctreeNode *child = (OctreeNode*)malloc(sizeof(OctreeNode));
        memset(child, 0, sizeof(OctreeNode));
        
        // Determine bounds for this octant
        float child_min[3], child_max[3];
        child_min[0] = (i & 1) ? center[0] : min[0];
        child_max[0] = (i & 1) ? max[0] : center[0];
        child_min[1] = (i & 2) ? center[1] : min[1];
        child_max[1] = (i & 2) ? max[1] : center[1];
        child_min[2] = (i & 4) ? center[2] : min[2];
        child_max[2] = (i & 4) ? max[2] : center[2];
        
        memcpy(child->bounds.min, child_min, sizeof(float) * 3);
        memcpy(child->bounds.max, child_max, sizeof(float) * 3);
        
        child->depth = node->depth + 1;
        child->is_leaf = 1;
        child->node_indices = dynarray_create(16);
        child->triangle_indices = dynarray_create(8);
        
        node->children[i] = child;
    }
    
    node->is_leaf = 0;
}

// Insert a node into the octree
void octree_insert_node(OctreeNode *root, int node_idx, float pos[3]) {
    if (!root) return;
    
    // Check if point is in bounds
    if (!aabb_contains_point(&root->bounds, pos)) {
        return; // Outside octree bounds
    }
    
    // If leaf, add to this node
    if (root->is_leaf) {
        int *idx_ptr = (int*)malloc(sizeof(int));
        *idx_ptr = node_idx;
        dynarray_append(root->node_indices, idx_ptr);
        
        // Subdivide if we have too many nodes and haven't reached max depth
        if (dynarray_size(root->node_indices) > 8 && root->depth < MAX_OCTREE_DEPTH) {
            octree_subdivide(root);
            
            // Redistribute existing nodes to children
            for (size_t i = 0; i < dynarray_size(root->node_indices); ++i) {
                int *stored_idx = (int*)dynarray_get(root->node_indices, i);
                // We don't have the position anymore, so we can't redistribute
                // For simplicity, keep them here
            }
        }
        return;
    }
    
    // Not a leaf - find appropriate child
    float center[3] = {
        (root->bounds.min[0] + root->bounds.max[0]) * 0.5f,
        (root->bounds.min[1] + root->bounds.max[1]) * 0.5f,
        (root->bounds.min[2] + root->bounds.max[2]) * 0.5f
    };
    
    int octant = octree_get_octant(center, pos);
    if (root->children[octant]) {
        octree_insert_node(root->children[octant], node_idx, pos);
    }
}

// Find the minimal common ancestor node that contains all three points
static OctreeNode* find_common_ancestor(OctreeNode *root, const float a[3], 
                                       const float b[3], const float c[3]) {
    if (!root) return NULL;
    
    // Check if all three points are in this node's bounds
    AABB tri_box = aabb_from_triangle(a, b, c);
    if (!aabb_intersects(&root->bounds, &tri_box)) {
        return NULL; // Triangle not in this subtree
    }
    
    // If leaf, this is the answer
    if (root->is_leaf) {
        return root;
    }
    
    // Check which children contain the triangle
    float center[3] = {
        (root->bounds.min[0] + root->bounds.max[0]) * 0.5f,
        (root->bounds.min[1] + root->bounds.max[1]) * 0.5f,
        (root->bounds.min[2] + root->bounds.max[2]) * 0.5f
    };
    
    int octant_a = octree_get_octant(center, a);
    int octant_b = octree_get_octant(center, b);
    int octant_c = octree_get_octant(center, c);
    
    // If all three points are in the same octant, recurse
    if (octant_a == octant_b && octant_b == octant_c) {
        if (root->children[octant_a]) {
            OctreeNode *child_result = find_common_ancestor(root->children[octant_a], a, b, c);
            if (child_result) return child_result;
        }
    }
    
    // Otherwise, this node is the minimal common ancestor
    return root;
}

// Insert triangle into octree at minimal common ancestor
int octree_insert_triangle(OctreeNode *root, int tri_idx,
                          float a[3], float b[3], float c[3]) {
    if (!root) return -1;
    
    OctreeNode *target = find_common_ancestor(root, a, b, c);
    if (!target) return -1;
    
    int *idx_ptr = (int*)malloc(sizeof(int));
    *idx_ptr = tri_idx;
    dynarray_append(target->triangle_indices, idx_ptr);
    
    return target->depth;
}

// Query triangles that could collide with a point
// Returns all triangles stored at the leaf containing the point and all ancestors
DynArray* octree_query_triangles(OctreeNode *root, float pos[3]) {
    DynArray *results = dynarray_create(32);
    
    if (!root) return results;
    
    OctreeNode *current = root;
    
    // Traverse down to the leaf containing this point, collecting triangles along the way
    while (current) {
        // Add all triangles at this level
        if (current->triangle_indices) {
            for (size_t i = 0; i < dynarray_size(current->triangle_indices); ++i) {
                int *tri_idx = (int*)dynarray_get(current->triangle_indices, i);
                if (tri_idx) {
                    int *result_idx = (int*)malloc(sizeof(int));
                    *result_idx = *tri_idx;
                    dynarray_append(results, result_idx);
                }
            }
        }
        
        // If leaf, stop
        if (current->is_leaf) break;
        
        // Find which child contains the point
        if (!aabb_contains_point(&current->bounds, pos)) break;
        
        float center[3] = {
            (current->bounds.min[0] + current->bounds.max[0]) * 0.5f,
            (current->bounds.min[1] + current->bounds.max[1]) * 0.5f,
            (current->bounds.min[2] + current->bounds.max[2]) * 0.5f
        };
        
        int octant = octree_get_octant(center, pos);
        current = current->children[octant];
    }
    
    return results;
}

// Clear all indices from octree (for rebuilding)
void octree_clear(OctreeNode *root) {
    if (!root) return;
    
    // Clear this node's lists
    if (root->node_indices) {
        for (size_t i = 0; i < dynarray_size(root->node_indices); ++i) {
            free(dynarray_get(root->node_indices, i));
        }
        root->node_indices->size = 0;
    }
    
    if (root->triangle_indices) {
        for (size_t i = 0; i < dynarray_size(root->triangle_indices); ++i) {
            free(dynarray_get(root->triangle_indices, i));
        }
        root->triangle_indices->size = 0;
    }
    
    // Recursively clear children
    for (int i = 0; i < 8; ++i) {
        if (root->children[i]) {
            octree_clear(root->children[i]);
        }
    }
}
