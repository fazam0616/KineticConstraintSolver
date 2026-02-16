#include "Octree.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#define MAX_OCTREE_DEPTH 8
#define MAX_CHILD_COUNT 1

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
    node->node_entries = dynarray_create(16);
    node->triangle_indices = dynarray_create(8);
    node->constraint_indices = dynarray_create(8);
    
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
    if (node->node_entries) dynarray_free(node->node_entries, free);
    if (node->triangle_indices) dynarray_free(node->triangle_indices, NULL);
    if (node->constraint_indices) dynarray_free(node->constraint_indices, NULL);
    
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
        child->node_entries = dynarray_create(16);
        child->triangle_indices = dynarray_create(8);
        child->constraint_indices = dynarray_create(8);
        
        node->children[i] = child;
    }
    
    // Redistribute existing nodes to appropriate children
    if (node->node_entries) {
        for (size_t i = 0; i < dynarray_size(node->node_entries); ++i) {
            NodeEntry *entry = (NodeEntry*)dynarray_get(node->node_entries, i);
            if (entry) {
                int octant = octree_get_octant(center, entry->pos);
                if (node->children[octant]) {
                    // Create new entry for child
                    NodeEntry *new_entry = (NodeEntry*)malloc(sizeof(NodeEntry));
                    new_entry->node_idx = entry->node_idx;
                    memcpy(new_entry->pos, entry->pos, sizeof(float) * 3);
                    dynarray_append(node->children[octant]->node_entries, new_entry);
                }
            }
        }
        // Clear parent's node list since children now own them
        for (size_t i = 0; i < dynarray_size(node->node_entries); ++i) {
            free(dynarray_get(node->node_entries, i));
        }
        node->node_entries->size = 0;
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
        NodeEntry *entry = (NodeEntry*)malloc(sizeof(NodeEntry));
        entry->node_idx = node_idx;
        memcpy(entry->pos, pos, sizeof(float) * 3);
        dynarray_append(root->node_entries, entry);
        
        // Subdivide if we have too many nodes and haven't reached max depth
        if (dynarray_size(root->node_entries) > MAX_CHILD_COUNT && root->depth < MAX_OCTREE_DEPTH) {
            octree_subdivide(root);
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

// Query nodes in the leaf cell containing a point
DynArray* octree_query_nodes(OctreeNode *root, float pos[3]) {
    DynArray *results = dynarray_create(32);
    
    if (!root) return results;
    
    OctreeNode *current = root;
    
    // Traverse down to the leaf containing this point
    while (current) {
        // If leaf, return its node entries
        if (current->is_leaf) {
            if (current->node_entries) {
                for (size_t i = 0; i < dynarray_size(current->node_entries); ++i) {
                    NodeEntry *entry = (NodeEntry*)dynarray_get(current->node_entries, i);
                    if (entry) {
                        int *result_idx = (int*)malloc(sizeof(int));
                        *result_idx = entry->node_idx;
                        dynarray_append(results, result_idx);
                    }
                }
            }
            break;
        }
        
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

// Find the closest node to a given position within max_distance
int octree_find_closest_node(OctreeNode *root, Node **all_nodes, size_t num_nodes,
                             float pos[3], float max_distance) {
    DynArray *candidates = octree_query_nodes(root, pos);
    
    int closest_idx = -1;
    float closest_dist_sq = max_distance * max_distance;
    
    for (size_t i = 0; i < dynarray_size(candidates); ++i) {
        int *idx_ptr = (int*)dynarray_get(candidates, i);
        if (!idx_ptr || *idx_ptr < 0 || (size_t)*idx_ptr >= num_nodes) continue;
        
        Node *n = all_nodes[*idx_ptr];
        if (!n) continue;
        
        float dx = n->pos[0] - pos[0];
        float dy = n->pos[1] - pos[1];
        float dz = n->pos[2] - pos[2];
        float dist_sq = dx*dx + dy*dy + dz*dz;
        
        if (dist_sq < closest_dist_sq) {
            closest_dist_sq = dist_sq;
            closest_idx = *idx_ptr;
        }
    }
    
    dynarray_free(candidates, free);
    return closest_idx;
}

// Clear all indices from octree (for rebuilding)
void octree_clear(OctreeNode *root) {
    if (!root) return;
    
    // Clear this node's lists
    if (root->node_entries) {
        for (size_t i = 0; i < dynarray_size(root->node_entries); ++i) {
            free(dynarray_get(root->node_entries, i));
        }
        root->node_entries->size = 0;
    }
    
    if (root->triangle_indices) {
        for (size_t i = 0; i < dynarray_size(root->triangle_indices); ++i) {
            free(dynarray_get(root->triangle_indices, i));
        }
        root->triangle_indices->size = 0;
    }
    
    if (root->constraint_indices) {
        for (size_t i = 0; i < dynarray_size(root->constraint_indices); ++i) {
            free(dynarray_get(root->constraint_indices, i));
        }
        root->constraint_indices->size = 0;
    }
    
    // Recursively clear children but preserve tree structure
    for (int i = 0; i < 8; ++i) {
        if (root->children[i]) {
            octree_clear(root->children[i]);
        }
    }
}

// Insert a constraint into the ancestor node containing both endpoints
void octree_insert_constraint(OctreeNode *root, int constraint_idx,
                             float pos_a[3], float pos_b[3]) {
    if (!root) return;
    
    // Check if both points are in bounds
    if (!aabb_contains_point(&root->bounds, pos_a) || 
        !aabb_contains_point(&root->bounds, pos_b)) {
        return;
    }
    
    // If leaf, store here
    if (root->is_leaf) {
        int *idx = (int*)malloc(sizeof(int));
        *idx = constraint_idx;
        dynarray_append(root->constraint_indices, idx);
        // NOTE: Do NOT trigger subdivision for constraints because we cannot
        // redistribute them (no position data stored). Constraints are inserted
        // after nodes/triangles have already established the tree structure.
        return;
    }
    
    // Find which child contains both points (if any)
    float center[3] = {
        (root->bounds.min[0] + root->bounds.max[0]) * 0.5f,
        (root->bounds.min[1] + root->bounds.max[1]) * 0.5f,
        (root->bounds.min[2] + root->bounds.max[2]) * 0.5f
    };
    
    int octant_a = octree_get_octant(center, pos_a);
    int octant_b = octree_get_octant(center, pos_b);
    
    // If both endpoints in same octant, recurse
    if (octant_a == octant_b && root->children[octant_a]) {
        octree_insert_constraint(root->children[octant_a], constraint_idx, pos_a, pos_b);
    } else {
        // Endpoints in different octants, store at this level
        int *idx = (int*)malloc(sizeof(int));
        *idx = constraint_idx;
        dynarray_append(root->constraint_indices, idx);
    }
}

// Helper: recursively collect constraints from node and all children
static void collect_constraints_recursive(OctreeNode *node, DynArray *results) {
    if (!node) return;
    
    // Add constraints at this level
    if (node->constraint_indices) {
        for (size_t i = 0; i < dynarray_size(node->constraint_indices); ++i) {
            int *idx_ptr = (int*)dynarray_get(node->constraint_indices, i);
            if (idx_ptr) {
                int *copy = (int*)malloc(sizeof(int));
                *copy = *idx_ptr;
                dynarray_append(results, copy);
            }
        }
    }
    
    // Recurse to all children
    if (!node->is_leaf) {
        for (int i = 0; i < 8; ++i) {
            if (node->children[i]) {
                collect_constraints_recursive(node->children[i], results);
            }
        }
    }
}

// Query constraints in region containing both points
DynArray* octree_query_constraints(OctreeNode *root, float pos_a[3], float pos_b[3]) {
    DynArray *result = dynarray_create(16);
    if (!root) return result;
    
    // Check if both points are in bounds
    if (!aabb_contains_point(&root->bounds, pos_a) || 
        !aabb_contains_point(&root->bounds, pos_b)) {
        return result;
    }
    
    // Traverse down to find the ancestor node
    OctreeNode *current = root;
    while (!current->is_leaf) {
        float center[3] = {
            (current->bounds.min[0] + current->bounds.max[0]) * 0.5f,
            (current->bounds.min[1] + current->bounds.max[1]) * 0.5f,
            (current->bounds.min[2] + current->bounds.max[2]) * 0.5f
        };
        
        int octant_a = octree_get_octant(center, pos_a);
        int octant_b = octree_get_octant(center, pos_b);
        
        // If in different octants, this is the common ancestor
        if (octant_a != octant_b) {
            break;
        }
        
        // Otherwise descend to the common child
        if (current->children[octant_a]) {
            current = current->children[octant_a];
        } else {
            break;
        }
    }
    
    // Collect constraints from this node AND all its children recursively
    collect_constraints_recursive(current, result);
    
    return result;
}

