#ifndef CONSTRAINT_H
#define CONSTRAINT_H

#include <stddef.h>
#include "Node.h"
#include "datastructures.h"

typedef struct Constraint Constraint;

/* CSparse is required for triplet emitters; include cs.h so the cs type is
    available to code that uses the Constraint API. The Makefile should set
    CSPARSE_INC appropriately so the header is found. */
#include <cs.h>

struct Constraint {
    Node *node;
    // public type and parameters
    enum { CT_DIST = 1, CT_SPRING = 2, CT_ANCHOR = 3 } type;
    float stiffness; // used for springs (and softening)
    // For convenience expose the other endpoint and rest length publicly so
    // simulators can treat springs as force-objects instead of constraints.
    Node *other;
    float rest_length;
    // virtual methods
    float (*err)(Constraint *self);
    // fill out dense jacobian row into provided buffer of size (n_nodes*2)
    void (*dc_node)(Constraint *self, Node **nodes, size_t n_nodes, float *out_row);
    // optional sparse representation: fill DynArray of pairs (node_idx, vec[2])
    void (*dc_sparse)(Constraint *self, Node **nodes, size_t n_nodes, DynArray *out);
    // optional: append entries directly into a sparse triplet structure.
    // The concrete triplet type varies (CSparse triplet or CHOLMOD triplet).
    // Use a generic void* here; implementations should cast to the expected
    // triplet type internally. If you don't use sparse triplet emission, set
    // this to NULL.
    void (*dc_triplet)(Constraint *self, void *T, int row);
    void (*draw)(Constraint *self);
};

// DistConstraint and derived types
Constraint* distconstraint_create(Node *node, Node *other, float distance);

Constraint* springconstraint_create(Node *node, Node *other, float stiffness, float distance);

Constraint* anchorconstraint_create(Node *node, float x, float y, float z);

// Triangle wall (3D boundary)
typedef struct TriangleWall {
    Node *A;
    Node *B;
    Node *C;
    float restitution;
    float friction;
    Constraint* edges[3];
} TriangleWall;

TriangleWall* trianglewall_create(Node *A, Node *B, Node *C, float restitution, float friction, Constraint* edge_AB, Constraint* edge_BC, Constraint* edge_CA);
void trianglewall_free(TriangleWall *w);

// Refresh internal cached node indices inside constraint implementations.
// Call this after node indices change (for example after deleting nodes).
void constraint_refresh_indices(DynArray *constraints);

#endif // CONSTRAINT_H
