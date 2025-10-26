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
    // optional: append entries directly into a CSparse triplet (cs_spalloc triplet)
    // signature: (self, T, row) where T is a cs* in triplet form and row is the
    // constraint row index (0..m-1). If you don't use CSparse, set to NULL.
    void (*dc_triplet)(Constraint *self, struct cs *T, int row);
    void (*draw)(Constraint *self);
};

// DistConstraint and derived types
Constraint* distconstraint_create(Node *node, Node *other, float distance);

Constraint* springconstraint_create(Node *node, Node *other, float stiffness, float distance);

Constraint* anchorconstraint_create(Node *node, float x, float y);

// Wall segment
typedef struct WallSegment {
    Node *A;
    Node *B;
    float restitution;
    float friction;
} WallSegment;

WallSegment* wallsegment_create(Node *A, Node *B, float restitution, float friction);
void wallsegment_free(WallSegment *w);

#endif // CONSTRAINT_H
