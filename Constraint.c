#include "Constraint.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <GL/gl.h>
#include <cs.h>

// Note about coordinate convention:
// The simulation uses a right-handed 2D world coordinate system where +X is right
// and +Y is up. Rendering flips Y to match screen coordinates (Y down).
// All constraint Jacobian and error computations operate in world coordinates
// (use node->pos[] directly). dc_node / dc_sparse implementations assume world
// coordinates and therefore need no additional sign flips.

// Helper: zero out dense row
static void zero_row(float *out, size_t n_nodes) {
    size_t n = n_nodes * 2;
    for (size_t i = 0; i < n; ++i) out[i] = 0.0f;
}

// DistConstraint implementation (simple)
typedef struct {
    Constraint base;
    Node *other;
    int node_idx;
    int other_idx;
    float distance;
} DistConstraintImpl;

static float dist_err(Constraint *self) {
    DistConstraintImpl *d = (DistConstraintImpl*)self;
    float dx = d->base.node->pos[0] - d->other->pos[0];
    float dy = d->base.node->pos[1] - d->other->pos[1];
    float dist = sqrtf(dx*dx + dy*dy);
    return dist - d->distance;
}

static void dist_dc_node(Constraint *self, Node **nodes, size_t n_nodes, float *out_row) {
    DistConstraintImpl *d = (DistConstraintImpl*)self;
    zero_row(out_row, n_nodes);
    float dx = d->other->pos[0] - d->base.node->pos[0];
    float dy = d->other->pos[1] - d->base.node->pos[1];
    float norm = sqrtf(dx*dx + dy*dy);
    float nx = 0.0f, ny = 0.0f;
    if (norm > 1e-9f) { nx = dx / norm; ny = dy / norm; }
    if (d->node_idx >= 0) {
        out_row[2 * d->node_idx + 0] = nx;
        out_row[2 * d->node_idx + 1] = ny;
    }
    if (d->other_idx >= 0 && !d->other->anchored) {
        out_row[2 * d->other_idx + 0] = -nx;
        out_row[2 * d->other_idx + 1] = -ny;
    }
}

static void dist_dc_sparse(Constraint *self, Node **nodes, size_t n_nodes, DynArray *out) {
    // out holds pointers to pairs int,int? We'll store as malloc'd struct {int idx; float v[2];}
    DistConstraintImpl *d = (DistConstraintImpl*)self;
    float dx = d->other->pos[0] - d->base.node->pos[0];
    float dy = d->other->pos[1] - d->base.node->pos[1];
    float norm = sqrtf(dx*dx + dy*dy);
    float nx = 0.0f, ny = 0.0f;
    if (norm > 1e-9f) { nx = dx / norm; ny = dy / norm; }
    typedef struct { int idx; float v[2]; } Pair;
    Pair *p1 = malloc(sizeof(Pair)); p1->idx = d->node_idx; p1->v[0] = -nx; p1->v[1] = -ny; dynarray_append(out, p1);
    if (d->other_idx >= 0 && !d->other->anchored) {
        Pair *p2 = malloc(sizeof(Pair)); p2->idx = d->other_idx; p2->v[0] = nx; p2->v[1] = ny; dynarray_append(out, p2);
    }
}

// AnchorConstraint implementation (uses a position instead of another Node)
typedef struct {
    DistConstraintImpl base_impl;
    float anchor_pos[2];
} AnchorImpl;

static void anchor_dc_node(Constraint *self, Node **nodes, size_t n_nodes, float *out_row) {
    AnchorImpl *a = (AnchorImpl*)self;
    zero_row(out_row, n_nodes);
    Node *node = a->base_impl.base.node;
    if (!node) return;
    float dx = a->anchor_pos[0] - node->pos[0];
    float dy = a->anchor_pos[1] - node->pos[1];
    float norm = sqrtf(dx*dx + dy*dy);
    float nx = 0.0f, ny = 0.0f;
    if (norm > 1e-9f) { nx = dx / norm; ny = dy / norm; }
    int idx = a->base_impl.node_idx;
    if (idx >= 0) {
        out_row[2*idx + 0] = nx;
        out_row[2*idx + 1] = ny;
    }
}

static void anchor_dc_sparse(Constraint *self, Node **nodes, size_t n_nodes, DynArray *out) {
    AnchorImpl *a = (AnchorImpl*)self;
    Node *node = a->base_impl.base.node;
    if (!node) return;
    float dx = a->anchor_pos[0] - node->pos[0];
    float dy = a->anchor_pos[1] - node->pos[1];
    float norm = sqrtf(dx*dx + dy*dy);
    float nx = 0.0f, ny = 0.0f;
    if (norm > 1e-9f) { nx = dx / norm; ny = dy / norm; }
    typedef struct { int idx; float v[2]; } Pair;
    Pair *p = malloc(sizeof(Pair)); p->idx = a->base_impl.node_idx; p->v[0] = nx; p->v[1] = ny; dynarray_append(out, p);
}

static void anchor_dc_triplet(Constraint *self, cs *T, int row) {
    AnchorImpl *a = (AnchorImpl*)self;
    Node *node = a->base_impl.base.node;
    if (!node) return;
    float dx = a->anchor_pos[0] - node->pos[0];
    float dy = a->anchor_pos[1] - node->pos[1];
    float norm = sqrtf(dx*dx + dy*dy);
    double nx = 0.0, ny = 0.0;
    if (norm > 1e-9f) { 
        float inv_n = 1.0f / norm;
        nx = (double)(dx * inv_n); ny = (double)(dy * inv_n);
    }
    int idx = 2 * a->base_impl.node_idx;
    if (idx >= 0) {
        int c0 = idx;
        int c1 = idx + 1;
        cs_entry(T, row, c0, nx);
        cs_entry(T, row, c1, ny);
    }
}

static float anchor_err(Constraint *self) {
    AnchorImpl *a = (AnchorImpl*)self;
    float dx = a->base_impl.base.node->pos[0] - a->anchor_pos[0];
    float dy = a->base_impl.base.node->pos[1] - a->anchor_pos[1];
    return sqrtf(dx*dx + dy*dy);
}

static void anchor_draw(Constraint *self) {
    AnchorImpl *a = (AnchorImpl*)self;
    glColor3f(0.8f,0.6f,0.0f);
    glBegin(GL_LINES);
    glVertex2f(a->base_impl.base.node->pos[0], a->base_impl.base.node->pos[1]);
    glVertex2f(a->anchor_pos[0], a->anchor_pos[1]);
    glEnd();
}

Constraint* anchorconstraint_create(Node *node, float x, float y) {
    AnchorImpl *a = (AnchorImpl*)malloc(sizeof(AnchorImpl));
    memset(a,0,sizeof(*a));
    a->base_impl.base.node = node;
    a->base_impl.base.type = CT_ANCHOR;
    a->base_impl.base.stiffness = 1.0f;
    a->base_impl.base.other = NULL;
    a->base_impl.base.rest_length = 0.0f;
    a->base_impl.other = NULL;
    a->base_impl.node_idx = node ? node->idx : -1;
    a->base_impl.other_idx = -1;
    a->anchor_pos[0] = x; a->anchor_pos[1] = y;
    a->base_impl.base.err = anchor_err;
    a->base_impl.base.dc_node = anchor_dc_node;
    a->base_impl.base.dc_sparse = anchor_dc_sparse;
    a->base_impl.base.dc_triplet = anchor_dc_triplet;
    a->base_impl.base.draw = anchor_draw;
    if (node) node->anchored = true;
    return (Constraint*)a;
}


// Emit entries for this distance constraint directly into a CSparse triplet.
// row is the constraint row index. T must be created with cs_spalloc(m, n2, nzmax, 1, 1).
static void dist_dc_triplet(Constraint *self, cs *T, int row) {
    DistConstraintImpl *d = (DistConstraintImpl*)self;
    
    float dx = d->other->pos[0] - d->base.node->pos[0];
    float dy = d->other->pos[1] - d->base.node->pos[1];
    float norm = sqrtf(dx*dx + dy*dy);
    double nx = 0.0, ny = 0.0;
    if (norm > 1e-9f) { nx = (double)(dx / norm); ny = (double)(dy / norm); }
    int c0, c1;
    if (d->node_idx >= 0) {
        c0 = 2 * d->node_idx + 0;
        c1 = 2 * d->node_idx + 1;
        cs_entry(T, row, c0, nx);
        cs_entry(T, row, c1, ny);
    }
    if (d->other_idx >= 0 && !d->other->anchored) {
        c0 = 2 * d->other_idx + 0;
        c1 = 2 * d->other_idx + 1;
        cs_entry(T, row, c0, -nx);
        cs_entry(T, row, c1, -ny);
    }
}

static void dist_draw(Constraint *self) {
    DistConstraintImpl *d = (DistConstraintImpl*)self;
    glColor3f(0.0f, 0.6f, 0.0f);
    glBegin(GL_LINES);
    glVertex2f(d->base.node->pos[0], d->base.node->pos[1]);
    glVertex2f(d->other->pos[0], d->other->pos[1]);
    glEnd();
}

Constraint* distconstraint_create(Node *node, Node *other, float distance) {
    DistConstraintImpl *d = (DistConstraintImpl*)malloc(sizeof(DistConstraintImpl));
    memset(d, 0, sizeof(*d));
    d->base.node = node;
    d->base.type = CT_DIST;
    d->base.stiffness = 1.0f;
    d->base.other = other;
    /* If distance <= 0 was passed, use the current distance between the nodes */
    if (distance <= 0.0f && node && other) {
        float dx = node->pos[0] - other->pos[0];
        float dy = node->pos[1] - other->pos[1];
        distance = sqrtf(dx*dx + dy*dy);
    }
    d->base.rest_length = distance;
    d->other = other;
    d->node_idx = node ? node->idx : -1;
    d->other_idx = other ? other->idx : -1;
    d->distance = distance;
    d->base.err = dist_err;
    d->base.dc_node = dist_dc_node;
    d->base.dc_sparse = dist_dc_sparse;
    d->base.dc_triplet = dist_dc_triplet;
    d->base.draw = dist_draw;
    if (node && node->constraints) dynarray_append(node->constraints, d);
    return (Constraint*)d;
}

// SpringConstraint: derive from DistConstraint but with stiffness and different draw
typedef struct {
    DistConstraintImpl base_impl; // reuse
    float k;
} SpringImpl;

static float spring_err(Constraint *self) {
    DistConstraintImpl *d = (DistConstraintImpl*)self;
    float dx = d->base.node->pos[0] - d->other->pos[0];
    float dy = d->base.node->pos[1] - d->other->pos[1];
    float dist = sqrtf(dx*dx + dy*dy);
    return dist - d->distance;
}

static void spring_draw(Constraint *self) {
    DistConstraintImpl *d = (DistConstraintImpl*)self;
    SpringImpl *sp = (SpringImpl*)self;
    // color based on energy stored in spring: white = no strain,
    // compression -> red, stretch -> blue. Strength scaled by spring energy.
    float dx = d->base.node->pos[0] - d->other->pos[0];
    float dy = d->base.node->pos[1] - d->other->pos[1];
    float dist = sqrtf(dx*dx + dy*dy);
    float rest = (d->distance > 0.0f) ? d->distance : 1.0f;
    float ext = dist - rest; // positive = stretch, negative = compression
    // energy = 0.5 * k * ext^2
    float k = sp->k;
    float energy = 0.5f * k * ext * ext;
    // map energy -> intensity in [0,1] (soft saturation so large energies don't clamp abruptly)
    float intensity = 0.5f * energy / (energy + 1.0f);
    if (intensity < 0.0f) intensity = 0.0f; if (intensity > 1.0f) intensity = 1.0f;
    float r = 1.0f, g = 1.0f, b = 1.0f;
    if (ext < -1e-6f) {
        // compression: red, reduce G and B by intensity
        r = 1.0f; g = 1.0f - intensity; b = 1.0f - intensity;
    } else if (ext > 1e-6f) {
        // stretch: blue, reduce R and G by intensity
        b = 1.0f; r = 1.0f - intensity; g = 1.0f - intensity;
    } else {
        // near-zero extension: white
        r = g = b = 1.0f;
    }
    glColor3f(r, g, b);
    glBegin(GL_LINES);
    glVertex2f(d->base.node->pos[0], d->base.node->pos[1]);
    glVertex2f(d->other->pos[0], d->other->pos[1]);
    glEnd();
}

Constraint* springconstraint_create(Node *node, Node *other, float stiffness, float distance) {
    SpringImpl *s = (SpringImpl*)malloc(sizeof(SpringImpl));
    memset(s,0,sizeof(*s));
    s->base_impl.base.node = node;
    s->base_impl.base.type = CT_SPRING;
    s->base_impl.base.stiffness = stiffness;
    s->base_impl.base.other = other;
    s->base_impl.base.rest_length = distance;
    s->base_impl.other = other;
    s->base_impl.node_idx = node ? node->idx : -1;
    s->base_impl.other_idx = other ? other->idx : -1;
    s->base_impl.distance = distance;
    s->k = stiffness;
    s->base_impl.base.err = spring_err;
    s->base_impl.base.dc_node = NULL;
    s->base_impl.base.dc_sparse = NULL;
    s->base_impl.base.dc_triplet = NULL;
    s->base_impl.base.draw = spring_draw;
    if (node && node->constraints) dynarray_append(node->constraints, s);
    return (Constraint*)s;
}



// AnchorConstraint (implemented above)
WallSegment* wallsegment_create(Node *A, Node *B, float restitution, float friction) {

    WallSegment *w = (WallSegment*)malloc(sizeof(WallSegment));
    w->A = A; w->B = B; w->restitution = restitution; w->friction = friction;
    return w;
}
void wallsegment_free(WallSegment *w) { free(w); }

// Note: callers must free pairs appended by dc_sparse.

