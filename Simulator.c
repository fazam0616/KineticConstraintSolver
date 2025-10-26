#include "Simulator.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <GL/gl.h>

#include <cs.h>

// Conjugate Gradient solver for symmetric positive-definite dense matrix
static int cg_solve_dense(size_t n, double *A, double *b, double *x, int max_iter, double tol) {
    // x should be initialized to zeros by caller
    double *r = (double*)calloc(n, sizeof(double));
    double *p = (double*)calloc(n, sizeof(double));
    double *Ap = (double*)calloc(n, sizeof(double));
    if (!r || !p || !Ap) {
        free(r); free(p); free(Ap);
        return -1;
    }
    // r = b - A*x (x assumed zero-initialized)
    for (size_t i = 0; i < n; ++i) r[i] = b[i];
    // p = r
    for (size_t i = 0; i < n; ++i) p[i] = r[i];
    double rsold = 0.0;
    for (size_t i = 0; i < n; ++i) rsold += r[i]*r[i];
    if (sqrt(rsold) < tol) {
        free(r); free(p); free(Ap);
        return 0;
    }
    for (int iter = 0; iter < max_iter; ++iter) {
        // Ap = A * p
        for (size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (size_t j = 0; j < n; ++j) s += A[i*n + j] * p[j];
            Ap[i] = s;
        }
        double alpha_num = rsold;
        double alpha_den = 0.0;
        for (size_t i = 0; i < n; ++i) alpha_den += p[i] * Ap[i];
        if (fabs(alpha_den) < 1e-18) break;
        double alpha = alpha_num / alpha_den;
        for (size_t i = 0; i < n; ++i) x[i] += alpha * p[i];
        for (size_t i = 0; i < n; ++i) r[i] -= alpha * Ap[i];
        double rsnew = 0.0;
        for (size_t i = 0; i < n; ++i) rsnew += r[i]*r[i];
        if (sqrt(rsnew) < tol) break;
        double beta = rsnew / rsold;
        for (size_t i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rsold = rsnew;
    }
    free(r); free(p); free(Ap);
    return 0;
}

// Conjugate Gradient solver that multiplies A (CSC) by vectors using CSparse data structures.
static int cg_solve_sparse(int n, cs *A, double *b, double *x, int max_iter, double tol) {
    if (!A || !b || !x) return -1;
    // allocate temporaries
    double *r = (double*)calloc(n, sizeof(double));
    double *p = (double*)calloc(n, sizeof(double));
    double *Ap = (double*)calloc(n, sizeof(double));
    if (!r || !p || !Ap) { free(r); free(p); free(Ap); return -1; }

    // r = b - A*x (x assumed zero)
    for (int i = 0; i < n; ++i) r[i] = b[i];
    for (int i = 0; i < n; ++i) p[i] = r[i];
    double rsold = 0.0;
    for (int i = 0; i < n; ++i) rsold += r[i]*r[i];
    if (sqrt(rsold) < tol) { free(r); free(p); free(Ap); return 0; }

    for (int iter = 0; iter < max_iter; ++iter) {
        // Ap = A * p  (A in CSC): out[row] += A->x * p[col]
        for (int i = 0; i < n; ++i) Ap[i] = 0.0;
        for (int col = 0; col < A->n; ++col) {
            double vcol = p[col];
            for (int pp = A->p[col]; pp < A->p[col+1]; ++pp) {
                int row = A->i[pp];
                Ap[row] += A->x[pp] * vcol;
            }
        }
        double alpha_den = 0.0;
        for (int i = 0; i < n; ++i) alpha_den += p[i] * Ap[i];
        if (fabs(alpha_den) < 1e-18) break;
        double alpha = rsold / alpha_den;
        for (int i = 0; i < n; ++i) x[i] += alpha * p[i];
        for (int i = 0; i < n; ++i) r[i] -= alpha * Ap[i];
        double rsnew = 0.0;
        for (int i = 0; i < n; ++i) rsnew += r[i]*r[i];
        if (sqrt(rsnew) < tol) break;
        double beta = rsnew / rsold;
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rsold = rsnew;
    }

    free(r); free(p); free(Ap);
    return 0;
}

// Build sparse A = J * diag(invm) * J^T using CSparse. Returns a dense double* (m x m)
// by converting the resulting sparse A into a dense matrix. Caller must free.
// This function builds J in triplet form using each constraint's dc_triplet, dc_sparse or dc_node.
// Build sparse A = J * diag(invm) * J^T using CSparse and return the CSC sparse A.
// Also return Jc (compressed J) via out_Jc if requested. Caller must free the returned cs* with cs_spfree().
static cs* build_sparse_A_from_constraints(Simulator *s, double *invm, size_t *out_m, cs **out_Jc, cs **out_Jct) {
    size_t m_total = dynarray_size(s->constraints);
    size_t n = dynarray_size(s->nodes);
    size_t n2 = n * 2;
    // count included constraints (non-spring)
    size_t m = 0;
    for (size_t ci = 0; ci < m_total; ++ci) {
        Constraint *c = (Constraint*)dynarray_get(s->constraints, ci);
        if (!c) continue;
        if (c->type == CT_SPRING) continue;
        m++;
    }
    if (m == 0) { *out_m = 0; return NULL; }
    *out_m = m;

    // estimate nzmax: each constraint affects few nodes; use 4 per constraint as heuristic
    int nzmax = (int)(m * 4 + 16);
    cs *T = cs_spalloc((int)m, (int)n2, nzmax, 1, 1); // triplet form
    if (!T) return NULL;

    // temporary storage used by dc_sparse
    size_t ridx = 0;
    for (size_t ci = 0; ci < m_total; ++ci) {
        Constraint *c = (Constraint*)dynarray_get(s->constraints, ci);
        if (!c) continue;
        if (c->type == CT_SPRING) continue;

        if(c->dc_triplet)
            c->dc_triplet(c, T, (int)ridx);
        
        ridx++;
    }

    // compress triplet -> CSC (this is J in compressed-column form)
    cs *Jc = cs_compress(T);
    cs_spfree(T);
    if (!Jc) return NULL;

    // keep Jc for later (used to compute corr_f). Return it via out_Jc
    if (out_Jc) *out_Jc = Jc; 
    // Build A = (Jc*dt) * diag(invm) * (Jc*dt)^T efficiently.
    // We will: 1) make a compressed copy of Jc and scale its columns by invm
    // (this yields Jc_scaled = Jc * diag(invm)), 2) compute A_sparse = Jc_scaled * Jc^T,
    // and finally scale A_sparse by dt^2 so A = (Jc*dt) * diag(invm) * (Jc*dt)^T.
    double dt = (double)(s->dt);
    int nnz = (int)Jc->p[Jc->n];
    cs *Jc_scaled = cs_spalloc((int)Jc->m, (int)Jc->n, nnz, 1, 0); /* compressed */
    if (!Jc_scaled) { /* leave Jc allocated for caller */ return NULL; }
    /* copy column pointers */
    for (int k = 0; k <= Jc->n; ++k) Jc_scaled->p[k] = Jc->p[k];
    /* copy row indices and values */
    for (int k = 0; k < nnz; ++k) {
        Jc_scaled->i[k] = Jc->i[k];
        Jc_scaled->x[k] = Jc->x[k];
    }
    Jc_scaled->nz = -1; /* mark compressed-col */

    /* scale columns by invm[col] (do NOT include dt here) */
    for (int col = 0; col < Jc_scaled->n; ++col) {
        double scale = (invm && col < (int)n2) ? invm[col] : 1.0;

        for (int p = Jc_scaled->p[col]; p < Jc_scaled->p[col+1]; ++p) Jc_scaled->x[p] *= scale * dt * dt;
    }

    // compute A_sparse = Jc_scaled * Jc^T
    *out_Jct = cs_transpose(Jc, 1);
    if (!*out_Jct) { cs_spfree(Jc_scaled); /* leave Jc allocated */ return NULL; }
    cs *A_sparse = cs_multiply(Jc_scaled, *out_Jct);
    cs_spfree(Jc_scaled);
    if (!A_sparse) return NULL;

    // Return A_sparse (CSC) directly. Caller will convert or solve against it.
    *out_m = m;
    return A_sparse;
}

Simulator* simulator_create(float dt) {
    Simulator *s = (Simulator*)malloc(sizeof(Simulator));
    memset(s,0,sizeof(Simulator));
    s->nodes = dynarray_create(16);
    s->constraints = dynarray_create(32);
    s->walls = dynarray_create(8);
    s->dt = dt;
    // Note: rendering uses a top-left origin (y increases downward),
    // Internal physics uses positive-up coordinates: negative y is downwards.
    // Set gravity to negative to point downward in world coordinates.
    s->gravity[0] = 0.0f; s->gravity[1] = -9.81f;
    s->solver_iters = 100;
    s->damping = 0.01f;
    s->velocity_blend = 0.5f; // blend factor between old velocity and position-derived velocity
    return s;
}

void simulator_free(Simulator *s) {
    if (!s) return;
    // free nodes
    for (size_t i = 0; i < dynarray_size(s->nodes); ++i) {
        Node *n = (Node*)dynarray_get(s->nodes, i);
        if (n) node_free(n);
    }
    dynarray_free(s->nodes, NULL);
    // constraints: they are malloc'd by constraint_create functions - free here
    for (size_t i = 0; i < dynarray_size(s->constraints); ++i) {
        Constraint *c = (Constraint*)dynarray_get(s->constraints, i);
        free(c);
    }
    dynarray_free(s->constraints, NULL);
    // free wall segments
    for (size_t i = 0; i < dynarray_size(s->walls); ++i) {
        WallSegment *w = (WallSegment*)dynarray_get(s->walls, i);
        if (w) wallsegment_free(w);
    }
    dynarray_free(s->walls, NULL);
    free(s);
}

void simulator_add_node(Simulator *s, Node *n) {
    if (!s || !n) return;
    n->idx = (int)dynarray_size(s->nodes);
    dynarray_append(s->nodes, n);
}

void simulator_add_constraint(Simulator *s, Constraint *c) {
    if (!s || !c) return;
    dynarray_append(s->constraints, c);
}

void simulator_add_wall(Simulator *s, WallSegment *w) {
    if (!s || !w) return;
    dynarray_append(s->walls, w);
}

// Helper: build spatial hash mapping cell_key -> DynArray of node indices (as int* allocated)
static HashMap* build_spatial_hash(Simulator *s, float cell_size) {
    size_t n = dynarray_size(s->nodes);
    HashMap *map = hashmap_create(1024);
    /* helper: faster integer -> "ix_iy" formatter to avoid snprintf overhead */
    char keybuf[32];
    for (size_t i = 0; i < n; ++i) {
        Node *node = (Node*)dynarray_get(s->nodes, i);
        if (!node) continue;
        int ix = (int)floorf(node->pos[0] / cell_size);
        int iy = (int)floorf(node->pos[1] / cell_size);
        /* write key as "<ix>_<iy>" using a small fast routine instead of snprintf */
        {
            char *p = keybuf;
            /* write ix */
            int v = ix;
            if (v == 0) { *p++ = '0'; }
            else {
                if (v < 0) { *p++ = '-'; v = -v; }
                /* write digits into tmp buffer reversed */
                char tmp[12]; int ti = 0;
                while (v > 0) { tmp[ti++] = (char)('0' + (v % 10)); v /= 10; }
                for (int k = ti-1; k >= 0; --k) *p++ = tmp[k];
            }
            *p++ = '_';
            /* write iy */
            v = iy;
            if (v == 0) { *p++ = '0'; }
            else {
                if (v < 0) { *p++ = '-'; v = -v; }
                char tmp2[12]; int ti2 = 0;
                while (v > 0) { tmp2[ti2++] = (char)('0' + (v % 10)); v /= 10; }
                for (int k = ti2-1; k >= 0; --k) *p++ = tmp2[k];
            }
            *p = '\0';
        }
        DynArray *cell = (DynArray*)hashmap_get(map, keybuf);
        if (!cell) {
            cell = dynarray_create(8);
            // store the DynArray pointer in the hashmap (will be freed by hashmap_free)
            hashmap_put(map, keybuf, cell);
        }
        int *idxptr = (int*)malloc(sizeof(int)); *idxptr = (int)i;
        dynarray_append(cell, idxptr);
    }
    return map;
}

// Helper: get grid cells spanned by a wall segment A->B. We will return a DynArray of keys (char*) allocated
static DynArray* wall_segment_cells(WallSegment *w, float cell_size) {
    DynArray *cells = dynarray_create(8);
    float ax = w->A->pos[0], ay = w->A->pos[1];
    float bx = w->B->pos[0], by = w->B->pos[1];
    float minx = fminf(ax,bx), miny = fminf(ay,by);
    float maxx = fmaxf(ax,bx), maxy = fmaxf(ay,by);
    int ix0 = (int)floorf(minx / cell_size), iy0 = (int)floorf(miny / cell_size);
    int ix1 = (int)floorf(maxx / cell_size), iy1 = (int)floorf(maxy / cell_size);
    char *key;
    for (int ix = ix0; ix <= ix1; ++ix) {
        for (int iy = iy0; iy <= iy1; ++iy) {
            /* allocate just enough for two ints, underscore and sign/term */
            int maxlen = 26; /* safe upper bound for two 32-bit ints with signs */
            key = (char*)malloc((size_t)maxlen);
            /* write key into allocated buffer */
            {
                char *p = key;
                int v = ix;
                if (v == 0) { *p++ = '0'; }
                else {
                    if (v < 0) { *p++ = '-'; v = -v; }
                    char tmp[12]; int ti = 0;
                    while (v > 0) { tmp[ti++] = (char)('0' + (v % 10)); v /= 10; }
                    for (int k = ti-1; k >= 0; --k) *p++ = tmp[k];
                }
                *p++ = '_';
                v = iy;
                if (v == 0) { *p++ = '0'; }
                else {
                    if (v < 0) { *p++ = '-'; v = -v; }
                    char tmp2[12]; int ti2 = 0;
                    while (v > 0) { tmp2[ti2++] = (char)('0' + (v % 10)); v /= 10; }
                    for (int k = ti2-1; k >= 0; --k) *p++ = tmp2[k];
                }
                *p = '\0';
            }
            dynarray_append(cells, key);
        }
    }
    return cells;
}

// Helper: check collision between wall segment and node; returns 1 if collision, fills penetration, normal (out_normal), t (0..1)
static int wall_check_collision(WallSegment *w, Node *node, float *out_penetration, float out_normal[2], float *out_t) {
    float ax = w->A->pos[0], ay = w->A->pos[1];
    float bx = w->B->pos[0], by = w->B->pos[1];
    float vx = bx - ax, vy = by - ay;
    float wx = node->pos[0] - ax, wy = node->pos[1] - ay;
    float len2 = vx*vx + vy*vy;
    float t = 0.0f;
    if (len2 > 1e-9f) t = (wx*vx + wy*vy) / len2;
    if (t < 0.0f) t = 0.0f; else if (t > 1.0f) t = 1.0f;
    float px = ax + vx * t;
    float py = ay + vy * t;
    float nx = node->pos[0] - px;
    float ny = node->pos[1] - py;
    float dist = sqrtf(nx*nx + ny*ny);
    float penetration = node->radius - dist;
    if (out_penetration) *out_penetration = penetration;
    if (out_t) *out_t = t;
    if (dist > 1e-9f) { out_normal[0] = nx / dist; out_normal[1] = ny / dist; }
    else { out_normal[0] = 0.0f; out_normal[1] = 1.0f; }
    return penetration > 0.0f ? 1 : 0;
}


void free_cell_dynarray(void *v) { dynarray_free((DynArray*)v, free); }

// Compute collision forces of nodes with walls and write into provided collision_forces (len = n_nodes*2)
static void compute_wall_collision_forces(Simulator *s, float *collision_forces) {
    if (!s || !collision_forces) return;
    const float PENETRATION_STIFFNESS = 500.0f;
    const float NORMAL_DAMP_K = 100.0f;
    const float Kt = 150.0f;
    float cell_size = 64.0f; // arbitrary; tune for your scene

    size_t n_nodes = dynarray_size(s->nodes);
    if (n_nodes == 0) return;

    HashMap *hash = build_spatial_hash(s, cell_size);

    // helper to free DynArray* values in hashmap

    // For each wall
    for (size_t wi = 0; wi < dynarray_size(s->walls); ++wi) {
        WallSegment *w = (WallSegment*)dynarray_get(s->walls, wi);
        if (!w) continue;
        DynArray *cells = wall_segment_cells(w, cell_size);
        // aggregate node indices in these cells into a temporary DynArray of unique ints
        DynArray *node_indices = dynarray_create(16);
        for (size_t ci = 0; ci < dynarray_size(cells); ++ci) {
            char *key = (char*)dynarray_get(cells, ci);
            DynArray *cell = (DynArray*)hashmap_get(hash, key);
            if (cell) {
                for (size_t k = 0; k < dynarray_size(cell); ++k) {
                    int *ip = (int*)dynarray_get(cell, k);
                    // append if not already present (simple linear check; cells are small)
                    int found = 0;
                    for (size_t z = 0; z < dynarray_size(node_indices); ++z) {
                        int *exist = (int*)dynarray_get(node_indices, z);
                        if (*exist == *ip) { found = 1; break; }
                    }
                    if (!found) {
                        int *copy = (int*)malloc(sizeof(int)); *copy = *ip;
                        dynarray_append(node_indices, copy);
                    }
                }
            }
        }

        int A_anchored = w->A ? w->A->anchored : 0;
        int B_anchored = w->B ? w->B->anchored : 0;
        int both_anchored = A_anchored && B_anchored;

        // For each candidate node
        for (size_t ni = 0; ni < dynarray_size(node_indices); ++ni) {
            int *ip = (int*)dynarray_get(node_indices, ni);
            int i = *ip;
            if (i < 0 || (size_t)i >= n_nodes) continue;
            Node *node = (Node*)dynarray_get(s->nodes, i);
            if (!node) continue;
            if (node->sim_ignore) continue;
            if (node->anchored && !node->collide_when_anchored) continue;
            if (!node->collide_with_walls) continue;

            float penetration = 0.0f; float normal[2]; float t = 0.0f;
            int collided = wall_check_collision(w, node, &penetration, normal, &t);
            if (!collided) continue;

            // separation force
            float sep_fx = normal[0] * (penetration * PENETRATION_STIFFNESS);
            float sep_fy = normal[1] * (penetration * PENETRATION_STIFFNESS);

            // wall velocity at contact point
            float wvx = 0.0f, wvy = 0.0f;
            if (both_anchored) {
                wvx = 0.0f; wvy = 0.0f;
            } else if (A_anchored) {
                wvx = w->B->vel[0] * t; wvy = w->B->vel[1] * t;
            } else if (B_anchored) {
                wvx = w->A->vel[0] * (1.0f - t); wvy = w->A->vel[1] * (1.0f - t);
            } else {
                wvx = w->A->vel[0] * (1.0f - t) + w->B->vel[0] * t;
                wvy = w->A->vel[1] * (1.0f - t) + w->B->vel[1] * t;
            }

            // relative velocity
            float rvx = node->vel[0] - wvx;
            float rvy = node->vel[1] - wvy;
            float normal_velocity = rvx * normal[0] + rvy * normal[1];

            // damping when moving into wall
            if (normal_velocity < 0.0f) {
                float damping_fx = normal[0] * (-normal_velocity * node->mass * NORMAL_DAMP_K);
                float damping_fy = normal[1] * (-normal_velocity * node->mass * NORMAL_DAMP_K);
                sep_fx += damping_fx; sep_fy += damping_fy;
            }

            // friction / tangential damping
            float tangent_x = rvx - normal[0] * normal_velocity;
            float tangent_y = rvy - normal[1] * normal_velocity;
            float t_mag_sq = tangent_x*tangent_x + tangent_y*tangent_y;
            if (t_mag_sq > 1e-16f) {
                float t_mag = sqrtf(t_mag_sq);
                float t_dir_x = tangent_x / t_mag;
                float t_dir_y = tangent_y / t_mag;
                float rv_dot_t = rvx * t_dir_x + rvy * t_dir_y;
                float tangential_damping_x = -t_dir_x * (node->mass * rv_dot_t * Kt);
                float tangential_damping_y = -t_dir_y * (node->mass * rv_dot_t * Kt);

                if (!node->anchored) {
                    float mu = node->friction * w->friction;
                    float sep_mag_sq = sep_fx*sep_fx + sep_fy*sep_fy;
                    float max_fric = mu * sqrtf(sep_mag_sq);
                    float fric_mag_sq = tangential_damping_x*tangential_damping_x + tangential_damping_y*tangential_damping_y;
                    if (fric_mag_sq > max_fric * max_fric) {
                        float sign = (rv_dot_t >= 0.0f) ? 1.0f : -1.0f;
                        tangential_damping_x = -t_dir_x * max_fric * sign;
                        tangential_damping_y = -t_dir_y * max_fric * sign;
                    }
                    sep_fx += tangential_damping_x;
                    sep_fy += tangential_damping_y;
                }
            }

            // accumulate collision force on node
            collision_forces[2*i + 0] += sep_fx;
            collision_forces[2*i + 1] += sep_fy;

            // reaction to wall endpoints
            float reaction_x = -sep_fx, reaction_y = -sep_fy;
            float one_minus_t = 1.0f - t;
            if (w->A && w->A->idx >= 0) {
                int ai = w->A->idx;
                if (ai >= 0 && (size_t)ai < n_nodes) {
                    collision_forces[2*ai+0] += reaction_x * one_minus_t;
                    collision_forces[2*ai+1] += reaction_y * one_minus_t;
                }
            }
            if (w->B && w->B->idx >= 0) {
                int bi = w->B->idx;
                if (bi >= 0 && (size_t)bi < n_nodes) {
                    collision_forces[2*bi+0] += reaction_x * t;
                    collision_forces[2*bi+1] += reaction_y * t;
                }
            }
        }

        // free node_indices and cells
        for (size_t z = 0; z < dynarray_size(node_indices); ++z) free(dynarray_get(node_indices, z));
        dynarray_free(node_indices, NULL);
        for (size_t z = 0; z < dynarray_size(cells); ++z) free(dynarray_get(cells, z));
        dynarray_free(cells, NULL);
    }

    // free hash map (values are DynArray* containing int* entries)
    hashmap_free(hash, free_cell_dynarray);
}

void simulator_step(Simulator *s) {
    if (!s) return;
    float dt = s->dt;
    size_t n_nodes = dynarray_size(s->nodes);
    const int N = s->solver_iters; // Python reference used 105 substeps
    if (n_nodes == 0) return;

    for (int sub = 0; sub < N; ++sub) {
        float sub_dt = dt / (float)N;
        // compute collision forces fresh each substep
        float *collision_forces = (float*)calloc(n_nodes * 2, sizeof(float));
        compute_wall_collision_forces(s, collision_forces);
        // printf("Substep %d: computed collision forces\n", sub);
        // 1) accumulate external forces (gravity + collisions) into external_forces
        //    instead of applying them directly to velocities. external_forces
        //    stores forces (not accelerations) and will be applied together with
        //    corrective constraint forces after solving.
        double *external_forces = (double*)calloc(n_nodes * 2, sizeof(double));
        if (!external_forces) {
            if (collision_forces) free(collision_forces);
            continue;
        }
        for (size_t i = 0; i < n_nodes; ++i) {
            Node *n = (Node*)dynarray_get(s->nodes, i);
            if (!n) continue;
            if (n->anchored || n->sim_ignore) continue;
            // gravity: accumulate as force = m * g
            if (n->isGravity) {
                double mass = (double)fmaxf(n->mass, 1e-9f);
                external_forces[2*i + 0] += mass * (double)s->gravity[0];
                external_forces[2*i + 1] += mass * (double)s->gravity[1];
            }
            // collision forces were computed earlier as forces (float)
            if (collision_forces) {
                double fx = (double)collision_forces[2*i + 0];
                double fy = (double)collision_forces[2*i + 1];
                external_forces[2*i + 0] += fx;
                external_forces[2*i + 1] += fy;
            }
        }
        if (collision_forces) free(collision_forces);

        // 1b) apply spring forces as forces (not constraints)
        for (size_t ci = 0; ci < dynarray_size(s->constraints); ++ci) {
            Constraint *c = (Constraint*)dynarray_get(s->constraints, ci);
            if (!c) continue;
            if (c->type != CT_SPRING) continue;
            // force-based spring: need endpoints in public fields
            Node *a = c->node;
            Node *b = c->other;
            if (!a || !b) continue;
            float dx = a->pos[0] - b->pos[0];
            float dy = a->pos[1] - b->pos[1];
            float dist = sqrtf(dx*dx + dy*dy);
            if (dist < 1e-9f) continue;
            float rest = c->rest_length;
            float k = c->stiffness;
            float mag = k * (dist - rest);
            float dirx = dx / dist, diry = dy / dist;
            // force on A = -mag * dir (pull A toward B if stretched)
            float fax = -mag * dirx, fay = -mag * diry;
            // accumulate spring forces into external_forces (force on A, opposite on B)
            if (!(a->anchored || a->sim_ignore)) {
                external_forces[2*a->idx + 0] += (double)fax;
                external_forces[2*a->idx + 1] += (double)fay;
            }
            if (!(b->anchored || b->sim_ignore)) {
                external_forces[2*b->idx + 0] -= (double)fax;
                external_forces[2*b->idx + 1] -= (double)fay;
            }
        }

        // 2) Build J, A and solve for constraint multipliers (only non-spring constraints included in J)
        size_t m = 0;
        size_t n2 = n_nodes * 2;
        // build invm vector (length n2)
        double *invm = (double*)malloc(sizeof(double) * n2);
        for (size_t i = 0; i < n_nodes; ++i) {
            Node *nn = (Node*)dynarray_get(s->nodes, i);
            double inv = (nn->anchored || nn->mass <= 0.0f) ? 0.0 : 1.0 / nn->mass;
            invm[2*i + 0] = invm[2*i + 1] = inv;
        }
        cs *Jc = NULL;
        
        cs *Jct = NULL; // Jct is n2 x m (CSC)
        cs *A_sparse = build_sparse_A_from_constraints(s, invm, &m, &Jc, &Jct);

        // regularize diagonal slightly where present
        for (int col = 0; col < A_sparse->n && (size_t)col < m; ++col) {
            int found = 0;
            for (int p = A_sparse->p[col]; p < A_sparse->p[col+1]; ++p) {
                int row = A_sparse->i[p];
                if (row == col) { A_sparse->x[p] += 1e-10; found = 1; break; }
            }
            (void)found; /* if diagonal not present, skip */
        }
        // build rhs b = -err for the included constraints (skip springs)
        double *b = (double*)calloc(m, sizeof(double));
        size_t ridx = 0;
        for (size_t ci = 0; ci < dynarray_size(s->constraints); ++ci) {
            Constraint *c = (Constraint*)dynarray_get(s->constraints, ci);
            if (!c) continue;
            if (c->type == CT_SPRING) continue;
            double C = 0.0;
            if (c->err) C = (double)c->err(c);
            b[ridx++] = -C;
        }

        double *l = (double*)calloc(m, sizeof(double));
        if (A_sparse && m > 0) {
            // Try to solve using CSparse direct Cholesky (sparse solver).
            // cs_cholsol returns non-zero on success (per cs_cholsol_mex usage).
            int chol_ok = cs_cholsol(1, A_sparse, b);
            if (chol_ok) {
                // solution is written into b by cs_cholsol
                memcpy(l, b, sizeof(double) * m);
            } else {
                // fallback to iterative sparse CG if Cholesky fails
                int cg_ret = cg_solve_sparse((int)m, A_sparse, b, l, (int)(m*10 + 10), 1e-8);
                (void)cg_ret;
            }
        }

        // corr_f = -J^T * l
        double *corr_f = (double*)calloc(n2, sizeof(double));

        if (Jc) {
            // Efficient: compute corr_f = -dt * (Jc^T * l).
            // Use CSparse's transpose + gaxpy to compute y = Jc^T * l where Jc is m x n2.
            // cs_gaxpy performs y = A * x (A in CSC). For A=Jct, x=l (len m), y=len n2.
            cs_gaxpy(Jct, l, corr_f);
            // scale by -dt
            for (int k = 0; k < Jct->m; ++k) corr_f[k] = -corr_f[k] * (double)dt;
            cs_spfree(Jct);
            
        }


        // DEBUG: Output vectors for each frame
        // fprintf(stderr, "-------------------------------\n");
        // fprintf(stderr, "A (sparse) nnz=%d\n", A_sparse->p[A_sparse->n]);
        // fprintf(stderr, "b (m):\n");
        // for (size_t i = 0; i < m; ++i) fprintf(stderr, "% .6f ", b[i]);
        // fprintf(stderr, "\n");
        // fprintf(stderr, "l (m):\n");
        // for (size_t i = 0; i < m; ++i) fprintf(stderr, "% .6f ", l[i]);
        // fprintf(stderr, "\n");
        // fprintf(stderr, "corr_f (n2):\n");
        // for (size_t i = 0; i < n2; ++i) fprintf(stderr, "% .6f ", corr_f[i]);
        // fprintf(stderr, "\n-------------------------------\n");

        // apply accel = M_inv @ (corr_f + external_forces) to velocities (scale by sub_dt)
        // corr_f and external_forces are forces; acceleration = M_inv * F_total
        for (size_t i = 0; i < n_nodes; ++i) {
            Node *nn = (Node*)dynarray_get(s->nodes, i);
            if (nn->anchored || nn->sim_ignore) continue;
            double fcx = (double)corr_f[2*i + 0]*N;
            double fcy = (double)corr_f[2*i + 1]*N;
            double extx = external_forces ? external_forces[2*i + 0] : 0.0;
            double exty = external_forces ? external_forces[2*i + 1] : 0.0;
            double total_fx = fcx + extx;
            double total_fy = fcy + exty;
            double ax = invm[2*i + 0] * total_fx;
            double ay = invm[2*i + 1] * total_fy;
            nn->vel[0] += (float)(ax * (double)sub_dt);
            nn->vel[1] += (float)(ay * (double)sub_dt);

            if (nn->anchored || nn->sim_ignore) continue;
            nn->pos[0] += (float)(nn->vel[0] * (double)sub_dt);
            nn->pos[1] += (float)(nn->vel[1] * (double)sub_dt);
        }

        free(corr_f);
        free(l);
        free(b);
        cs_spfree(A_sparse);
        if (external_forces) free(external_forces);
        free(invm);
        if (Jc) cs_spfree(Jc);
    }
}

void simulator_draw(Simulator *s) {
    if (!s) return;
    // draw constraints
    for (size_t i = 0; i < dynarray_size(s->constraints); ++i) {
        Constraint *c = (Constraint*)dynarray_get(s->constraints, i);
        if (c && c->draw) c->draw(c);
    }
    // draw wall segments (black lines)
    for (size_t i = 0; i < dynarray_size(s->walls); ++i) {
        WallSegment *w = (WallSegment*)dynarray_get(s->walls, i);
        if (!w) continue;
        glColor3f(0.0f, 0.0f, 0.0f);
        glBegin(GL_LINES);
        glVertex2f(w->A->pos[0], w->A->pos[1]);
        glVertex2f(w->B->pos[0], w->B->pos[1]);
        glEnd();
    // draw endpoints as small black dots
    glPointSize(4.0f);
    glBegin(GL_POINTS);
    glVertex2f(w->A->pos[0], w->A->pos[1]);
    glVertex2f(w->B->pos[0], w->B->pos[1]);
    glEnd();
    glPointSize(1.0f);
    }
    // draw nodes
    for (size_t i = 0; i < dynarray_size(s->nodes); ++i) {
        Node *n = (Node*)dynarray_get(s->nodes, i);
        node_draw(n, 6.0f);
    }
}

// Triangle struct for mesh output
typedef struct {
    Node *a, *b, *c;
} Triangle;

// Helper: compute signed area of polygon (nodes in order)
static double polygon_signed_area(Node **pts, int n) {
    double area = 0.0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        area += (double)pts[i]->pos[0] * (double)pts[j]->pos[1] - (double)pts[j]->pos[0] * (double)pts[i]->pos[1];
    }
    return area * 0.5;
}

static double cross_z(const Node *a, const Node *b, const Node *c) {
    double ux = b->pos[0] - a->pos[0];
    double uy = b->pos[1] - a->pos[1];
    double vx = c->pos[0] - a->pos[0];
    double vy = c->pos[1] - a->pos[1];
    return ux * vy - uy * vx;
}

// point-in-triangle (strict), returns 1 if p inside triangle abc
static int point_in_triangle(const Node *a, const Node *b, const Node *c, const Node *p) {
    double c1 = cross_z(a,b,p);
    double c2 = cross_z(b,c,p);
    double c3 = cross_z(c,a,p);
    int has_neg = (c1 < 0) || (c2 < 0) || (c3 < 0);
    int has_pos = (c1 > 0) || (c2 > 0) || (c3 > 0);
    return !(has_neg && has_pos);
}

// point-in-polygon (ray-casting). returns 1 if (x,y) is inside polygon pts[0..n-1]
static int point_in_polygon(Node **pts, int n, double x, double y) {
    int inside = 0;
    for (int i = 0, j = n-1; i < n; j = i++) {
        double xi = pts[i]->pos[0], yi = pts[i]->pos[1];
        double xj = pts[j]->pos[0], yj = pts[j]->pos[1];
        int intersect = ((yi > y) != (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi + 0.0) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

// Generate mesh from polygon nodes
DynArray* simulator_generate_mesh_from_nodes(Simulator *sim, DynArray *poly_nodes, int k) {
    if (!sim || !poly_nodes || k < 1) return NULL;
    int n_orig = (int)dynarray_size(poly_nodes);
    if (n_orig < 3) return NULL;

    // 1) sample polygon boundary (outline) to create boundary nodes and constraints
    DynArray *outline = dynarray_create(n_orig * k + 4);
    for (int i = 0; i < n_orig; ++i) {
        Node *A = (Node*)dynarray_get(poly_nodes, i);
        Node *B = (Node*)dynarray_get(poly_nodes, (i+1)%n_orig);
        for (int j = 0; j < k; ++j) {
            float t = (float)j / (float)k;
            float x = A->pos[0] * (1.0f - t) + B->pos[0] * t;
            float y = A->pos[1] * (1.0f - t) + B->pos[1] * t;
            Node *nn = node_create(-1, 1.0f, x, y);
            simulator_add_node(sim, nn);
            dynarray_append(outline, nn);
        }
    }
    int N = (int)dynarray_size(outline);
    if (N < 3) { dynarray_free(outline, NULL); return NULL; }
    for (int i = 0; i < N; ++i) {
        Node *a = (Node*)dynarray_get(outline, i);
        Node *b = (Node*)dynarray_get(outline, (i+1)%N);
        simulator_add_constraint(sim, distconstraint_create(a, b, -1.0f));
    }

    // prepare polygon points array for point-in-polygon tests
    Node **poly_pts = (Node**)malloc(sizeof(Node*) * n_orig);
    for (int i = 0; i < n_orig; ++i) poly_pts[i] = (Node*)dynarray_get(poly_nodes, i);

    // compute average edge length to choose grid spacing
    double total_len = 0.0;
    for (int i = 0; i < n_orig; ++i) {
        Node *A = poly_pts[i]; Node *B = poly_pts[(i+1)%n_orig];
        double dx = A->pos[0] - B->pos[0], dy = A->pos[1] - B->pos[1];
        total_len += sqrt(dx*dx + dy*dy);
    }
    double avg_edge = total_len / (double)n_orig;
    double spacing = avg_edge / (double)k;
    if (spacing <= 1e-6) spacing = 1.0; // fallback

    // bounding box
    double minx = poly_pts[0]->pos[0], maxx = minx;
    double miny = poly_pts[0]->pos[1], maxy = miny;
    for (int i = 1; i < n_orig; ++i) {
        double x = poly_pts[i]->pos[0], y = poly_pts[i]->pos[1];
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
    }

    int nx = (int)floor((maxx - minx) / spacing) + 1;
    int ny = (int)floor((maxy - miny) / spacing) + 1;
    if (nx < 1) nx = 1; if (ny < 1) ny = 1;

    // grid nodes storage
    Node **grid = (Node**)malloc(sizeof(Node*) * (size_t)nx * (size_t)ny);
    for (int i = 0; i < nx*ny; ++i) grid[i] = NULL;

    // helper: threshold to avoid duplicating boundary nodes
    double dup_thresh2 = (spacing * 0.45) * (spacing * 0.45);

    // create interior grid nodes
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            double x = minx + ix * spacing;
            double y = miny + iy * spacing;
            // test inside polygon
            if (!point_in_polygon(poly_pts, n_orig, x, y)) continue;
            // avoid points too close to boundary sample nodes
            int too_close = 0;
            for (int b = 0; b < N; ++b) {
                Node *bn = (Node*)dynarray_get(outline, b);
                double dx = bn->pos[0] - x, dy = bn->pos[1] - y;
                if (dx*dx + dy*dy < dup_thresh2) { too_close = 1; break; }
            }
            if (too_close) continue;
            Node *nn = node_create(-1, 1.0f, (float)x, (float)y);
            simulator_add_node(sim, nn);
            grid[ix*ny + iy] = nn;
        }
    }

    // create constraints between neighboring grid nodes to form a lattice
    DynArray *triangles = dynarray_create(128);
    for (int ix = 0; ix < nx; ++ix) {
        for (int iy = 0; iy < ny; ++iy) {
            Node *n00 = grid[ix*ny + iy];
            if (!n00) continue;
            // right neighbor
            if (ix + 1 < nx) {
                Node *n10 = grid[(ix+1)*ny + iy];
                if (n10) simulator_add_constraint(sim, distconstraint_create(n00, n10, -1.0f));
            }
            // up neighbor
            if (iy + 1 < ny) {
                Node *n01 = grid[ix*ny + (iy+1)];
                if (n01) simulator_add_constraint(sim, distconstraint_create(n00, n01, -1.0f));
            }
            // diagonals
            if (ix + 1 < nx && iy + 1 < ny) {
                Node *n11 = grid[(ix+1)*ny + (iy+1)];
                Node *n10 = grid[(ix+1)*ny + iy];
                Node *n01 = grid[ix*ny + (iy+1)];
                if (n11) {
                    // cross diagonals for stiffness
                    if (n11 && n00) simulator_add_constraint(sim, distconstraint_create(n00, n11, -1.0f));
                    if (n10 && n01) simulator_add_constraint(sim, distconstraint_create(n10, n01, -1.0f));
                    // also produce triangles for output if all four corners exist
                    if (n00 && n10 && n11 && n01) {
                        Triangle *t1 = (Triangle*)malloc(sizeof(Triangle)); t1->a = n00; t1->b = n10; t1->c = n11; dynarray_append(triangles, t1);
                        Triangle *t2 = (Triangle*)malloc(sizeof(Triangle)); t2->a = n00; t2->b = n11; t2->c = n01; dynarray_append(triangles, t2);
                    }
                }
            }
        }
    }

    // Connect outline (boundary) nodes to the interior lattice. We try three
    // strategies in order: (1) find the closest existing interior grid node
    // (global scan), (2) search nearby grid cells and create a new interior
    // node if none found, (3) as a last resort connect to the closest
    // outline neighbor. This guarantees each outline sample connects to the
    // lattice.
    double connect_thresh2 = (spacing * 1.25) * (spacing * 1.25);
    int max_create_radius = 3; /* in grid cells */
    for (int b = 0; b < N; ++b) {
        Node *bn = (Node*)dynarray_get(outline, b);
        if (!bn) continue;
        double fx = bn->pos[0], fy = bn->pos[1];

        /* 1) global scan for nearest interior grid node */
        double best_d2 = 1e300; int best_idx = -1;
        for (int gx = 0; gx < nx; ++gx) {
            for (int gy = 0; gy < ny; ++gy) {
                Node *g = grid[gx*ny + gy];
                if (!g) continue;
                double dxp = g->pos[0] - fx;
                double dyp = g->pos[1] - fy;
                double d2 = dxp*dxp + dyp*dyp;
                if (d2 < best_d2) { best_d2 = d2; best_idx = gx*ny + gy; }
            }
        }
        if (best_idx >= 0 && best_d2 <= connect_thresh2) {
            Node *g = grid[best_idx];
            if (g) simulator_add_constraint(sim, distconstraint_create(bn, g, -1.0f));
            continue;
        }

        /* 2) try to create a new interior node in a nearby empty grid cell */
        int bix = (int)floor((fx - minx) / spacing + 0.5);
        int biy = (int)floor((fy - miny) / spacing + 0.5);
        int created = 0;
        for (int r = 0; r <= max_create_radius && !created; ++r) {
            for (int dx = -r; dx <= r && !created; ++dx) {
                for (int dy = -r; dy <= r && !created; ++dy) {
                    if (abs(dx) != r && abs(dy) != r) continue; /* perimeter only */
                    int gx = bix + dx; int gy = biy + dy;
                    if (gx < 0 || gx >= nx || gy < 0 || gy >= ny) continue;
                    int idx = gx*ny + gy;
                    /* if already occupied, skip (we already tried global scan) */
                    if (grid[idx]) continue;
                    double cx = minx + gx * spacing;
                    double cy = miny + gy * spacing;
                    if (!point_in_polygon(poly_pts, n_orig, cx, cy)) continue;
                    /* create new interior node at cell center */
                    Node *nn = node_create(-1, 1.0f, (float)cx, (float)cy);
                    simulator_add_node(sim, nn);
                    grid[idx] = nn;
                    /* connect new node to any existing neighbor nodes (8-connectivity) */
                    for (int nx_off = -1; nx_off <= 1; ++nx_off) {
                        for (int ny_off = -1; ny_off <= 1; ++ny_off) {
                            if (nx_off == 0 && ny_off == 0) continue;
                            int ngx = gx + nx_off, ngy = gy + ny_off;
                            if (ngx < 0 || ngx >= nx || ngy < 0 || ngy >= ny) continue;
                            Node *nbr = grid[ngx*ny + ngy];
                            if (nbr) simulator_add_constraint(sim, distconstraint_create(nn, nbr, -1.0f));
                        }
                    }
                    /* connect outline sample to this new interior node */
                    simulator_add_constraint(sim, distconstraint_create(bn, nn, -1.0f));
                    created = 1;
                }
            }
        }
        if (created) continue;

        /* 3) fallback: connect to nearest outline neighbor (avoid leaving isolated) */
        Node *alt = (Node*)dynarray_get(outline, (b+1)%N);
        if (alt) simulator_add_constraint(sim, distconstraint_create(bn, alt, -1.0f));
    }

    // free temporary structures
    free(poly_pts);
    // free grid array (nodes are kept in simulator)
    free(grid);

    // outline array container is freed but nodes remain in simulator
    dynarray_free(outline, NULL);
    return triangles;
}
