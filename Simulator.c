#include "Simulator.h"
#include "Octree.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
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
    size_t n3 = n * 3;
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
    cs *T = cs_spalloc((int)m, (int)n3, nzmax, 1, 1); // triplet form
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
        double scale = (invm && col < (int)n3) ? invm[col] : 1.0;

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
    s->gravity[0] = 0.0f; s->gravity[1] = -9.81f; s->gravity[2] = 0.0f;
    s->solver_iters = 100;
    s->damping = 0.01f;
    s->velocity_blend = 0.5f; // blend factor between old velocity and position-derived velocity
    // Create octree spanning a reasonable world space (can be resized later)
    s->octree = octree_create(-1000.0f, -1000.0f, -1000.0f, 1000.0f, 1000.0f, 1000.0f, 8);
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
    // free triangle walls
    for (size_t i = 0; i < dynarray_size(s->walls); ++i) {
        TriangleWall *w = (TriangleWall*)dynarray_get(s->walls, i);
        if (w) trianglewall_free(w);
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

void simulator_add_wall(Simulator *s, TriangleWall *w) {
    if (!s || !w) return;
    dynarray_append(s->walls, w);
}

// Helper: check collision between triangle wall and node (3D point-triangle collision)
// Returns 1 if collision detected, fills out_penetration, out_normal[3], and barycentric coords
static int triangle_check_collision(TriangleWall *w, Node *node, 
                                    float *out_penetration, float out_normal[3],
                                    float *out_u, float *out_v) {
    // Get triangle vertices
    float *a = w->A->pos;
    float *b = w->B->pos;
    float *c = w->C->pos;
    float *p = node->pos;
    
    // Compute triangle edges
    float e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    float e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    
    // Compute triangle normal via cross product
    float normal[3] = {
        e1[1]*e2[2] - e1[2]*e2[1],
        e1[2]*e2[0] - e1[0]*e2[2],
        e1[0]*e2[1] - e1[1]*e2[0]
    };
    float normal_len = sqrtf(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
    if (normal_len < 1e-9f) return 0; // degenerate triangle
    normal[0] /= normal_len; normal[1] /= normal_len; normal[2] /= normal_len;
    
    // Project point onto triangle plane
    float ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
    float dist_to_plane = ap[0]*normal[0] + ap[1]*normal[1] + ap[2]*normal[2];
    
    // Check if point is within radius distance from plane
    if (fabsf(dist_to_plane) > node->radius) return 0;
    
    // Project point onto plane
    float proj[3] = {
        p[0] - normal[0] * dist_to_plane,
        p[1] - normal[1] * dist_to_plane,
        p[2] - normal[2] * dist_to_plane
    };
    
    // Compute barycentric coordinates of projected point
    // Using method: solve [e1 e2] * [u v]^T = proj - a
    float v0[3] = {proj[0] - a[0], proj[1] - a[1], proj[2] - a[2]};
    
    float dot00 = e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2];
    float dot01 = e1[0]*e2[0] + e1[1]*e2[1] + e1[2]*e2[2];
    float dot02 = e1[0]*v0[0] + e1[1]*v0[1] + e1[2]*v0[2];
    float dot11 = e2[0]*e2[0] + e2[1]*e2[1] + e2[2]*e2[2];
    float dot12 = e2[0]*v0[0] + e2[1]*v0[1] + e2[2]*v0[2];
    
    float inv_denom = 1.0f / (dot00 * dot11 - dot01 * dot01);
    float u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    float v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
    
    // Check if point is inside triangle
    if (u < 0.0f || v < 0.0f || (u + v) > 1.0f) return 0;
    
    // Collision detected - compute penetration
    float penetration = node->radius - fabsf(dist_to_plane);
    if (penetration <= 0.0f) return 0;
    
    // Set outputs
    if (out_penetration) *out_penetration = penetration;
    if (out_normal) {
        // Normal points away from triangle toward node
        if (dist_to_plane < 0.0f) {
            out_normal[0] = -normal[0];
            out_normal[1] = -normal[1];
            out_normal[2] = -normal[2];
        } else {
            out_normal[0] = normal[0];
            out_normal[1] = normal[1];
            out_normal[2] = normal[2];
        }
    }
    if (out_u) *out_u = u;
    if (out_v) *out_v = v;
    
    return 1;
}

// Compute collision forces using octree acceleration structure
static void compute_triangle_collision_forces(Simulator *s, float *collision_forces) {
    if (!s || !collision_forces) return;
    const float PENETRATION_STIFFNESS = 500.0f;
    const float NORMAL_DAMP_K = 100.0f;
    const float Kt = 150.0f;
    
    size_t n_nodes = dynarray_size(s->nodes);
    size_t n_walls = dynarray_size(s->walls);
    if (n_nodes == 0 || n_walls == 0) return;
    
    // Rebuild octree each frame (simple approach - could be optimized)
    if (s->octree) {
        octree_clear(s->octree);
    }
    
    // Insert all triangle walls into octree
    for (size_t wi = 0; wi < n_walls; ++wi) {
        TriangleWall *w = (TriangleWall*)dynarray_get(s->walls, wi);
        if (!w || !w->A || !w->B || !w->C) continue;
        octree_insert_triangle(s->octree, (int)wi, w->A->pos, w->B->pos, w->C->pos);
    }
    
    // Check each node against candidate triangles from octree
    for (size_t ni = 0; ni < n_nodes; ++ni) {
        Node *node = (Node*)dynarray_get(s->nodes, ni);
        if (!node) continue;
        if (node->anchored && !node->collide_when_anchored) continue;
        if (!node->collide_with_walls) continue;
        
        // Query triangles that could collide with this node
        DynArray *candidates = octree_query_triangles(s->octree, node->pos);
        if (!candidates) continue;
        
        for (size_t ci = 0; ci < dynarray_size(candidates); ++ci) {
            int *tri_idx_ptr = (int*)dynarray_get(candidates, ci);
            if (!tri_idx_ptr) continue;
            int tri_idx = *tri_idx_ptr;
            if (tri_idx < 0 || (size_t)tri_idx >= n_walls) continue;
            
            TriangleWall *w = (TriangleWall*)dynarray_get(s->walls, tri_idx);
            if (!w) continue;
            
            float penetration;
            float normal[3];
            float u, v;
            
            if (triangle_check_collision(w, node, &penetration, normal, &u, &v)) {
                // Compute separation force
                float sep_force[3] = {
                    normal[0] * penetration * PENETRATION_STIFFNESS,
                    normal[1] * penetration * PENETRATION_STIFFNESS,
                    normal[2] * penetration * PENETRATION_STIFFNESS
                };
                
                // Compute relative velocity (node vel relative to triangle)
                // Triangle velocity is weighted average of vertices
                float wall_vel[3] = {0, 0, 0};
                if (!w->A->anchored) {
                    wall_vel[0] += (1.0f - u - v) * w->A->vel[0];
                    wall_vel[1] += (1.0f - u - v) * w->A->vel[1];
                    wall_vel[2] += (1.0f - u - v) * w->A->vel[2];
                }
                if (!w->B->anchored) {
                    wall_vel[0] += u * w->B->vel[0];
                    wall_vel[1] += u * w->B->vel[1];
                    wall_vel[2] += u * w->B->vel[2];
                }
                if (!w->C->anchored) {
                    wall_vel[0] += v * w->C->vel[0];
                    wall_vel[1] += v * w->C->vel[1];
                    wall_vel[2] += v * w->C->vel[2];
                }
                
                float rel_vel[3] = {
                    node->vel[0] - wall_vel[0],
                    node->vel[1] - wall_vel[1],
                    node->vel[2] - wall_vel[2]
                };
                
                float normal_vel = rel_vel[0]*normal[0] + rel_vel[1]*normal[1] + rel_vel[2]*normal[2];
                
                // Normal damping (only if moving into wall)
                if (normal_vel < 0.0f) {
                    sep_force[0] -= normal[0] * normal_vel * NORMAL_DAMP_K;
                    sep_force[1] -= normal[1] * normal_vel * NORMAL_DAMP_K;
                    sep_force[2] -= normal[2] * normal_vel * NORMAL_DAMP_K;
                }
                
                // Tangential friction
                float tangent[3] = {
                    rel_vel[0] - normal[0] * normal_vel,
                    rel_vel[1] - normal[1] * normal_vel,
                    rel_vel[2] - normal[2] * normal_vel
                };
                float tangent_len = sqrtf(tangent[0]*tangent[0] + tangent[1]*tangent[1] + tangent[2]*tangent[2]);
                
                if (tangent_len > 1e-9f) {
                    float sep_mag = sqrtf(sep_force[0]*sep_force[0] + sep_force[1]*sep_force[1] + sep_force[2]*sep_force[2]);
                    float max_fric = node->friction * sep_mag;
                    float fric_mag = fminf(Kt * tangent_len, max_fric);
                    
                    float fric_force[3] = {
                        -(tangent[0] / tangent_len) * fric_mag,
                        -(tangent[1] / tangent_len) * fric_mag,
                        -(tangent[2] / tangent_len) * fric_mag
                    };
                    
                    sep_force[0] += fric_force[0];
                    sep_force[1] += fric_force[1];
                    sep_force[2] += fric_force[2];
                }
                
                // Apply force to node
                collision_forces[3*ni + 0] += sep_force[0];
                collision_forces[3*ni + 1] += sep_force[1];
                collision_forces[3*ni + 2] += sep_force[2];
                
                // Apply reaction forces to triangle vertices (distributed by barycentric weights)
                if (!w->A->anchored) {
                    float weight_a = 1.0f - u - v;
                    collision_forces[3*w->A->idx + 0] -= sep_force[0] * weight_a;
                    collision_forces[3*w->A->idx + 1] -= sep_force[1] * weight_a;
                    collision_forces[3*w->A->idx + 2] -= sep_force[2] * weight_a;
                }
                if (!w->B->anchored) {
                    collision_forces[3*w->B->idx + 0] -= sep_force[0] * u;
                    collision_forces[3*w->B->idx + 1] -= sep_force[1] * u;
                    collision_forces[3*w->B->idx + 2] -= sep_force[2] * u;
                }
                if (!w->C->anchored) {
                    collision_forces[3*w->C->idx + 0] -= sep_force[0] * v;
                    collision_forces[3*w->C->idx + 1] -= sep_force[1] * v;
                    collision_forces[3*w->C->idx + 2] -= sep_force[2] * v;
                }
            }
        }
        
        // Free candidate list
        dynarray_free(candidates, free);
    }
}

// Legacy 2D spatial hash collision code (deprecated, kept for reference)
// Helper: build spatial hash mapping cell_key -> DynArray of node indices (as int* allocated)
// Simple spatial hash for two 32-bit integers (ix,iy).
void simulator_step(Simulator *s) {
    if (!s) return;
    float dt = s->dt;
    size_t n_nodes = dynarray_size(s->nodes);
    const int N = s->solver_iters; // Python reference used 105 substeps
    if (n_nodes == 0) return;
    /* Allocate scratch arrays once and reuse across substeps to avoid
       repeated malloc/free overhead. Sizes depend on n_nodes and the
       current number of non-spring constraints (m). We conservatively
       compute an initial allocation for b/l based on the current
       constraint count; if m grows we will realloc. */
    float *collision_forces = (float*)malloc(n_nodes * 3 * sizeof(float));
    double *external_forces = (double*)malloc(sizeof(double) * n_nodes * 3);
    size_t n3 = n_nodes * 3;
    double *invm = (double*)malloc(sizeof(double) * n3);

    // estimate maximum m (non-spring constraints) at start
    size_t m_alloc = dynarray_size(s->constraints);
    
    double *b = (double*)malloc(m_alloc * sizeof(double));
    double *l = (double*)malloc(m_alloc * sizeof(double));
    double *corr_f = (double*)malloc(n3 * sizeof(double));

    for (int sub = 0; sub < N; ++sub) {
        float sub_dt = dt / (float)N;

        /* zero scratch arrays for this substep */
        memset(collision_forces, 0, sizeof(float) * n_nodes * 3);
        memset(external_forces, 0, sizeof(double) * n_nodes * 3);

        // compute collision forces fresh each substep
        compute_triangle_collision_forces(s, collision_forces);
        // printf("Substep %d: computed collision forces\n", sub);
        // 1) accumulate external forces (gravity + collisions) into external_forces
        //    instead of applying them directly to velocities. external_forces
        //    stores forces (not accelerations) and will be applied together with
        //    corrective constraint forces after solving.
        for (size_t i = 0; i < n_nodes; ++i) {
            Node *n = (Node*)dynarray_get(s->nodes, i);
            if (!n) continue;
            if (n->anchored || n->sim_ignore) continue;
            // gravity: accumulate as force = m * g
            if (n->isGravity) {
                double mass = (double)fmaxf(n->mass, 1e-9f);
                external_forces[3*i + 0] += mass * (double)s->gravity[0];
                external_forces[3*i + 1] += mass * (double)s->gravity[1];
                external_forces[3*i + 2] += mass * (double)s->gravity[2];
            }
            // collision forces were computed earlier as forces (float)
            if (collision_forces) {
                double fx = (double)collision_forces[3*i + 0];
                double fy = (double)collision_forces[3*i + 1];
                double fz = (double)collision_forces[3*i + 2];
                external_forces[3*i + 0] += fx;
                external_forces[3*i + 1] += fy;
                external_forces[3*i + 2] += fz;
            }
        }

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
            float dz = a->pos[2] - b->pos[2];
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            if (dist < 1e-9f) continue;
            float rest = c->rest_length;
            float k = c->stiffness;
            float mag = k * (dist - rest);
            float dirx = dx / dist, diry = dy / dist, dirz = dz / dist;
            // force on A = -mag * dir (pull A toward B if stretched)
            float fax = -mag * dirx, fay = -mag * diry, faz = -mag * dirz;
            // accumulate spring forces into external_forces (force on A, opposite on B)
            if (!(a->anchored || a->sim_ignore)) {
                external_forces[3*a->idx + 0] += (double)fax;
                external_forces[3*a->idx + 1] += (double)fay;
                external_forces[3*a->idx + 2] += (double)faz;
            }
            if (!(b->anchored || b->sim_ignore)) {
                external_forces[3*b->idx + 0] -= (double)fax;
                external_forces[3*b->idx + 1] -= (double)fay;
                external_forces[3*b->idx + 2] -= (double)faz;
            }
        }

        // 2) Build J, A and solve for constraint multipliers (only non-spring constraints included in J)
        size_t m = 0;
        // build invm vector (length n3) — reuse allocated array
        for (size_t i = 0; i < n_nodes; ++i) {
            Node *nn = (Node*)dynarray_get(s->nodes, i);
            double inv = (nn->anchored || nn->mass <= 0.0f) ? 0.0 : 1.0 / nn->mass;
            invm[3*i + 0] = invm[3*i + 1] = invm[3*i + 2] = inv;
        }
        cs *Jc = NULL;
        cs *Jct = NULL; // Jct is n3 x m (CSC)
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
        // Ensure b/l have enough space for m (realloc if constraints changed)
        if (m > m_alloc) {
            double *nb = (double*)realloc(b, sizeof(double) * m);
            double *nl = (double*)realloc(l, sizeof(double) * m);
            if (!nb || !nl) {
                // allocation failed — bail out of this substep safely
                if (nb) b = nb; if (nl) l = nl; m_alloc = m; /* best-effort */
            } else {
                b = nb; l = nl; m_alloc = m;
            }
        }
        // build rhs b = -err for the included constraints (skip springs)
        memset(b, 0, sizeof(double) * m);
        size_t ridx = 0;
        for (size_t ci = 0; ci < dynarray_size(s->constraints); ++ci) {
            Constraint *c = (Constraint*)dynarray_get(s->constraints, ci);
            if (!c) continue;
            if (c->type == CT_SPRING) continue;
            double C = 0.0;
            if (c->err) C = (double)c->err(c);
            b[ridx++] = -C;
        }

        // reuse l buffer
        memset(l, 0, sizeof(double) * m);
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
        // reuse corr_f buffer — zero before use (cs_gaxpy accumulates into y)
        memset(corr_f, 0, sizeof(double) * n3);
        if (Jc) {
            // Efficient: compute corr_f = -dt * (Jc^T * l).
            // Use CSparse's transpose + gaxpy to compute y = Jc^T * l where Jc is m x n3.
            // cs_gaxpy performs y = A * x (A in CSC). For A=Jct, x=l (len m), y=len n3.
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
        // fprintf(stderr, "corr_f (n3):\n");
        // for (size_t i = 0; i < n3; ++i) fprintf(stderr, "% .6f ", corr_f[i]);
        // fprintf(stderr, "\n-------------------------------\n");

        // apply accel = M_inv @ (corr_f + external_forces) to velocities (scale by sub_dt)
        // corr_f and external_forces are forces; acceleration = M_inv * F_total
        for (size_t i = 0; i < n_nodes; ++i) {
            Node *nn = (Node*)dynarray_get(s->nodes, i);
            if (nn->anchored || nn->sim_ignore) continue;
            double fcx = (double)corr_f[3*i + 0]*N;
            double fcy = (double)corr_f[3*i + 1]*N;
            double fcz = (double)corr_f[3*i + 2]*N;
            double extx = external_forces ? external_forces[3*i + 0] : 0.0;
            double exty = external_forces ? external_forces[3*i + 1] : 0.0;
            double extz = external_forces ? external_forces[3*i + 2] : 0.0;
            double total_fx = fcx + extx;
            double total_fy = fcy + exty;
            double total_fz = fcz + extz;
            double ax = invm[3*i + 0] * total_fx;
            double ay = invm[3*i + 1] * total_fy;
            double az = invm[3*i + 2] * total_fz;
            nn->vel[0] += (float)(ax * (double)sub_dt);
            nn->vel[1] += (float)(ay * (double)sub_dt);
            nn->vel[2] += (float)(az * (double)sub_dt);

            if (nn->anchored || nn->sim_ignore) continue;
            nn->pos[0] += (float)(nn->vel[0] * (double)sub_dt);
            nn->pos[1] += (float)(nn->vel[1] * (double)sub_dt);
            nn->pos[2] += (float)(nn->vel[2] * (double)sub_dt);
        }

        /* free A_sparse and Jc for this substep (they were allocated inside build_sparse_A_from_constraints)
           but keep our scratch buffers for reuse across substeps */
        cs_spfree(A_sparse);
        if (Jc) cs_spfree(Jc);
    }

    /* Free scratch buffers allocated once for all substeps */
    if (collision_forces) free(collision_forces);
    if (external_forces) free(external_forces);
    if (invm) free(invm);
    if (b) free(b);
    if (l) free(l);
    if (corr_f) free(corr_f);

}
void simulator_draw(Simulator *s) {
    if (!s) return;
    // draw constraints
    for (size_t i = 0; i < dynarray_size(s->constraints); ++i) {
        Constraint *c = (Constraint*)dynarray_get(s->constraints, i);
        if (c && c->draw) c->draw(c);
    }
    // draw triangle walls (all 3 edges as black lines in full 3D)
    for (size_t i = 0; i < dynarray_size(s->walls); ++i) {
        TriangleWall *w = (TriangleWall*)dynarray_get(s->walls, i);
        if (!w || !w->A || !w->B || !w->C) continue;
        glColor3f(0.0f, 0.0f, 0.0f);
        glBegin(GL_LINE_LOOP);
        glVertex3f(w->A->pos[0], w->A->pos[1], w->A->pos[2]);
        glVertex3f(w->B->pos[0], w->B->pos[1], w->B->pos[2]);
        glVertex3f(w->C->pos[0], w->C->pos[1], w->C->pos[2]);
        glEnd();
        // draw vertices as small black dots
        glPointSize(4.0f);
        glBegin(GL_POINTS);
        glVertex3f(w->A->pos[0], w->A->pos[1], w->A->pos[2]);
        glVertex3f(w->B->pos[0], w->B->pos[1], w->B->pos[2]);
        glVertex3f(w->C->pos[0], w->C->pos[1], w->C->pos[2]);
        glEnd();
        glPointSize(1.0f);
    }
    // draw nodes (use each node's radius so rendering matches physics/visual size)
    for (size_t i = 0; i < dynarray_size(s->nodes); ++i) {
        Node *n = (Node*)dynarray_get(s->nodes, i);
        if (!n) continue;
        node_draw(n, n->radius);
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
            Node *nn = node_create(-1, 1.0f, x, y, 0.0f);
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
            Node *nn = node_create(-1, 1.0f, (float)x, (float)y, 0.0f);
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
                    Node *nn = node_create(-1, 1.0f, (float)cx, (float)cy, 0.0f);
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
