#include "Simulator.h"
// #include "Octree.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <GL/glew.h>
#include <GL/gl.h>

/* High-resolution CPU wall-clock timestamp in seconds */
static double cpu_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#include <cs.h>

// Helper: mark node dirty if Manhattan distance exceeds threshold
void mark_node_dirty_if_moved(Node *node, float threshold) {
    float dx = fabsf(node->pos[0] - node->prev_position[0]);
    float dy = fabsf(node->pos[1] - node->prev_position[1]);
    float dz = fabsf(node->pos[2] - node->prev_position[2]);
    float manhattan = dx + dy + dz;
    if (manhattan > threshold) {
        node->dirty = true;
    }
    // Update prev_position for next check
    node->prev_position[0] = node->pos[0];
    node->prev_position[1] = node->pos[1];
    node->prev_position[2] = node->pos[2];
}

// Full GPU context owning all shaders and SSBOs for the complete solver pipeline.
// SSBO binding layout mirrors the compute shaders:
//  0 positions  1 velocities  2 inv_mass  3 ext_forces  4 node_flags
//  5 constraints  6 J_cols  7 J_vals  8 A_dense(stub)  9 b_vec  10 l_vec(x)
//  11 collision  12 r_vec  13 p_vec  14 Ap_vec
//  23 jt_vec  24 cg_scalars  25 reduce_buf  26 csr_offsets  27 csr_data
typedef struct GPUContext {
    // ----- programs -----
    GLuint prog_build_jb;      // merged build J + b (1 thread/constraint)
    GLuint prog_ext_forces;    // gravity + spring external forces (1 thread/node)
    GLuint prog_apply_corr;    // apply corr_f + symplectic-Euler integrate
    // ----- sparse CG programs -----
    GLuint prog_cg_init;       // x=0,r=p=b,partial rr→reduce_buf; finalizes rr inline
    GLuint prog_cg_j_gather;   // Ap = J jt_vec; partial p·Ap → reduce_buf; finalizes α inline
    GLuint prog_cg_update_xr;  // x += α*p; r -= α*Ap; partial r·r → reduce_buf; finalizes β inline
    GLuint prog_cg_update_p;   // p = r + β*p
    GLuint prog_cg_jt_gather;  // CSR J^T gather, one thread per DOF, no atomics
    // ----- SSBOs -----
    GLuint ssbo_positions;     // binding  0 : n x vec4
    GLuint ssbo_velocities;    // binding  1 : n x vec4
    GLuint ssbo_inv_mass;      // binding  2 : n x float
    GLuint ssbo_ext_forces;    // binding  3 : n x vec4  (gravity + springs, GPU-written)
    GLuint ssbo_node_flags;    // binding  4 : n x uint
    GLuint ssbo_constraints;   // binding  5 : m_total x 8 int  (non-spring first)
    GLuint ssbo_J_cols;        // binding  6 : m_sparse x 6 int
    GLuint ssbo_J_vals;        // binding  7 : m_sparse x 6 float
    GLuint ssbo_A_dense;       // binding  8 : stub (1 float); build_A not dispatched
    GLuint ssbo_b_vec;         // binding  9 : m_sparse float
    GLuint ssbo_l_vec;         // binding 10 : m_sparse float  (CG solution x)
    GLuint ssbo_collision;     // binding 11 : n x vec4  (collision forces, CPU-uploaded)
    GLuint ssbo_r_vec;         // binding 12 : m_sparse float  (CG temp r)
    GLuint ssbo_p_vec;         // binding 13 : m_sparse float  (CG temp p)
    GLuint ssbo_Ap_vec;        // binding 14 : m_sparse float  (CG temp Ap)
    // ----- sparse CG SSBOs -----
    GLuint ssbo_jt_vec;        // binding 23 : 3n floats  (J^T p or J^T λ, written by cg_jt_gather)
    GLuint ssbo_cg_scalars;    // binding 24 : 8 floats [rr,pAp,alpha,beta,...] + uint reduce_ctr
    GLuint ssbo_reduce_buf;    // binding 25 : ceil(m/64) floats
    GLuint ssbo_csr_offsets;   // binding 26 : (3n+1) ints  [static J^T sparsity CSR]
    GLuint ssbo_csr_data;      // binding 27 : n_csr_entries×2 ints {c, k_slot}
    int    n_csr_entries;      // total CSR entries (set at upload)
    // ----- sizes filled during scene upload -----
    int node_count;
    int m_sparse;   // non-spring constraint count
    int m_total;    // total constraint count (including springs)
    int gpu_debug;  // 1 → print CPU vs GPU comparison each first substep
    // ----- cached uniform locations (set once at init / upload) -----
    GLint u_ef_n, u_ef_m_sparse, u_ef_m_total, u_ef_gravity;
    GLint u_bjb_m;
    GLint u_ac_n, u_ac_dt, u_ac_sub_dt, u_ac_N;
    // ----- sparse CG uniform locations -----
    GLint u_ci_m;                          // cg_init
    GLint u_cjg_m, u_cjg_dt;              // cg_j_gather
    GLint u_cuxr_m;                        // cg_update_xr
    GLint u_cup_m;                         // cg_update_p
    GLint u_cjtg_n3, u_cjtg_src, u_cjtg_apply_minv; // cg_jt_gather
    // ----- LBVH build programs (parallel, 5 shaders) -----
    // Old serial single-thread builders replaced by a 3-pass pipeline:
    //   Pass 1 (morton): compute per-prim AABB + Morton code → scratch slot [7]
    //   Pass 2 (sort):   bitonic sort of scratch by Morton code (local or 2-pass global)
    //   Pass 3 (build):  Karras 2012 parallel internal/leaf node construction
    GLuint prog_lbvh_morton_walls; // Pass 1 walls: SSBO 0+17→22, dispatch ceil(n/64)
    GLuint prog_lbvh_morton_edges; // Pass 1 edges: SSBO 0+5→21,  dispatch ceil(m/64)
    GLuint prog_lbvh_sort_walls;   // Pass 2 walls: SSBO 22,       dispatch 1 or ceil(n2/2/64)
    GLuint prog_lbvh_sort_edges;   // Pass 2 edges: SSBO 21,       dispatch 1 or ceil(n2/2/64)
    GLuint prog_lbvh_build_walls;  // Pass 3 walls: SSBO 22→15,    dispatch ceil((2n-1)/64)
    GLuint prog_lbvh_build_edges;  // Pass 3 edges: SSBO 21→20,    dispatch ceil((2n-1)/64)
    // Uniform locations – Pass 1
    GLint  u_mw_n_prims, u_mw_scene_min, u_mw_scene_max;   // walls morton
    GLint  u_me_m_total, u_me_scene_min, u_me_scene_max;   // edges morton
    // Uniform locations – Pass 2
    GLint  u_sw_n_prims, u_sw_pass_len, u_sw_sub_step, u_sw_local;  // walls sort
    GLint  u_se_n_prims, u_se_pass_len, u_se_sub_step, u_se_local;  // edges sort
    // Uniform locations – Pass 3
    GLint  u_bw_n_prims;   // walls build
    GLint  u_be_n_prims;   // edges build
    // Scene AABB used for Morton normalization (set at upload from wall geometry)
    float  scene_min[3];
    float  scene_max[3];
    int    last_n_edges;  // actual CT_DIST edge count from last substep build
    // (Keep old names as aliases for any code that still references them)
    GLuint prog_bvh_walls;         // alias → prog_lbvh_build_walls (unused dispatch)
    GLuint prog_bvh_edges;         // alias → prog_lbvh_build_edges (unused dispatch)
    GLint  u_bvh_w_n_tris;         // unused (kept to avoid linker warnings)
    GLint  u_bvh_e_m_total;        // unused
    // ----- collision BVH program + SSBOs (bindings 15-17) -----
    GLuint prog_collision;         // GPU sphere-triangle collision (BVH-traversal)
    GLuint ssbo_bvh_nodes;         // binding 15: flat wall BVH (10 ints/node, GPU-built)
    GLuint ssbo_triangles;         // binding 16: (reserved / unused)
    GLuint ssbo_wall_indices;      // binding 17: 4 ints/tri [a_idx, b_idx, c_idx, 0]
    int    n_triangles;            // number of triangles
    GLint  u_col_n, u_col_n_tris, u_col_stiffness, u_col_damp, u_col_fr;
    // ----- edge-edge collision (bindings 18-21) -----
    GLuint prog_edge_edge;         // GPU edge-edge capsule collision (BVH-traversal)
    GLuint ssbo_velcorr;           // binding 18: n*3 uint (float-as-uint velocity corrections)
    GLuint ssbo_edge_bvh;          // binding 20: flat edge BVH (10 ints/node, GPU-built each substep)
    GLuint ssbo_edge_bvh_scratch;  // binding 21: sorted edge prim list (8 ints/prim)
    GLint  u_ee_n, u_ee_m_total, u_ee_restitution, u_ee_mu;
    // ----- wall BVH scratch (binding 22) -----
    GLuint ssbo_wall_bvh_scratch;  // binding 22: sorted wall prim list (8 ints/prim, GPU-built)
    // ----- particle-mesh force SSBO (written by ParticleSim, read by apply_corr) -----
    GLuint ssbo_particle_mesh_forces; // binding 39: n x vec4 (forces from particle collisions)
    // ----- dynamic wall flags -----
    int rebuild_wall_bvh_per_substep; // when 1: rebuild wall LBVH each substep (for moving walls)
    int enable_edge_edge;             // when 0: skip edge LBVH build + edge-edge collisions
    // ----- indirect dispatch infra (bindings 28-29) -----
    // prog_lbvh_prepare_edge_dispatch: 1-thread shader, reads escratch atomic counter,
    // writes {sort_x,1,1, build_x,1,1} into ssbo_indirect_args and n_edges into ssbo_edge_meta.
    // LBVH
    GLuint prog_lbvh_prepare_edge_dispatch;
    GLint  u_ped_m_total;
    GLuint ssbo_indirect_args;
    GLuint ssbo_edge_meta;
    // baked CG jt_gather variant for CG inloop (src=p, apply_minv=1)
    GLuint prog_cg_jt_gather_inloop;
    GLint  u_cjtg_il_n3;
    // warm-start CG init: given x0 in l_data + jt_vec=M^{-1}J^Tx0, sets r=b-Ax0, p=r, rr
    GLuint prog_cg_warm_init;
    GLint  u_cwi_m, u_cwi_n_partials, u_cwi_dt;
    // per-scene cached n_partials locations for inline-reduction shaders
    GLint  u_ci_n_partials;    // cg_init
    GLint  u_cjg_n_partials;   // cg_j_gather
    GLint  u_cuxr_n_partials;  // cg_update_xr
    /* Persistent coherent readback — eliminates glGetBufferSubData pipeline drain.
       ssbo_positions and ssbo_velocities are allocated with glBufferStorage so they
       can be persistently mapped; the CPU reads pos_map/vel_map directly after the
       readback_fence signals (checked non-blocking in simulator_sync_positions).    */
    float  *pos_map;          /* persistently-mapped ssbo_positions  (read-only) */
    float  *vel_map;          /* persistently-mapped ssbo_velocities (read-only) */
    GLsync  readback_fence;   /* fence planted at end of each simulator_gpu_step  */
    GLuint prog_draw_nodes;        // vert+frag: instanced sphere per node
    GLuint prog_draw_constraints;  // vert+frag: GL_LINES per constraint
    GLuint prog_draw_walls;        // vert+frag: GL_TRIANGLES from ssbo_triangles
    // sphere mesh (unit sphere, used with instancing)
    GLuint vao_sphere;
    GLuint vbo_sphere;
    GLuint ibo_sphere;
    int    sphere_index_count;
    // empty VAO for gl_VertexID-only draws (constraints, walls)
    GLuint vao_empty;
    // cached draw uniform locations
    GLint  u_draw_node_mvp;
    GLint  u_draw_con_mvp;
    GLint  u_draw_wall_mvp;
    // ── Timing ──────────────────────────────────────────────────────────────
    // Double-buffered GL_TIME_ELAPSED queries, one per major phase.
    // Phase indices: 0=lbvh_edges, 1=wall_bvh, 2=collision, 3=cg, 4=apply_corr, 5=particles
#define SIM_TQ_PHASES 6
    GLuint tq[2][SIM_TQ_PHASES]; // tq[frame&1][phase]
    int    tq_idx;               // which slot we are writing this frame
    int    tq_ready;             // 1 after the first completed frame
    SimTimings timings;          // last fully-completed frame's timings
} GPUContext;

// Pack the pointer-based simulator data into contiguous buffers for GPU upload.
// Non-spring constraints are placed first (indices 0..m_sparse-1); springs follow.
PackedScene* simulator_pack_scene(Simulator *s) {
    if (!s) return NULL;
    PackedScene *p = (PackedScene*)calloc(1, sizeof(PackedScene));
    int n = (int)dynarray_size(s->nodes);
    p->node_count = n;
    if (n > 0) {
        p->positions  = (float*)malloc(sizeof(float) * n * 3);
        p->velocities = (float*)malloc(sizeof(float) * n * 3);
        p->inv_mass   = (float*)malloc(sizeof(float) * n);
        p->radius     = (float*)malloc(sizeof(float) * n);
        p->flags      = (unsigned int*)malloc(sizeof(unsigned int) * n);
        for (int i = 0; i < n; ++i) {
            Node *nd = (Node*)dynarray_get(s->nodes, i);
            p->positions[3*i+0]  = nd->pos[0];
            p->positions[3*i+1]  = nd->pos[1];
            p->positions[3*i+2]  = nd->pos[2];
            p->velocities[3*i+0] = nd->vel[0];
            p->velocities[3*i+1] = nd->vel[1];
            p->velocities[3*i+2] = nd->vel[2];
            p->inv_mass[i] = (nd->anchored || nd->mass <= 0.0f) ? 0.0f : 1.0f / nd->mass;
            p->radius[i]   = nd->radius;
            unsigned int f = 0;
            if (nd->anchored)                f |= 1u;
            if (nd->collide_with_walls)         f |= 2u;
            if (nd->sim_ignore)                 f |= 4u;
            if (nd->isGravity)                  f |= 8u;  // bit 3
            if (nd->collide_when_anchored)      f |= 16u; // bit 4
            p->flags[i] = f;
        }
    }

    int m_all = (int)dynarray_size(s->constraints);
    p->constraint_count = m_all;
    // count non-spring constraints
    int m_sp = 0;
    for (int i = 0; i < m_all; i++) {
        Constraint *c = (Constraint*)dynarray_get(s->constraints, i);
        if (c && c->type != CT_SPRING) m_sp++;
    }
    p->m_sparse = m_sp;

    if (m_all > 0) {
        p->ctype      = (int*)  malloc(sizeof(int)   * m_all);
        p->a_idx      = (int*)  malloc(sizeof(int)   * m_all);
        p->b_idx      = (int*)  malloc(sizeof(int)   * m_all);
        p->rest       = (float*)malloc(sizeof(float) * m_all);
        p->stiffness  = (float*)malloc(sizeof(float) * m_all);
        p->anchor_pos = (float*)calloc(m_all * 3, sizeof(float));
        // pass 0: non-spring; pass 1: spring
        int dst = 0;
        for (int pass = 0; pass < 2; pass++) {
            for (int i = 0; i < m_all; i++) {
                Constraint *c = (Constraint*)dynarray_get(s->constraints, i);
                if (!c) continue;
                int is_spring = (c->type == CT_SPRING);
                if (pass == 0 &&  is_spring) continue;
                if (pass == 1 && !is_spring) continue;
                p->ctype[dst]     = (int)c->type;
                p->a_idx[dst]     = c->node  ? c->node->idx  : -1;
                p->b_idx[dst]     = c->other ? c->other->idx : -1;
                p->rest[dst]      = c->rest_length;
                p->stiffness[dst] = c->stiffness;
                float ax = 0, ay = 0, az = 0;
                constraint_get_anchor(c, &ax, &ay, &az);
                p->anchor_pos[3*dst+0] = ax;
                p->anchor_pos[3*dst+1] = ay;
                p->anchor_pos[3*dst+2] = az;
                dst++;
            }
        }
    }

    int tcount = (int)dynarray_size(s->walls);
    p->triangle_count = tcount;
    if (tcount > 0) {
        p->triangle_vertices = (float*)malloc(sizeof(float) * tcount * 9);
        for (int i = 0; i < tcount; ++i) {
            TriangleWall *w = (TriangleWall*)dynarray_get(s->walls, i);
            int base = i * 9;
            p->triangle_vertices[base+0] = w->A->pos[0]; p->triangle_vertices[base+1] = w->A->pos[1]; p->triangle_vertices[base+2] = w->A->pos[2];
            p->triangle_vertices[base+3] = w->B->pos[0]; p->triangle_vertices[base+4] = w->B->pos[1]; p->triangle_vertices[base+5] = w->B->pos[2];
            p->triangle_vertices[base+6] = w->C->pos[0]; p->triangle_vertices[base+7] = w->C->pos[1]; p->triangle_vertices[base+8] = w->C->pos[2];
        }
    }
    return p;
}

void simulator_free_packed_scene(PackedScene *p) {
    if (!p) return;
    free(p->positions); free(p->velocities); free(p->inv_mass);
    free(p->radius);    free(p->flags);
    free(p->ctype);     free(p->a_idx);      free(p->b_idx);
    free(p->rest);      free(p->stiffness);  free(p->anchor_pos);
    free(p->triangle_vertices);
    free(p);
}

// ── Flat BVH builder for static triangle walls ──────────────────────────────
// BVH nodes: 10 ints each  [min.xyz | left | max.xyz | right_or_first_tri | count | pad]
// Triangles: 12 ints each  [A.xyz  -1 | B.xyz  -1 | C.xyz  -1]
// The prims array is partitioned in-place; leaf indices refer to the reordered prim array.

static inline int fbits(float f) { int i; memcpy(&i, &f, 4); return i; }

typedef struct { float min[3]; float max[3]; int prim; } BVHPrim;
typedef struct { int *data; int size; int cap; } IntBuf;

static void ibuf_push10(IntBuf *b,
                        int v0,int v1,int v2,int v3,int v4,
                        int v5,int v6,int v7,int v8,int v9) {
    if (b->size + 10 > b->cap) {
        b->cap = (b->cap + 10 + 16) * 2;
        b->data = (int*)realloc(b->data, sizeof(int) * (size_t)b->cap);
    }
    b->data[b->size+0]=v0; b->data[b->size+1]=v1; b->data[b->size+2]=v2;
    b->data[b->size+3]=v3; b->data[b->size+4]=v4; b->data[b->size+5]=v5;
    b->data[b->size+6]=v6; b->data[b->size+7]=v7; b->data[b->size+8]=v8;
    b->data[b->size+9]=v9; b->size += 10;
}

static void bvh_bounds(BVHPrim *pr, int s, int e, float *mn, float *mx) {
    mn[0]=mn[1]=mn[2]= 1e30f; mx[0]=mx[1]=mx[2]=-1e30f;
    for (int i=s; i<e; i++)
        for (int d=0; d<3; d++) {
            if (pr[i].min[d] < mn[d]) mn[d] = pr[i].min[d];
            if (pr[i].max[d] > mx[d]) mx[d] = pr[i].max[d];
        }
}

// Returns index of the new node in buf (buf has 10 ints per node).
// After return, buf->data may have been reallocated; always use buf->data[idx*10+k].
static int bvh_build_rec(BVHPrim *pr, int start, int end, IntBuf *buf) {
    float mn[3], mx[3];
    bvh_bounds(pr, start, end, mn, mx);
    int node_idx = buf->size / 10;
    ibuf_push10(buf, fbits(mn[0]),fbits(mn[1]),fbits(mn[2]), -1,
                     fbits(mx[0]),fbits(mx[1]),fbits(mx[2]),  0, 0, 0);
    int count = end - start;
    if (count <= 4) {               // leaf
        buf->data[node_idx*10+3] = -1;      /* left=-1 marks leaf */
        buf->data[node_idx*10+7] = start;   /* first_tri */
        buf->data[node_idx*10+8] = count;   /* tri_count */
        return node_idx;
    }
    float ext[3] = { mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2] };
    int axis = (ext[1] > ext[0]) ? 1 : 0;
    if (ext[2] > ext[axis]) axis = 2;
    float mid = (mn[axis] + mx[axis]) * 0.5f;
    int l = start, r = end - 1;
    while (l <= r) {
        float cen = (pr[l].min[axis] + pr[l].max[axis]) * 0.5f;
        if (cen < mid) { l++; }
        else { BVHPrim tmp = pr[l]; pr[l] = pr[r]; pr[r--] = tmp; }
    }
    if (l == start || l == end) l = start + count / 2;
    int lc = bvh_build_rec(pr, start, l, buf);
    int rc = bvh_build_rec(pr, l,     end, buf);
    buf->data[node_idx*10+3] = lc;
    buf->data[node_idx*10+7] = rc;
    buf->data[node_idx*10+8] = 0;
    return node_idx;
}

// Helper: compile and link a compute shader from file, return program or 0 on error
static GLuint compile_compute_program_from_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    char *src = (char*)malloc(len + 1);
    fread(src, 1, len, f); src[len] = '\0'; fclose(f);
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, (const char**)&src, NULL);
    glCompileShader(sh);
    GLint ok = 0; glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[8192]; GLsizei l = 0; glGetShaderInfoLog(sh, sizeof(log), &l, log);
        fprintf(stderr, "Compute shader compile error (%s):\n%.*s\n", path, (int)l, log);
        free(src); glDeleteShader(sh); return 0;
    }
    free(src);
    GLuint prog = glCreateProgram(); glAttachShader(prog, sh); glLinkProgram(prog);
    GLint link_ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &link_ok);
    if (!link_ok) {
        char log[8192]; GLsizei l = 0; glGetProgramInfoLog(prog, sizeof(log), &l, log);
        fprintf(stderr, "Compute shader link error (%s):\n%.*s\n", path, (int)l, log);
        glDeleteShader(sh); glDeleteProgram(prog); return 0;
    }
    glDetachShader(prog, sh); glDeleteShader(sh);
    return prog;
}

/* Compile a vert+frag render program from two GLSL files. */
static GLuint compile_render_program_from_files(const char *vert_path, const char *frag_path) {
    /* read helper (local lambda via inline lambda) */
    GLuint sh[2] = {0, 0};
    const char *paths[2] = { vert_path, frag_path };
    const GLenum types[2] = { GL_VERTEX_SHADER, GL_FRAGMENT_SHADER };
    for (int i = 0; i < 2; i++) {
        FILE *f = fopen(paths[i], "rb");
        if (!f) { fprintf(stderr, "Render shader: cannot open %s\n", paths[i]); goto fail; }
        fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
        char *src = (char*)malloc(len + 1);
        fread(src, 1, len, f); src[len] = '\0'; fclose(f);
        sh[i] = glCreateShader(types[i]);
        glShaderSource(sh[i], 1, (const char**)&src, NULL);
        glCompileShader(sh[i]);
        free(src);
        GLint ok = 0; glGetShaderiv(sh[i], GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[8192]; GLsizei l = 0; glGetShaderInfoLog(sh[i], sizeof(log), &l, log);
            fprintf(stderr, "Render shader compile error (%s):\n%.*s\n", paths[i], (int)l, log);
            goto fail;
        }
    }
    {
        GLuint prog = glCreateProgram();
        glAttachShader(prog, sh[0]); glAttachShader(prog, sh[1]);
        glLinkProgram(prog);
        glDetachShader(prog, sh[0]); glDetachShader(prog, sh[1]);
        glDeleteShader(sh[0]); glDeleteShader(sh[1]);
        GLint link_ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &link_ok);
        if (!link_ok) {
            char log[8192]; GLsizei l = 0; glGetProgramInfoLog(prog, sizeof(log), &l, log);
            fprintf(stderr, "Render shader link error (%s + %s):\n%.*s\n", vert_path, frag_path, (int)l, log);
            glDeleteProgram(prog); return 0;
        }
        return prog;
    }
fail:
    if (sh[0]) glDeleteShader(sh[0]);
    if (sh[1]) glDeleteShader(sh[1]);
    return 0;
}

/* Column-major 4×4 matrix multiply: out = A * B */
static void mat4_mul_cm(float *out, const float *A, const float *B) {
    for (int col = 0; col < 4; col++) {
        for (int row = 0; row < 4; row++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) sum += A[k*4+row] * B[col*4+k];
            out[col*4+row] = sum;
        }
    }
}

/* Build a UV sphere VAO with the specified slices/stacks.
   Vertices store xyz = unit outward normal (== position on unit sphere).
   Attribute location 0 = vec3.
   Returns index count via *out_index_count. */
static void build_sphere_vao(int slices, int stacks,
                              GLuint *out_vao, GLuint *out_vbo, GLuint *out_ibo,
                              int *out_index_count) {
    int vert_count = (slices + 1) * (stacks + 1);
    float *verts = (float*)malloc(sizeof(float) * 3 * vert_count);
    int v = 0;
    for (int j = 0; j <= stacks; j++) {
        float phi = (float)M_PI * ((float)j / (float)stacks); /* 0..PI */
        float sin_phi = sinf(phi), cos_phi = cosf(phi);
        for (int i = 0; i <= slices; i++) {
            float theta = 2.0f * (float)M_PI * ((float)i / (float)slices);
            verts[v++] = sinf(theta) * sin_phi;
            verts[v++] = cos_phi;
            verts[v++] = cosf(theta) * sin_phi;
        }
    }
    int idx_count = slices * stacks * 6;
    unsigned short *idx = (unsigned short*)malloc(sizeof(unsigned short) * idx_count);
    int k = 0;
    for (int j = 0; j < stacks; j++) {
        for (int i = 0; i < slices; i++) {
            unsigned short a = (unsigned short)(j * (slices+1) + i);
            unsigned short b = (unsigned short)(a + slices + 1);
            idx[k++] = a;   idx[k++] = b;   idx[k++] = a+1;
            idx[k++] = a+1; idx[k++] = b;   idx[k++] = b+1;
        }
    }
    glGenVertexArrays(1, out_vao);
    glGenBuffers(1, out_vbo);
    glGenBuffers(1, out_ibo);
    glBindVertexArray(*out_vao);
    glBindBuffer(GL_ARRAY_BUFFER, *out_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*3*vert_count, verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *out_ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned short)*idx_count, idx, GL_STATIC_DRAW);
    glBindVertexArray(0);
    free(verts); free(idx);
    *out_index_count = idx_count;
}


// ── GPU lifecycle ─────────────────────────────────────────────────────────────

int simulator_init_gpu(Simulator *s) {
    if (!s) return -1;
    GLenum glew_status = glewInit();
    if (glew_status != GLEW_OK)
        fprintf(stderr, "GPU: GLEW init failed: %s\n", glewGetErrorString(glew_status));

    GPUContext *ctx = (GPUContext*)calloc(1, sizeof(GPUContext));
    ctx->enable_edge_edge = 1; /* on by default; disable for scenes with no cloth edges */

    ctx->prog_build_jb   = compile_compute_program_from_file("gpu/build_jb.comp.glsl");
    ctx->prog_ext_forces = compile_compute_program_from_file("gpu/ext_forces.comp.glsl");
    ctx->prog_apply_corr  = compile_compute_program_from_file("gpu/apply_corr.comp.glsl");
    ctx->prog_collision   = compile_compute_program_from_file("gpu/collision.comp.glsl");
    ctx->prog_edge_edge   = compile_compute_program_from_file("gpu/edge_edge.comp.glsl");
    // Sparse CG shaders
    ctx->prog_cg_init        = compile_compute_program_from_file("gpu/cg_init.comp.glsl");
    ctx->prog_cg_j_gather    = compile_compute_program_from_file("gpu/cg_j_gather.comp.glsl");
    ctx->prog_cg_update_xr   = compile_compute_program_from_file("gpu/cg_update_xr.comp.glsl");
    ctx->prog_cg_update_p    = compile_compute_program_from_file("gpu/cg_update_p.comp.glsl");
    ctx->prog_cg_jt_gather   = compile_compute_program_from_file("gpu/cg_jt_gather.comp.glsl");
    ctx->prog_bvh_walls   = compile_compute_program_from_file("gpu/lbvh_build_walls.comp.glsl");
    ctx->prog_bvh_edges   = compile_compute_program_from_file("gpu/lbvh_build_edges.comp.glsl");
    ctx->prog_lbvh_morton_walls = compile_compute_program_from_file("gpu/lbvh_morton_walls.comp.glsl");
    ctx->prog_lbvh_morton_edges = compile_compute_program_from_file("gpu/lbvh_morton_edges.comp.glsl");
    ctx->prog_lbvh_sort_walls   = compile_compute_program_from_file("gpu/lbvh_sort_walls.comp.glsl");
    ctx->prog_lbvh_sort_edges   = compile_compute_program_from_file("gpu/lbvh_sort_edges.comp.glsl");
    ctx->prog_lbvh_build_walls  = ctx->prog_bvh_walls;
    ctx->prog_lbvh_build_edges  = ctx->prog_bvh_edges;
    ctx->prog_lbvh_prepare_edge_dispatch = compile_compute_program_from_file(
        "gpu/lbvh_prepare_edge_dispatch.comp.glsl");
    ctx->prog_cg_jt_gather_inloop = compile_compute_program_from_file("gpu/cg_jt_gather.comp.glsl");
    ctx->prog_cg_warm_init         = compile_compute_program_from_file("gpu/cg_warm_init.comp.glsl");
    if (!ctx->prog_cg_warm_init)
        fprintf(stderr, "GPU: cg_warm_init failed to compile — warm start disabled.\n");
    else {
        ctx->u_cwi_m          = glGetUniformLocation(ctx->prog_cg_warm_init, "u_m");
        ctx->u_cwi_n_partials = glGetUniformLocation(ctx->prog_cg_warm_init, "u_n_partials");
        ctx->u_cwi_dt         = glGetUniformLocation(ctx->prog_cg_warm_init, "u_dt");
    }

    int ok = ctx->prog_build_jb && ctx->prog_ext_forces &&
             ctx->prog_apply_corr && ctx->prog_collision && ctx->prog_edge_edge &&
             ctx->prog_cg_init && ctx->prog_cg_j_gather &&
             ctx->prog_cg_update_xr && ctx->prog_cg_update_p &&
             ctx->prog_cg_jt_gather &&
             ctx->prog_lbvh_morton_walls && ctx->prog_lbvh_morton_edges &&
             ctx->prog_lbvh_sort_walls   && ctx->prog_lbvh_sort_edges   &&
             ctx->prog_lbvh_build_walls  && ctx->prog_lbvh_build_edges  &&
             ctx->prog_lbvh_prepare_edge_dispatch;
    if (!ok) fprintf(stderr, "GPU: one or more shaders failed to compile.\n");

    /* Cache uniform locations */
    ctx->u_ef_n        = glGetUniformLocation(ctx->prog_ext_forces, "u_n");
    ctx->u_ef_m_sparse = glGetUniformLocation(ctx->prog_ext_forces, "u_m_sparse");
    ctx->u_ef_m_total  = glGetUniformLocation(ctx->prog_ext_forces, "u_m_total");
    ctx->u_ef_gravity  = glGetUniformLocation(ctx->prog_ext_forces, "u_gravity");
    ctx->u_bjb_m       = glGetUniformLocation(ctx->prog_build_jb,   "u_m");
    ctx->u_ac_n        = glGetUniformLocation(ctx->prog_apply_corr, "u_n");
    ctx->u_ac_dt       = glGetUniformLocation(ctx->prog_apply_corr, "u_dt");
    ctx->u_ac_sub_dt   = glGetUniformLocation(ctx->prog_apply_corr, "u_sub_dt");
    ctx->u_ac_N        = glGetUniformLocation(ctx->prog_apply_corr, "u_N");
    // Sparse CG uniform locations
    ctx->u_ci_m           = glGetUniformLocation(ctx->prog_cg_init,       "u_m");
    ctx->u_cjg_m          = glGetUniformLocation(ctx->prog_cg_j_gather,   "u_m");
    ctx->u_cjg_dt         = glGetUniformLocation(ctx->prog_cg_j_gather,   "u_dt");
    ctx->u_cuxr_m         = glGetUniformLocation(ctx->prog_cg_update_xr,  "u_m");
    ctx->u_cup_m          = glGetUniformLocation(ctx->prog_cg_update_p,   "u_m");
    ctx->u_cjtg_n3        = glGetUniformLocation(ctx->prog_cg_jt_gather,  "u_n3");
    ctx->u_cjtg_src       = glGetUniformLocation(ctx->prog_cg_jt_gather,  "u_src");
    ctx->u_cjtg_apply_minv= glGetUniformLocation(ctx->prog_cg_jt_gather,  "u_apply_minv");
    // u_n_partials for inline-reduction shaders (uploaded once per scene load)
    ctx->u_ci_n_partials   = glGetUniformLocation(ctx->prog_cg_init,      "u_n_partials");
    ctx->u_cjg_n_partials  = glGetUniformLocation(ctx->prog_cg_j_gather,  "u_n_partials");
    ctx->u_cuxr_n_partials = glGetUniformLocation(ctx->prog_cg_update_xr, "u_n_partials");
    ctx->u_col_n        = glGetUniformLocation(ctx->prog_collision,  "u_n");
    ctx->u_col_n_tris   = glGetUniformLocation(ctx->prog_collision,  "u_n_tris");
    ctx->u_col_stiffness= glGetUniformLocation(ctx->prog_collision,  "u_stiffness");
    ctx->u_col_damp     = glGetUniformLocation(ctx->prog_collision,  "u_damp");
    ctx->u_col_fr       = glGetUniformLocation(ctx->prog_collision,  "u_friction_kt");
    ctx->u_ee_n           = glGetUniformLocation(ctx->prog_edge_edge, "u_n");
    ctx->u_ee_m_total     = glGetUniformLocation(ctx->prog_edge_edge, "u_m_total");
    ctx->u_ee_restitution = glGetUniformLocation(ctx->prog_edge_edge, "u_restitution");
    ctx->u_ee_mu          = glGetUniformLocation(ctx->prog_edge_edge, "u_mu");
    ctx->u_bvh_w_n_tris   = 0;  // unused (old serial builder removed)
    ctx->u_bvh_e_m_total  = 0;  // unused
    // LBVH Pass 1 uniform locations
    ctx->u_mw_n_prims   = glGetUniformLocation(ctx->prog_lbvh_morton_walls, "u_n_tris");
    ctx->u_mw_scene_min = glGetUniformLocation(ctx->prog_lbvh_morton_walls, "u_scene_min");
    ctx->u_mw_scene_max = glGetUniformLocation(ctx->prog_lbvh_morton_walls, "u_scene_max");
    ctx->u_me_m_total   = glGetUniformLocation(ctx->prog_lbvh_morton_edges, "u_m_total");
    ctx->u_me_scene_min = glGetUniformLocation(ctx->prog_lbvh_morton_edges, "u_scene_min");
    ctx->u_me_scene_max = glGetUniformLocation(ctx->prog_lbvh_morton_edges, "u_scene_max");
    // LBVH Pass 2 uniform locations
    ctx->u_sw_n_prims  = glGetUniformLocation(ctx->prog_lbvh_sort_walls, "u_n_prims");
    ctx->u_sw_pass_len = glGetUniformLocation(ctx->prog_lbvh_sort_walls, "u_pass_len");
    ctx->u_sw_sub_step = glGetUniformLocation(ctx->prog_lbvh_sort_walls, "u_sub_step");
    ctx->u_sw_local    = glGetUniformLocation(ctx->prog_lbvh_sort_walls, "u_local");
    ctx->u_se_n_prims  = glGetUniformLocation(ctx->prog_lbvh_sort_edges, "u_n_prims");
    ctx->u_se_pass_len = glGetUniformLocation(ctx->prog_lbvh_sort_edges, "u_pass_len");
    ctx->u_se_sub_step = glGetUniformLocation(ctx->prog_lbvh_sort_edges, "u_sub_step");
    ctx->u_se_local    = glGetUniformLocation(ctx->prog_lbvh_sort_edges, "u_local");
    // LBVH Pass 3 uniform locations
    ctx->u_bw_n_prims = glGetUniformLocation(ctx->prog_lbvh_build_walls, "u_n_prims");
    ctx->u_be_n_prims = glGetUniformLocation(ctx->prog_lbvh_build_edges, "u_n_prims");
    ctx->u_ped_m_total = glGetUniformLocation(ctx->prog_lbvh_prepare_edge_dispatch, "u_m_total");
    /* Bake src/minv into jt_gather variants */
    if (ctx->prog_cg_jt_gather_inloop) {
        glUseProgram(ctx->prog_cg_jt_gather_inloop);
        glUniform1i(glGetUniformLocation(ctx->prog_cg_jt_gather_inloop, "u_src"),        0);
        glUniform1i(glGetUniformLocation(ctx->prog_cg_jt_gather_inloop, "u_apply_minv"), 1);
        ctx->u_cjtg_il_n3 = glGetUniformLocation(ctx->prog_cg_jt_gather_inloop, "u_n3");
    }

    GLuint *ss[] = {
        &ctx->ssbo_positions, &ctx->ssbo_velocities, &ctx->ssbo_inv_mass,
        &ctx->ssbo_ext_forces, &ctx->ssbo_node_flags, &ctx->ssbo_constraints,
        &ctx->ssbo_J_cols, &ctx->ssbo_J_vals, &ctx->ssbo_A_dense,
        &ctx->ssbo_b_vec, &ctx->ssbo_l_vec, &ctx->ssbo_collision,
        &ctx->ssbo_r_vec, &ctx->ssbo_p_vec, &ctx->ssbo_Ap_vec,
        &ctx->ssbo_bvh_nodes, &ctx->ssbo_triangles, &ctx->ssbo_wall_indices,
        &ctx->ssbo_velcorr,
        &ctx->ssbo_edge_bvh, &ctx->ssbo_edge_bvh_scratch, &ctx->ssbo_wall_bvh_scratch,
        &ctx->ssbo_jt_vec, &ctx->ssbo_cg_scalars, &ctx->ssbo_reduce_buf,
        &ctx->ssbo_csr_offsets, &ctx->ssbo_csr_data,
        &ctx->ssbo_indirect_args, &ctx->ssbo_edge_meta
    };
    for (int i = 0; i < 29; i++) glGenBuffers(1, ss[i]);
    glGenBuffers(1, &ctx->ssbo_particle_mesh_forces);

    /* Render programs */
    ctx->prog_draw_nodes       = compile_render_program_from_files(
                                     "gpu/node_vert.glsl", "gpu/node_frag.glsl");
    ctx->prog_draw_constraints = compile_render_program_from_files(
                                     "gpu/constraint_vert.glsl", "gpu/constraint_frag.glsl");
    ctx->prog_draw_walls       = compile_render_program_from_files(
                                     "gpu/wall_vert.glsl", "gpu/wall_frag.glsl");
    if (!ctx->prog_draw_nodes || !ctx->prog_draw_constraints || !ctx->prog_draw_walls)
        fprintf(stderr, "GPU: one or more render shaders failed to compile.\n");

    ctx->u_draw_node_mvp = glGetUniformLocation(ctx->prog_draw_nodes,       "u_mvp");
    ctx->u_draw_con_mvp  = glGetUniformLocation(ctx->prog_draw_constraints, "u_mvp");
    ctx->u_draw_wall_mvp = glGetUniformLocation(ctx->prog_draw_walls,       "u_mvp");

    /* Sphere mesh (16 slices x 8 stacks) for node instanced draw */
    build_sphere_vao(16, 8,
                     &ctx->vao_sphere, &ctx->vbo_sphere, &ctx->ibo_sphere,
                     &ctx->sphere_index_count);

    /* Empty VAO for gl_VertexID-only draws */
    glGenVertexArrays(1, &ctx->vao_empty);

    /* Timer queries — double buffered, 6 phases */
    for (int f = 0; f < 2; f++)
        glGenQueries(SIM_TQ_PHASES, ctx->tq[f]);
    ctx->tq_idx   = 0;
    ctx->tq_ready = 0;

    s->gpu_ctx = ctx;
    s->use_gpu = 1;
    fprintf(stderr, "GPU: initialized (%s).\n", ok ? "all shaders OK" : "SHADER ERRORS");
    fflush(stderr);
    return ok ? 0 : -1;
}

void simulator_free_gpu(Simulator *s) {
    if (!s || !s->gpu_ctx) { if (s) s->use_gpu = 0; return; }
    GPUContext *ctx = (GPUContext*)s->gpu_ctx;
    if (ctx->prog_build_jb)   glDeleteProgram(ctx->prog_build_jb);
    if (ctx->prog_ext_forces)  glDeleteProgram(ctx->prog_ext_forces);
    if (ctx->prog_apply_corr)  glDeleteProgram(ctx->prog_apply_corr);
    if (ctx->prog_collision)   glDeleteProgram(ctx->prog_collision);
    if (ctx->prog_edge_edge)   glDeleteProgram(ctx->prog_edge_edge);
    // Sparse CG programs
    if (ctx->prog_cg_init)       glDeleteProgram(ctx->prog_cg_init);
    if (ctx->prog_cg_j_gather)   glDeleteProgram(ctx->prog_cg_j_gather);

    if (ctx->prog_cg_update_xr)  glDeleteProgram(ctx->prog_cg_update_xr);
    if (ctx->prog_cg_update_p)   glDeleteProgram(ctx->prog_cg_update_p);
    if (ctx->prog_cg_jt_gather)  glDeleteProgram(ctx->prog_cg_jt_gather);
    // LBVH programs (prog_bvh_walls/edges are aliases for build programs, don't double-free)
    if (ctx->prog_lbvh_morton_walls) glDeleteProgram(ctx->prog_lbvh_morton_walls);
    if (ctx->prog_lbvh_morton_edges) glDeleteProgram(ctx->prog_lbvh_morton_edges);
    if (ctx->prog_lbvh_sort_walls)   glDeleteProgram(ctx->prog_lbvh_sort_walls);
    if (ctx->prog_lbvh_sort_edges)   glDeleteProgram(ctx->prog_lbvh_sort_edges);
    if (ctx->prog_lbvh_build_walls)  glDeleteProgram(ctx->prog_lbvh_build_walls);
    if (ctx->prog_lbvh_build_edges)  glDeleteProgram(ctx->prog_lbvh_build_edges);
    if (ctx->prog_lbvh_prepare_edge_dispatch) glDeleteProgram(ctx->prog_lbvh_prepare_edge_dispatch);
    if (ctx->prog_cg_jt_gather_inloop) glDeleteProgram(ctx->prog_cg_jt_gather_inloop);
    if (ctx->prog_cg_warm_init)         glDeleteProgram(ctx->prog_cg_warm_init);
    if (ctx->prog_draw_nodes)       glDeleteProgram(ctx->prog_draw_nodes);
    if (ctx->prog_draw_constraints) glDeleteProgram(ctx->prog_draw_constraints);
    if (ctx->prog_draw_walls)       glDeleteProgram(ctx->prog_draw_walls);
    if (ctx->vao_sphere) glDeleteVertexArrays(1, &ctx->vao_sphere);
    if (ctx->vbo_sphere) glDeleteBuffers(1, &ctx->vbo_sphere);
    if (ctx->ibo_sphere) glDeleteBuffers(1, &ctx->ibo_sphere);
    if (ctx->vao_empty)  glDeleteVertexArrays(1, &ctx->vao_empty);
    /* Timer queries */
    for (int f = 0; f < 2; f++)
        glDeleteQueries(SIM_TQ_PHASES, ctx->tq[f]);
    /* Unmap persistent buffers before deletion (required by spec) */
    if (ctx->pos_map) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_positions);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        ctx->pos_map = NULL;
    }
    if (ctx->vel_map) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_velocities);
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        ctx->vel_map = NULL;
    }
    if (ctx->readback_fence) {
        glDeleteSync(ctx->readback_fence);
        ctx->readback_fence = NULL;
    }
    GLuint bufs[29] = {
        ctx->ssbo_positions, ctx->ssbo_velocities, ctx->ssbo_inv_mass,
        ctx->ssbo_ext_forces, ctx->ssbo_node_flags, ctx->ssbo_constraints,
        ctx->ssbo_J_cols, ctx->ssbo_J_vals, ctx->ssbo_A_dense,
        ctx->ssbo_b_vec, ctx->ssbo_l_vec, ctx->ssbo_collision,
        ctx->ssbo_r_vec, ctx->ssbo_p_vec, ctx->ssbo_Ap_vec,
        ctx->ssbo_bvh_nodes, ctx->ssbo_triangles, ctx->ssbo_wall_indices,
        ctx->ssbo_velcorr,
        ctx->ssbo_edge_bvh, ctx->ssbo_edge_bvh_scratch, ctx->ssbo_wall_bvh_scratch,
        ctx->ssbo_jt_vec, ctx->ssbo_cg_scalars, ctx->ssbo_reduce_buf,
        ctx->ssbo_csr_offsets, ctx->ssbo_csr_data,
        ctx->ssbo_indirect_args, ctx->ssbo_edge_meta
    };
    glDeleteBuffers(29, bufs);
    free(ctx);
    s->gpu_ctx = NULL;
    s->use_gpu = 0;
}

const SimTimings* simulator_get_timings(const Simulator *s) {
    if (!s || !s->gpu_ctx) return NULL;
    return &((const GPUContext*)s->gpu_ctx)->timings;
}

// ── LBVH sort helper ───────────────────────────────────────────────────────────
// Dispatches the bitonic sort for a scratch SSBO that has already been filled
// with n_prims entries (Morton code in slot [7]).
// Use u_local=1 for n<=1024 (single workgroup, 32KB shared mem), else 2-pass global bitonic.
#define LBVH_SORT_LOCAL_MAX 1024
static void lbvh_dispatch_sort(GLuint prog, GLint u_n, GLint u_pass, GLint u_step, GLint u_local,
                                int n_prims) {
    if (n_prims <= 0) return;
    glUseProgram(prog);
    glUniform1i(u_n, n_prims);

    if (n_prims <= LBVH_SORT_LOCAL_MAX) {
        // Single-workgroup shared-memory bitonic sort
        glUniform1i(u_local, 1);
        glUniform1i(u_pass,  0);
        glUniform1i(u_step,  0);
        glDispatchCompute(1, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        return;
    }

    // Global 2-pass bitonic sort: O(log^2 n) dispatch rounds.
    // Round up to next power of two.
    int n2 = 1;
    while (n2 < n_prims) n2 <<= 1;
    int half_n2   = n2 >> 1;
    int groups    = (half_n2 + 63) / 64;

    glUniform1i(u_local, 0);
    for (int len = 2; len <= n2; len <<= 1) {
        for (int step = len >> 1; step >= 1; step >>= 1) {
            glUniform1i(u_pass, len);
            glUniform1i(u_step, step);
            glDispatchCompute((GLuint)groups, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }
}

// ── LBVH 3-pass build helper ──────────────────────────────────────────────────
// Runs Morton → Sort → Build for walls.
// scene_min/max: tight AABB of ALL primitives (for Morton normalization).
// n_prims: number of triangles.
static void lbvh_build_walls_gpu(GPUContext *ctx, int n_prims,
                                  float *scene_min, float *scene_max) {
    if (n_prims <= 0) return;

    // Pass 1: Morton codes
    glUseProgram(ctx->prog_lbvh_morton_walls);
    glUniform1i(ctx->u_mw_n_prims,   n_prims);
    glUniform3f(ctx->u_mw_scene_min, scene_min[0], scene_min[1], scene_min[2]);
    glUniform3f(ctx->u_mw_scene_max, scene_max[0], scene_max[1], scene_max[2]);
    int mg = (n_prims + 63) / 64;
    glDispatchCompute((GLuint)mg, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Pass 2: Sort
    lbvh_dispatch_sort(ctx->prog_lbvh_sort_walls,
                       ctx->u_sw_n_prims, ctx->u_sw_pass_len,
                       ctx->u_sw_sub_step, ctx->u_sw_local,
                       n_prims);

    // Pass 3: Build
    glUseProgram(ctx->prog_lbvh_build_walls);
    glUniform1i(ctx->u_bw_n_prims, n_prims);
    int total = (n_prims == 1) ? 1 : (2 * n_prims - 1);
    int bg    = (total + 63) / 64;
    glDispatchCompute((GLuint)bg, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// ── LBVH 3-pass build helper ──────────────────────────────────────────────────
// Runs Morton → (GPU-prepare args) → Sort → Build for edges.
// No CPU readback.  The edge count and dispatch args are written GPU-side by
// prog_lbvh_prepare_edge_dispatch into ssbo_indirect_args (binding 28) and
// ssbo_edge_meta (binding 29).  Sort and Build use glDispatchComputeIndirect.
static void lbvh_build_edges_gpu(GPUContext *ctx, int m_total,
                                  float *scene_min, float *scene_max,
                                  int *n_edges_out) {
    if (m_total <= 0) { if (n_edges_out) *n_edges_out = 0; return; }

    /* Zero the atomic counter at escratch[m_total * 8].                         */
    /* glBufferSubData uploads 1 int without a GPU pipeline stall.               */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_edge_bvh_scratch);
    GLint zero_i = 0;
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, (GLintptr)(sizeof(int) * (size_t)m_total * 8),
                    sizeof(GLint), &zero_i);

    /* Pass 1: Morton + compact ─────────────────────────────────────────────── */
    glUseProgram(ctx->prog_lbvh_morton_edges);
    glUniform1i(ctx->u_me_m_total, m_total);
    /* scene_min/max are pre-uploaded at scene load time (ctx->scene_min/max) */
    int mg = (m_total + 63) / 64;
    glDispatchCompute((GLuint)mg, 1, 1);
    /* Barrier: morton writes escratch[] and the atomic counter.                 */
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /* Pass 1.5: Prepare indirect dispatch args (GPU-side, no CPU readback) ─── */
    /* Reads escratch[m_total*8] → writes ssbo_indirect_args[0..5] + edge_meta[0] */
    glUseProgram(ctx->prog_lbvh_prepare_edge_dispatch);
    glUniform1i(ctx->u_ped_m_total, m_total);
    glDispatchCompute(1, 1, 1);
    /* GL_COMMAND_BARRIER_BIT: makes indirect_args visible to glDispatchComputeIndirect */
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_COMMAND_BARRIER_BIT);

    /* Pass 2: Sort (indirect) ─────────────────────────────────────────────── */
    /* The sort shader reads n_prims from edge_meta[0]; indirect_args[0..2] = {sort_x,1,1} */
    glUseProgram(ctx->prog_lbvh_sort_edges);
    /* u_n_prims=-1 → shader reads n from edge_meta[0] */
    glUniform1i(ctx->u_se_n_prims, -1);
    glUniform1i(ctx->u_se_local,    1);   /* always try local first; shader checks n */
    glUniform1i(ctx->u_se_pass_len, 0);
    glUniform1i(ctx->u_se_sub_step, 0);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, ctx->ssbo_indirect_args);
    glDispatchComputeIndirect(0);  /* sort_x at offset 0 */
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /* Pass 3: Build (indirect) ─────────────────────────────────────────────── */
    /* u_n_prims=-1 → shader reads n from edge_meta[0] */
    glUseProgram(ctx->prog_lbvh_build_edges);
    glUniform1i(ctx->u_be_n_prims, -1);
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, ctx->ssbo_indirect_args);
    glDispatchComputeIndirect(12); /* build_x at byte offset 12 (second uvec3) */
    glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    /* n_edges_out is no longer available without a stall.  Callers that only   */
    /* used it for ctx->last_n_edges (informational) get 0 here; that field is  */
    /* now stale but not used for any logic.                                     */
    if (n_edges_out) *n_edges_out = 0;
}

// Upload static scene data (positions/velocities/inv_mass/flags/constraints) and
// allocate the dynamic per-substep SSBOs (J, A, b, l, r, p, Ap, collision).
void simulator_upload_scene_to_gpu(Simulator *s, PackedScene *p) {
    if (!s || !p || !s->gpu_ctx) return;
    GPUContext *ctx = (GPUContext*)s->gpu_ctx;
    int n       = p->node_count;
    int m       = p->m_sparse;
    int m_total = p->constraint_count;
    ctx->node_count  = n;
    ctx->m_sparse    = m;
    ctx->m_total     = m_total;
    ctx->last_n_edges = 0;
    /* Default generous scene AABB (overwritten below if walls are present) */
    ctx->scene_min[0] = ctx->scene_min[1] = ctx->scene_min[2] = -100.0f;
    ctx->scene_max[0] = ctx->scene_max[1] = ctx->scene_max[2] =  100.0f;

#define _BIND(b, ssbo) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, (b), (ssbo))

    if (n > 0) {
        // Positions/velocities as vec4 arrays
        float *pos4 = (float*)calloc((size_t)n * 4, sizeof(float));
        float *vel4 = (float*)calloc((size_t)n * 4, sizeof(float));
        for (int i = 0; i < n; i++) {
            pos4[4*i+0]=p->positions[3*i+0]; pos4[4*i+1]=p->positions[3*i+1]; pos4[4*i+2]=p->positions[3*i+2]; pos4[4*i+3]=p->radius[i];
            vel4[4*i+0]=p->velocities[3*i+0]; vel4[4*i+1]=p->velocities[3*i+1]; vel4[4*i+2]=p->velocities[3*i+2];
        }
        /* Allocate positions/velocities with immutable persistent+coherent storage so
           simulator_sync_positions can read them without ever stalling the pipeline.
           glBufferStorage is one-shot per buffer object: unmap, delete, recreate.   */
        if (ctx->pos_map) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_positions);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            ctx->pos_map = NULL;
        }
        if (ctx->vel_map) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_velocities);
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            ctx->vel_map = NULL;
        }
        /* Delete + recreate so we can call glBufferStorage on a fresh object */
        glDeleteBuffers(1, &ctx->ssbo_positions);  glGenBuffers(1, &ctx->ssbo_positions);
        glDeleteBuffers(1, &ctx->ssbo_velocities); glGenBuffers(1, &ctx->ssbo_velocities);
        GLbitfield stor_flags = GL_MAP_PERSISTENT_BIT | GL_MAP_READ_BIT
                              | GL_MAP_COHERENT_BIT   | GL_DYNAMIC_STORAGE_BIT;
        GLbitfield map_flags  = GL_MAP_PERSISTENT_BIT | GL_MAP_READ_BIT | GL_MAP_COHERENT_BIT;
        GLsizeiptr pv_sz = (GLsizeiptr)(sizeof(float) * (size_t)n * 4);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_positions);
        glBufferStorage(GL_SHADER_STORAGE_BUFFER, pv_sz, pos4, stor_flags);
        ctx->pos_map = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, pv_sz, map_flags);
        _BIND(0, ctx->ssbo_positions);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_velocities);
        glBufferStorage(GL_SHADER_STORAGE_BUFFER, pv_sz, vel4, stor_flags);
        ctx->vel_map = (float*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, pv_sz, map_flags);
        _BIND(1, ctx->ssbo_velocities);
        free(pos4); free(vel4);
        if (!ctx->pos_map || !ctx->vel_map)
            fprintf(stderr, "GPU: persistent map failed — sync_positions will use slow path\n");

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_inv_mass);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float)*n, p->inv_mass, GL_STATIC_DRAW);
        _BIND(2, ctx->ssbo_inv_mass);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_ext_forces);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float)*n*4, NULL, GL_DYNAMIC_DRAW);
        _BIND(3, ctx->ssbo_ext_forces);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_node_flags);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(unsigned int)*n, p->flags, GL_STATIC_DRAW);
        _BIND(4, ctx->ssbo_node_flags);

        // collision forces: zeroed, CPU writes each substep
        float *zeros = (float*)calloc((size_t)n*4, sizeof(float));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_collision);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float)*n*4, zeros, GL_DYNAMIC_DRAW);
        _BIND(11, ctx->ssbo_collision);

        // particle-mesh forces: zeroed, written by ParticleSim each substep
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_particle_mesh_forces);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float)*n*4, zeros, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 39, ctx->ssbo_particle_mesh_forces);

        free(zeros);
    }

    // Constraint SSBO: 8 ints per entry (non-spring first, then springs)
    // Layout: [ctype, a_idx, b_idx, float_rest, float_ax, float_ay, float_az, float_stiffness]
    if (m_total > 0) {
        int *con = (int*)malloc(sizeof(int) * m_total * 8);
        for (int i = 0; i < m_total; i++) {
            con[i*8+0] = p->ctype[i];
            con[i*8+1] = p->a_idx[i];
            con[i*8+2] = p->b_idx[i];
            float v;
            v = p->rest[i];            memcpy(&con[i*8+3], &v, 4);
            v = p->anchor_pos[3*i+0];  memcpy(&con[i*8+4], &v, 4);
            v = p->anchor_pos[3*i+1];  memcpy(&con[i*8+5], &v, 4);
            v = p->anchor_pos[3*i+2];  memcpy(&con[i*8+6], &v, 4);
            v = p->stiffness[i];       memcpy(&con[i*8+7], &v, 4);
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_constraints);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int)*m_total*8, con, GL_STATIC_DRAW);
        _BIND(5, ctx->ssbo_constraints);
        free(con);
    }

    // Dynamic per-substep SSBOs (allocated now, filled by compute shaders each substep)
    if (m > 0) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_J_cols);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int)*m*6, NULL, GL_DYNAMIC_DRAW);
        _BIND(6, ctx->ssbo_J_cols);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_J_vals);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float)*m*6, NULL, GL_DYNAMIC_DRAW);
        _BIND(7, ctx->ssbo_J_vals);
        // A_dense: stub (1 float) — build_A is not dispatched in the sparse CG path
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_A_dense);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float), NULL, GL_DYNAMIC_DRAW);
        _BIND(8, ctx->ssbo_A_dense);
        int bindings[] = { 9, 10, 12, 13, 14 };
        GLuint *vbufs[] = { &ctx->ssbo_b_vec, &ctx->ssbo_l_vec, &ctx->ssbo_r_vec, &ctx->ssbo_p_vec, &ctx->ssbo_Ap_vec };
        for (int i = 0; i < 5; i++) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, *vbufs[i]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float)*m, NULL, GL_DYNAMIC_DRAW);
            _BIND(bindings[i], *vbufs[i]);
        }
        // Sparse CG SSBOs
        int n_partials = (m + 63) / 64;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_jt_vec);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float)*(size_t)n*3, NULL, GL_DYNAMIC_DRAW);
        _BIND(23, ctx->ssbo_jt_vec);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_cg_scalars);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float)*8 + sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
        _BIND(24, ctx->ssbo_cg_scalars);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_reduce_buf);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float)*(size_t)n_partials, NULL, GL_DYNAMIC_DRAW);
        _BIND(25, ctx->ssbo_reduce_buf);

        /* Build static J^T CSR (topology fixed at upload; only J_vals change per substep).
           For each DOF d = 3*node+dim, store list of (c, k_slot) pairs where
           J_cols[c*6+k] == d.  CT_DIST=1: slots 0-2 for a_idx, 3-5 for b_idx if not anchored.
           CT_ANCHOR=3: slots 0-2 for a_idx only. */
        {
            /* max 6 entries per constraint */
            int *tmp = (int*)malloc(sizeof(int) * m * 6 * 3); /* (dof,c,k) triples */
            int n_ent = 0;
            for (int c = 0; c < m; c++) {
                int ctype = p->ctype[c];
                int a_idx = p->a_idx[c];
                int b_idx = p->b_idx[c];
                if (a_idx >= 0 && (ctype == 1 || ctype == 3)) {
                    for (int d = 0; d < 3; d++) {
                        tmp[n_ent*3+0] = 3*a_idx+d; tmp[n_ent*3+1] = c; tmp[n_ent*3+2] = d;
                        n_ent++;
                    }
                }
                if (ctype == 1 && b_idx >= 0 && !(p->flags[b_idx] & 1u)) {
                    for (int d = 0; d < 3; d++) {
                        tmp[n_ent*3+0] = 3*b_idx+d; tmp[n_ent*3+1] = c; tmp[n_ent*3+2] = 3+d;
                        n_ent++;
                    }
                }
            }
            ctx->n_csr_entries = n_ent;
            int dof_count = 3 * n;
            /* counting sort by dof */
            int *cnts = (int*)calloc((size_t)dof_count, sizeof(int));
            for (int e = 0; e < n_ent; e++) cnts[tmp[e*3+0]]++;
            int *off = (int*)malloc(sizeof(int) * ((size_t)dof_count + 1));
            off[0] = 0;
            for (int d = 0; d < dof_count; d++) off[d+1] = off[d] + cnts[d];
            int *csr_d = (int*)(n_ent > 0 ? malloc(sizeof(int)*n_ent*2) : malloc(sizeof(int)));
            int *pos   = (int*)calloc((size_t)dof_count, sizeof(int));
            for (int e = 0; e < n_ent; e++) {
                int dof = tmp[e*3+0];
                int idx = off[dof] + pos[dof]++;
                csr_d[idx*2+0] = tmp[e*3+1]; /* c */
                csr_d[idx*2+1] = tmp[e*3+2]; /* k */
            }
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_csr_offsets);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int)*((size_t)dof_count+1), off, GL_STATIC_DRAW);
            _BIND(26, ctx->ssbo_csr_offsets);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_csr_data);
            glBufferData(GL_SHADER_STORAGE_BUFFER, n_ent > 0 ? sizeof(int)*n_ent*2 : sizeof(int), csr_d, GL_STATIC_DRAW);
            _BIND(27, ctx->ssbo_csr_data);
            free(tmp); free(cnts); free(off); free(csr_d); free(pos);
        }

        /* ssbo_indirect_args (binding 28): 6 uints = 2x DispatchIndirectCommand.
           Initialise to {1,1,1, 1,1,1} — safe default (shader guards on n<=0). */
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_indirect_args);
        static const GLuint init_args[6] = {1,1,1, 1,1,1};
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(init_args), init_args, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 28, ctx->ssbo_indirect_args);

        /* ssbo_edge_meta (binding 29): 2 ints [n_edges, n_wall_prims(unused)].
           Initialise to {0, n_wall_prims} — n_wall_prims is filled once at upload. */
        int edge_meta_init[2] = {0, 0};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_edge_meta);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(edge_meta_init), edge_meta_init, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 29, ctx->ssbo_edge_meta);
    }
#undef _BIND

    /* --- Wall triangle index SSBO (binding 17) + LBVH build (bindings 15, 22) --- */
    {
        int tcount = (int)dynarray_size(s->walls);
        ctx->n_triangles = tcount;
        if (tcount > 0) {
            /* Upload wall node indices in original scene order (not sorted).
               The LBVH morton shader reads SSBO 17 + SSBO 0 → SSBO 22 scratch.
               SSBO 17 stays in scene order (required by wall_vert). */
            int *widx_data = (int*)malloc(sizeof(int) * 4 * (size_t)tcount);
            /* Compute scene AABB on CPU for Morton normalization */
            float scene_min[3] = { 1e30f,  1e30f,  1e30f};
            float scene_max[3] = {-1e30f, -1e30f, -1e30f};
            for (int i = 0; i < tcount; i++) {
                TriangleWall *w = (TriangleWall*)dynarray_get(s->walls, i);
                widx_data[4*i+0] = w->A->idx;
                widx_data[4*i+1] = w->B->idx;
                widx_data[4*i+2] = w->C->idx;
                widx_data[4*i+3] = w->translucent ? 1 : 0;
                Node *verts[3] = {w->A, w->B, w->C};
                for (int v = 0; v < 3; v++) {
                    for (int d = 0; d < 3; d++) {
                        if (verts[v]->pos[d] < scene_min[d]) scene_min[d] = verts[v]->pos[d];
                        if (verts[v]->pos[d] > scene_max[d]) scene_max[d] = verts[v]->pos[d];
                    }
                }
            }
            /* Pad scene AABB slightly so edge prims at the boundary get valid codes */
            for (int d = 0; d < 3; d++) { scene_min[d] -= 0.1f; scene_max[d] += 0.1f; }

            /* Store in ctx for per-substep edge build (generous bounds) */
            for (int d = 0; d < 3; d++) {
                ctx->scene_min[d] = scene_min[d] - 10.0f;
                ctx->scene_max[d] = scene_max[d] + 10.0f;
            }

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_wall_indices);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int)*4*(size_t)tcount, widx_data, GL_STATIC_DRAW);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 17, ctx->ssbo_wall_indices);
            free(widx_data);

            /* Allocate BVH node buffer (binding 15): 2*tcount nodes (LBVH: n-1 internal + n leaf) */
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_bvh_nodes);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int)*10*2*(size_t)tcount, NULL, GL_DYNAMIC_DRAW);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, ctx->ssbo_bvh_nodes);

            /* Allocate wall BVH scratch (binding 22): 8 ints/prim */
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_wall_bvh_scratch);
            glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int)*8*(size_t)tcount, NULL, GL_DYNAMIC_DRAW);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 22, ctx->ssbo_wall_bvh_scratch);

            /* Dispatch LBVH 3-pass pipeline (walls are static — built once) */
            lbvh_build_walls_gpu(ctx, tcount, scene_min, scene_max);
            fprintf(stderr, "GPU: wall LBVH built on GPU (%d tris, %d nodes).\n",
                    tcount, (tcount == 1) ? 1 : (2*tcount - 1));
        }
    }

    /* --- ssbo_velcorr (binding 18): edge-edge velocity impulse accumulator --- */
    if (n > 0) {
        static const GLuint zero = 0u;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_velcorr);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLuint) * (size_t)n * 3, NULL, GL_DYNAMIC_DRAW);
        glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 18, ctx->ssbo_velcorr);
    }

    /* --- edge BVH SSBOs (bindings 20-21): allocated once, rebuilt GPU-side each substep ---
       Scratch needs one extra int at offset [m_total*8] for the atomic compact counter. */
    if (m_total > 0) {
        /* worst case: 2*m_total BVH nodes */
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_edge_bvh);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int)*10*2*(size_t)m_total, NULL, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20, ctx->ssbo_edge_bvh);

        /* scratch: 8 ints/prim + 1 extra int for atomic counter at slot [m_total*8] */
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_edge_bvh_scratch);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(int)*((size_t)m_total*8 + 1), NULL, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 21, ctx->ssbo_edge_bvh_scratch);
    }

    fprintf(stderr, "GPU: scene uploaded (n=%d, m_sparse=%d, m_total=%d).\n", n, m, m_total);

    /* Pre-upload scene_min/max to the edge morton shader so lbvh_build_edges_gpu
       does not re-upload them every substep. */
    if (ctx->prog_lbvh_morton_edges) {
        glUseProgram(ctx->prog_lbvh_morton_edges);
        glUniform3f(ctx->u_me_scene_min, ctx->scene_min[0], ctx->scene_min[1], ctx->scene_min[2]);
        glUniform3f(ctx->u_me_scene_max, ctx->scene_max[0], ctx->scene_max[1], ctx->scene_max[2]);
    }
    fflush(stderr);
}

void simulator_enable_gpu(Simulator *s, int enable) {
    if (!s) return;
    if (enable) {
        if (!s->use_gpu) { simulator_init_gpu(s); s->use_gpu = 1; }
    } else {
        if (s->use_gpu) { simulator_free_gpu(s); s->use_gpu = 0; }
    }
}
// Check collision between two distance constraints (modeled as cylinders)
// Returns 1 if collision detected, fills out_penetration, out_normal[3], and closest points on each segment
static int constraint_constraint_collision(
    Node *a0, Node *a1, // endpoints of first constraint
    Node *b0, Node *b1, // endpoints of second constraint
    float *out_penetration, float out_normal[3],
    float out_pa[3], float out_pb[3] // closest points on each segment
) {
    // Compute segment vectors and lengths
    float A[3] = {a0->pos[0], a0->pos[1], a0->pos[2]};
    float B[3] = {a1->pos[0], a1->pos[1], a1->pos[2]};
    float C[3] = {b0->pos[0], b0->pos[1], b0->pos[2]};
    float D[3] = {b1->pos[0], b1->pos[1], b1->pos[2]};
    float u[3] = {B[0]-A[0], B[1]-A[1], B[2]-A[2]};
    float v[3] = {D[0]-C[0], D[1]-C[1], D[2]-C[2]};
    float w[3] = {A[0]-C[0], A[1]-C[1], A[2]-C[2]};
    float len_u = sqrtf(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
    float len_v = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (len_u < 1e-8f || len_v < 1e-8f) return 0; // degenerate

    float ru = 0.01f * len_u;
    float rv = 0.01f * len_v;

    // Compute closest points between segments (see Real-Time Collision Detection, Christer Ericson)
    float a = u[0]*u[0] + u[1]*u[1] + u[2]*u[2];
    float b = u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
    float c = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    float d = u[0]*w[0] + u[1]*w[1] + u[2]*w[2];
    float e = v[0]*w[0] + v[1]*w[1] + v[2]*w[2];
    float D_ = a*c - b*b;
    float sc, sN, sD = D_;
    float tc, tN, tD = D_;

    // Default sN = D_, tN = D_
    if (D_ < 1e-8f) {
        sN = 0.0f;
        sD = 1.0f;
        tN = e;
        tD = c;
    } else {
        sN = (b*e - c*d);
        tN = (a*e - b*d);
        if (sN < 0.0f) { sN = 0.0f; tN = e; tD = c; }
        else if (sN > sD) { sN = sD; tN = e + b; tD = c; }
    }
    if (tN < 0.0f) { tN = 0.0f;
        if (-d < 0.0f) sN = 0.0f;
        else if (-d > a) sN = sD;
        else { sN = -d; sD = a; }
    } else if (tN > tD) { tN = tD;
        if ((-d + b) < 0.0f) sN = 0.0f;
        else if ((-d + b) > a) sN = sD;
        else { sN = (-d + b); sD = a; }
    }
    sc = (fabsf(sN) < 1e-8f ? 0.0f : sN / sD);
    tc = (fabsf(tN) < 1e-8f ? 0.0f : tN / tD);

    // Closest points
    float pa[3] = {A[0] + sc * u[0], A[1] + sc * u[1], A[2] + sc * u[2]};
    float pb[3] = {C[0] + tc * v[0], C[1] + tc * v[1], C[2] + tc * v[2]};
    float dx = pa[0] - pb[0], dy = pa[1] - pb[1], dz = pa[2] - pb[2];
    float dist = sqrtf(dx*dx + dy*dy + dz*dz);

    float min_dist = ru + rv;
    if (dist < min_dist) {
        if (out_penetration) *out_penetration = min_dist - dist;
        if (out_normal) {
            float nlen = sqrtf(dx*dx + dy*dy + dz*dz);
            if (nlen > 1e-8f) {
                out_normal[0] = dx / nlen;
                out_normal[1] = dy / nlen;
                out_normal[2] = dz / nlen;
            } else {
                out_normal[0] = 1.0f; out_normal[1] = 0.0f; out_normal[2] = 0.0f;
            }
        }
        if (out_pa) { out_pa[0] = pa[0]; out_pa[1] = pa[1]; out_pa[2] = pa[2]; }
        if (out_pb) { out_pb[0] = pb[0]; out_pb[1] = pb[1]; out_pb[2] = pb[2]; }
        return 1;
    }
    return 0;
}


// --- Edge-Edge Collision Detection and Resolution using EdgeBVH self-traversal ---
typedef struct {
    DynArray *edges;
    float *collision_forces;
    float sub_dt;
} EdgeEdgeContext;

void edge_edge_callback(int idxA, int idxB, void *userdata) {
    float DAMP_K = 200.0f;
    float PENETRATION_STIFFNESS = 500.0f;

    EdgeEdgeContext *ctx = (EdgeEdgeContext*)userdata;
    DynArray *edges = ctx->edges;
    float *collision_forces = ctx->collision_forces;
    Constraint *edgeA = (Constraint*)dynarray_get(edges, idxA);
    Constraint *edgeB = (Constraint*)dynarray_get(edges, idxB);
    if (!edgeA || !edgeB || !edgeA->node || !edgeA->other || !edgeB->node || !edgeB->other) return;
    // Skip if constraints share a node as an endpoint
    if (edgeA->node == edgeB->node || edgeA->node == edgeB->other ||
        edgeA->other == edgeB->node || edgeA->other == edgeB->other) return;
    // Only check distance constraints
    if (edgeA->type != CT_DIST || edgeB->type != CT_DIST) return;

    float penetration, normal[3], pa[3], pb[3];
    if (!constraint_constraint_collision(edgeA->node, edgeA->other, edgeB->node, edgeB->other, &penetration, normal, pa, pb)) {
        return;
    }

    // Compute barycentric-like weights along each edge for velocity interpolation
    float dA0 = sqrtf((pa[0]-edgeA->node->pos[0])*(pa[0]-edgeA->node->pos[0]) +
                      (pa[1]-edgeA->node->pos[1])*(pa[1]-edgeA->node->pos[1]) +
                      (pa[2]-edgeA->node->pos[2])*(pa[2]-edgeA->node->pos[2]));
    float dA1 = sqrtf((pa[0]-edgeA->other->pos[0])*(pa[0]-edgeA->other->pos[0]) +
                      (pa[1]-edgeA->other->pos[1])*(pa[1]-edgeA->other->pos[1]) +
                      (pa[2]-edgeA->other->pos[2])*(pa[2]-edgeA->other->pos[2]));
    float lenA = dA0 + dA1;
    float wA0 = (lenA > 1e-8f) ? (dA1 / lenA) : 0.5f;
    float wA1 = (lenA > 1e-8f) ? (dA0 / lenA) : 0.5f;

    float dB0 = sqrtf((pb[0]-edgeB->node->pos[0])*(pb[0]-edgeB->node->pos[0]) +
                      (pb[1]-edgeB->node->pos[1])*(pb[1]-edgeB->node->pos[1]) +
                      (pb[2]-edgeB->node->pos[2])*(pb[2]-edgeB->node->pos[2]));
    float dB1 = sqrtf((pb[0]-edgeB->other->pos[0])*(pb[0]-edgeB->other->pos[0]) +
                      (pb[1]-edgeB->other->pos[1])*(pb[1]-edgeB->other->pos[1]) +
                      (pb[2]-edgeB->other->pos[2])*(pb[2]-edgeB->other->pos[2]));
    float lenB = dB0 + dB1;
    float wB0 = (lenB > 1e-8f) ? (dB1 / lenB) : 0.5f;
    float wB1 = (lenB > 1e-8f) ? (dB0 / lenB) : 0.5f;

    // printf("Edge-Edge Collision: idxA=%d idxB=%d\n", idxA, idxB);
    // printf("  Penetration: %.4f\n", penetration);
    // printf("  Normal: [%.4f %.4f %.4f]\n", normal[0], normal[1], normal[2]);
    // printf("  Closest points: pa=[%.4f %.4f %.4f], pb=[%.4f %.4f %.4f]\n", pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]);

    // Interpolated velocities at contact points
    float va[3], vb[3];
    for (int i = 0; i < 3; ++i) {
        va[i] = edgeA->node->vel[i] * wA0 + edgeA->other->vel[i] * wA1;
        vb[i] = edgeB->node->vel[i] * wB0 + edgeB->other->vel[i] * wB1;
    }

    float rel_vel[3] = { va[0] - vb[0], va[1] - vb[1], va[2] - vb[2] };
    float normal_vel = rel_vel[0]*normal[0] + rel_vel[1]*normal[1] + rel_vel[2]*normal[2];
    // printf("  Relative velocity: [%.4f %.4f %.4f], normal_vel=%.4f\n", rel_vel[0], rel_vel[1], rel_vel[2], normal_vel);

    // Impulse-based resolution to avoid tunneling: compute effective inverse-mass
    float inv_mA0 = (edgeA->node->anchored || edgeA->node->sim_ignore) ? 0.0f : 1.0f / fmaxf(edgeA->node->mass, 1e-9f);
    float inv_mA1 = (edgeA->other->anchored || edgeA->other->sim_ignore) ? 0.0f : 1.0f / fmaxf(edgeA->other->mass, 1e-9f);
    float inv_mB0 = (edgeB->node->anchored || edgeB->node->sim_ignore) ? 0.0f : 1.0f / fmaxf(edgeB->node->mass, 1e-9f);
    float inv_mB1 = (edgeB->other->anchored || edgeB->other->sim_ignore) ? 0.0f : 1.0f / fmaxf(edgeB->other->mass, 1e-9f);

    float weff = wA0*wA0*inv_mA0 + wA1*wA1*inv_mA1 + wB0*wB0*inv_mB0 + wB1*wB1*inv_mB1;
    if (weff <= 1e-12f) return;

    // Only resolve contacts that are closing (negative relative normal velocity)
    // Note: earlier code used normal_vel positive for closing; flip sign accordingly
    if (normal_vel >= 0.0f) return;

    float restitution = 0.05f; // small restitution to avoid bounciness
    float J = -(1.0f + restitution) * normal_vel / weff;
    // clamp impulse magnitude to avoid extreme corrections
    float maxJ = 1e4f;
    if (J > maxJ) J = maxJ;

    // Tangential (Coulomb) friction impulse
    float rel_t[3] = {
        rel_vel[0] - normal_vel * normal[0],
        rel_vel[1] - normal_vel * normal[1],
        rel_vel[2] - normal_vel * normal[2]
    };
    float t_len = sqrtf(rel_t[0]*rel_t[0] + rel_t[1]*rel_t[1] + rel_t[2]*rel_t[2]);
    float jt = 0.0f;
    float t_dir[3] = {0.0f, 0.0f, 0.0f};
    if (t_len > 1e-9f) {
        t_dir[0] = rel_t[0] / t_len;
        t_dir[1] = rel_t[1] / t_len;
        t_dir[2] = rel_t[2] / t_len;
        // desired tangential impulse to remove relative tangential velocity
        jt = -(rel_vel[0]*t_dir[0] + rel_vel[1]*t_dir[1] + rel_vel[2]*t_dir[2]) / weff;
    }

    // friction coefficient: average of involved node frictions (fallback 0.5)
    float mu_sum = 0.0f; int mu_count = 0;
    if (edgeA->node) { mu_sum += edgeA->node->friction; mu_count++; }
    if (edgeA->other) { mu_sum += edgeA->other->friction; mu_count++; }
    if (edgeB->node) { mu_sum += edgeB->node->friction; mu_count++; }
    if (edgeB->other) { mu_sum += edgeB->other->friction; mu_count++; }
    float mu = (mu_count > 0) ? (mu_sum / (float)mu_count) : 0.5f;

    // clamp tangential impulse by Coulomb: |jt| <= mu * J
    float jmax = fabsf(mu * J);
    if (jt > jmax) jt = jmax;
    if (jt < -jmax) jt = -jmax;

    // printf("  Impulse J=%.6f jt=%.6f weff=%.6e mu=%.3f\n", J, jt, weff, mu);

    // Apply velocity impulse (normal + tangential) distributed to nodes
    if (inv_mA0 > 0.0f) {
        for (int k = 0; k < 3; ++k) edgeA->node->vel[k] += (J * wA0 * inv_mA0) * normal[k] + (jt * wA0 * inv_mA0) * t_dir[k];
    }
    if (inv_mA1 > 0.0f) {
        for (int k = 0; k < 3; ++k) edgeA->other->vel[k] += (J * wA1 * inv_mA1) * normal[k] + (jt * wA1 * inv_mA1) * t_dir[k];
    }
    if (inv_mB0 > 0.0f) {
        for (int k = 0; k < 3; ++k) edgeB->node->vel[k] -= (J * wB0 * inv_mB0) * normal[k] + (jt * wB0 * inv_mB0) * t_dir[k];
    }
    if (inv_mB1 > 0.0f) {
        for (int k = 0; k < 3; ++k) edgeB->other->vel[k] -= (J * wB1 * inv_mB1) * normal[k] + (jt * wB1 * inv_mB1) * t_dir[k];
    }
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
static void compute_triangle_collision_forces(Simulator *s, float *collision_forces, float sub_dt) {
    if (!s || !collision_forces) return;
    const float PENETRATION_STIFFNESS = 500.0f;
    const float NORMAL_DAMP_K = 100.0f;
    const float Kt = 150.0f;
    
    size_t n_nodes = dynarray_size(s->nodes);
    size_t n_walls = dynarray_size(s->walls);
    if (n_nodes == 0 || n_walls == 0) return;
    
    // --- Build/Refit BVHs if needed (for now, rebuild every frame) ---
    if (s->triangle_bvh) triangle_bvh_free(s->triangle_bvh);
    s->triangle_bvh = triangle_bvh_build(s->walls);
    if (s->edge_bvh) edge_bvh_free(s->edge_bvh);
    // Collect all triangle edges into a DynArray
    DynArray *edges = dynarray_create(n_walls * 3);
    for (size_t wi = 0; wi < n_walls; ++wi) {
        TriangleWall *w = (TriangleWall*)dynarray_get(s->walls, wi);
        if (!w) continue;
        for (int ei = 0; ei < 3; ++ei) {
            Constraint *edge = w->edges[ei];
            if (edge) dynarray_append(edges, edge);
        }
    }
    s->edge_bvh = edge_bvh_build(edges, 0.03f); // TODO: use actual edge radius

    // --- Node→Triangle broadphase using TriangleBVH ---
    for (size_t ni = 0; ni < n_nodes; ++ni) {
        Node *node = (Node*)dynarray_get(s->nodes, ni);
        if (!node) continue;
        if (node->anchored && !node->collide_when_anchored) continue;
        if (!node->collide_with_walls) continue;
        // Build a small AABB around node (sphere AABB)
        AABB node_box;
        for (int d = 0; d < 3; ++d) {
            node_box.min[d] = node->pos[d] - node->radius;
            node_box.max[d] = node->pos[d] + node->radius;
        }
        DynArray *candidates = dynarray_create(8);
        triangle_bvh_query(s->triangle_bvh, &node_box, candidates);
        // printf("Node %zu: found %zu candidate triangles\n", ni, dynarray_size(candidates));
        for (size_t ci = 0; ci < dynarray_size(candidates); ++ci) {
            int tri_idx = *(int*)dynarray_get(candidates, ci);
            if (tri_idx < 0 || (size_t)tri_idx >= n_walls) continue;
            TriangleWall *w = (TriangleWall*)dynarray_get(s->walls, tri_idx);
            if (!w) continue;
            float penetration, normal[3], u, v;
            if (triangle_check_collision(w, node, &penetration, normal, &u, &v)) {
                // Compute separation force
                float sep_force[3] = {
                    normal[0] * penetration * PENETRATION_STIFFNESS,
                    normal[1] * penetration * PENETRATION_STIFFNESS,
                    normal[2] * penetration * PENETRATION_STIFFNESS
                };
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
                if (normal_vel < 0.0f) {
                    sep_force[0] -= normal[0] * normal_vel * NORMAL_DAMP_K;
                    sep_force[1] -= normal[1] * normal_vel * NORMAL_DAMP_K;
                    sep_force[2] -= normal[2] * normal_vel * NORMAL_DAMP_K;
                }
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
                collision_forces[3*ni + 0] += sep_force[0];
                collision_forces[3*ni + 1] += sep_force[1];
                collision_forces[3*ni + 2] += sep_force[2];
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
        dynarray_free(candidates, free);
    }

    
        EdgeEdgeContext ctx = { edges, collision_forces, sub_dt };
    edge_bvh_self_traverse(s->edge_bvh, edge_edge_callback, &ctx);
    dynarray_free(edges, NULL);
}


/* --------------------------------------------------------------------------
 * gpu_debug_compare: read back GPU J/b/A/l, solve CG on CPU, compare.
 * Called once per simulator_gpu_step (sub==0) when ctx->gpu_debug is set.
 * -------------------------------------------------------------------------- */
static void gpu_debug_compare(GPUContext *ctx, Simulator *s, float sub_dt) {
    int m = ctx->m_sparse;
    if (m <= 0) return;
    fprintf(stderr, "[GPU_DEBUG] sub_dt=%.6f  m_sparse=%d  m_total=%d\n",
            sub_dt, m, ctx->m_total);

    /* Read GPU b and l vectors */
    float *b_gpu = (float*)malloc(sizeof(float) * (size_t)m);
    float *l_gpu = (float*)malloc(sizeof(float) * (size_t)m);
    float *A_gpu = (float*)malloc(sizeof(float) * (size_t)m * (size_t)m);
    if (!b_gpu || !l_gpu || !A_gpu) {
        free(b_gpu); free(l_gpu); free(A_gpu);
        fprintf(stderr, "[GPU_DEBUG] alloc failed\n");
        return;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_b_vec);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float)*(size_t)m, b_gpu);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_l_vec);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float)*(size_t)m, l_gpu);
    /* ssbo_A_dense is a 1-float stub — build_A is not dispatched in the sparse CG path.
       Zero A_gpu so the CPU CG solve below is skipped (pAp=0 → break on first iter). */
    memset(A_gpu, 0, sizeof(float)*(size_t)m*(size_t)m);

    /* CPU CG solve using the same A and b from GPU readback */
    float *l_cpu = (float*)calloc((size_t)m, sizeof(float));
    float *r     = (float*)malloc(sizeof(float) * (size_t)m);
    float *p     = (float*)malloc(sizeof(float) * (size_t)m);
    float *Ap    = (float*)malloc(sizeof(float) * (size_t)m);
    if (l_cpu && r && p && Ap) {
        for (int i = 0; i < m; i++) { r[i] = b_gpu[i]; p[i] = r[i]; }
        float rsold = 0.0f;
        for (int i = 0; i < m; i++) rsold += r[i]*r[i];

        for (int iter = 0; iter < 200 && rsold > 1e-20f; iter++) {
            for (int i = 0; i < m; i++) {
                Ap[i] = 0.0f;
                for (int j = 0; j < m; j++) Ap[i] += A_gpu[i*m+j] * p[j];
            }
            float pAp = 0.0f;
            for (int i = 0; i < m; i++) pAp += p[i]*Ap[i];
            if (fabsf(pAp) < 1e-30f) break;
            float alpha = rsold / pAp;
            float rsnew = 0.0f;
            for (int i = 0; i < m; i++) {
                l_cpu[i] += alpha * p[i];
                r[i]     -= alpha * Ap[i];
                rsnew    += r[i]*r[i];
            }
            if (rsnew < 1e-20f) break;
            float beta = rsnew / rsold;
            for (int i = 0; i < m; i++) p[i] = r[i] + beta*p[i];
            rsold = rsnew;
        }

        int print_n = (m < 10) ? m : 10;
        fprintf(stderr, "[GPU_DEBUG]  idx |    b_gpu    |   l_gpu   |   l_cpu\n");
        fprintf(stderr, "[GPU_DEBUG] -----+-------------+-----------+----------\n");
        for (int i = 0; i < print_n; i++) {
            fprintf(stderr, "[GPU_DEBUG]  %3d | %11.6f | %9.6f | %9.6f\n",
                    i, b_gpu[i], l_gpu[i], l_cpu[i]);
        }
    }

    free(b_gpu); free(l_gpu); free(A_gpu);
    free(l_cpu); free(r); free(p); free(Ap);
}

/* --------------------------------------------------------------------------
 * simulator_gpu_step: full N-substep GPU pipeline.
 *   For each substep:
 *     0. CPU collision → ssbo_collision
 *     1. ext_forces shader   (gravity + springs)
 *     2. build_j shader      (Jacobian rows)
 *     3. build_b shader      (RHS b = -err)
 *     4. build_A shader      (dense A = dt² J M⁻¹ Jᵀ)
 *     5. cg_solve shader     (A λ = b, single workgroup)
 *     6. [optional debug]
 *     7. apply_corr shader   (corr_f + symplectic Euler integrate)
 *   Then readback positions/velocities to CPU Node structs.
 * -------------------------------------------------------------------------- */
void simulator_gpu_step(Simulator *s) {
    if (!s || !s->gpu_ctx) return;
    GPUContext *ctx = (GPUContext*)s->gpu_ctx;

    /* Validate all shaders compiled */
    if (!ctx->prog_ext_forces || !ctx->prog_build_jb ||
        !ctx->prog_cg_init    || !ctx->prog_cg_j_gather ||
        !ctx->prog_cg_update_xr  || !ctx->prog_cg_update_p || !ctx->prog_apply_corr) {
        fprintf(stderr, "[GPU] One or more compute programs not compiled — cannot step\n");
        return;
    }

    int n       = ctx->node_count;
    int m       = ctx->m_sparse;
    int m_total = ctx->m_total;
    if (n <= 0) return;
    /* Sparse CG supports unlimited m — no 512-constraint cap */

    int   N      = (s->solver_iters > 0) ? s->solver_iters : 1;
    float sub_dt = s->dt / (float)N;

    /* ── Timing setup ─────────────────────────────────────────────────────── */
    double t_frame_start   = cpu_now();
    double acc_lbvh_ms     = 0.0, acc_wbvh_ms   = 0.0, acc_col_ms  = 0.0;
    double acc_cg_ms       = 0.0, acc_apply_ms   = 0.0, acc_part_ms = 0.0;
    int    tq_w = ctx->tq_idx; /* query buffer we write this frame */
    double t0;                 /* scratch timestamp for phase measurement */

    /* -- Bind all SSBOs once (never change within a frame except slot 13 swap) -- */
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  0, ctx->ssbo_positions);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  1, ctx->ssbo_velocities);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  2, ctx->ssbo_inv_mass);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  3, ctx->ssbo_ext_forces);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  4, ctx->ssbo_node_flags);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  5, ctx->ssbo_constraints);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  6, ctx->ssbo_J_cols);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  7, ctx->ssbo_J_vals);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  8, ctx->ssbo_A_dense);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  9, ctx->ssbo_b_vec);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, ctx->ssbo_l_vec);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, ctx->ssbo_collision);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, ctx->ssbo_r_vec);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 13, ctx->ssbo_p_vec);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 14, ctx->ssbo_Ap_vec);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, ctx->ssbo_bvh_nodes);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 16, ctx->ssbo_triangles);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 17, ctx->ssbo_wall_indices);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 18, ctx->ssbo_velcorr);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 20, ctx->ssbo_edge_bvh);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 21, ctx->ssbo_edge_bvh_scratch);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 22, ctx->ssbo_wall_bvh_scratch);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 23, ctx->ssbo_jt_vec);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 24, ctx->ssbo_cg_scalars);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 25, ctx->ssbo_reduce_buf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 26, ctx->ssbo_csr_offsets);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 27, ctx->ssbo_csr_data);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 28, ctx->ssbo_indirect_args);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 29, ctx->ssbo_edge_meta);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 39, ctx->ssbo_particle_mesh_forces);

    /* -- Set all uniforms once (constant across substeps) -- */
    int ng = (n + 63) / 64;
    int mg = (m > 0) ? (m + 63) / 64 : 0;

    if (ctx->prog_collision && ctx->n_triangles > 0) {
        glUseProgram(ctx->prog_collision);
        glUniform1i(ctx->u_col_n,         n);
        glUniform1i(ctx->u_col_n_tris,    ctx->n_triangles);
        glUniform1f(ctx->u_col_stiffness, 100000.0f);
        glUniform1f(ctx->u_col_damp,       2000.0f);
        glUniform1f(ctx->u_col_fr,          0.4f);
    }

    if (ctx->prog_edge_edge && m_total > 0) {
        glUseProgram(ctx->prog_edge_edge);
        glUniform1i(ctx->u_ee_n,           n);
        glUniform1i(ctx->u_ee_m_total,     m_total);
        glUniform1f(ctx->u_ee_restitution, 0.05f);
        glUniform1f(ctx->u_ee_mu,          0.3f);
    }

    glUseProgram(ctx->prog_ext_forces);
    glUniform1i(ctx->u_ef_n,        n);
    glUniform1i(ctx->u_ef_m_sparse, m);
    glUniform1i(ctx->u_ef_m_total,  m_total);
    glUniform3f(ctx->u_ef_gravity,  s->gravity[0], s->gravity[1], s->gravity[2]);

    if (m > 0) {
        glUseProgram(ctx->prog_build_jb);
        glUniform1i(ctx->u_bjb_m, m);

        // --- Sparse CG uniforms (invariant across substeps) ---
        int n_partials = (m + 63) / 64;

        glUseProgram(ctx->prog_cg_init);
        glUniform1i(ctx->u_ci_m, m);

        /* Baked jt_gather variants — only n3 varies per scene upload */
        if (ctx->prog_cg_jt_gather_inloop) {
            glUseProgram(ctx->prog_cg_jt_gather_inloop);
            glUniform1i(ctx->u_cjtg_il_n3, 3 * n);
        }
        /* Legacy jt_gather — kept as fallback */
        glUseProgram(ctx->prog_cg_jt_gather);
        glUniform1i(ctx->u_cjtg_n3, 3 * n);

        glUseProgram(ctx->prog_cg_j_gather);
        glUniform1i(ctx->u_cjg_m,  m);
        glUniform1f(ctx->u_cjg_dt, sub_dt);

        /* Upload n_partials to the 3 inline-reduction shaders (replaces separate dr_ dispatches) */
        glUseProgram(ctx->prog_cg_init);
        glUniform1i(ctx->u_ci_n_partials, n_partials);
        glUseProgram(ctx->prog_cg_j_gather);
        glUniform1i(ctx->u_cjg_n_partials, n_partials);
        glUseProgram(ctx->prog_cg_update_xr);
        glUniform1i(ctx->u_cuxr_n_partials, n_partials);

        /* Warm-init shader uniforms (constant per scene) */
        if (ctx->prog_cg_warm_init) {
            glUseProgram(ctx->prog_cg_warm_init);
            glUniform1i(ctx->u_cwi_m,          m);
            glUniform1i(ctx->u_cwi_n_partials,  n_partials);
            glUniform1f(ctx->u_cwi_dt,          sub_dt);
        }

        glUseProgram(ctx->prog_cg_update_xr);
        glUniform1i(ctx->u_cuxr_m, m);

        glUseProgram(ctx->prog_cg_update_p);
        glUniform1i(ctx->u_cup_m, m);

        glUseProgram(ctx->prog_apply_corr);
        glUniform1i(ctx->u_ac_n,      n);
        glUniform1f(ctx->u_ac_dt,     s->dt);
        glUniform1f(ctx->u_ac_sub_dt, sub_dt);
        glUniform1i(ctx->u_ac_N,      N);
    }

    for (int sub = 0; sub < N; sub++) {

        /* ── Phase 0: LBVH edge build ─────────────────────────────────── */
        t0 = cpu_now();
        if (sub == 0) glBeginQuery(GL_TIME_ELAPSED, ctx->tq[tq_w][0]);
        /* -- Build edge LBVH for this substep (positions may have changed) --
           Use a generous fixed scene AABB for Morton normalization.           */
        if (ctx->enable_edge_edge && m_total > 0 && ctx->prog_lbvh_morton_edges && ctx->prog_lbvh_sort_edges && ctx->prog_lbvh_build_edges) {
            float smin[3] = { ctx->scene_min[0], ctx->scene_min[1], ctx->scene_min[2] };
            float smax[3] = { ctx->scene_max[0], ctx->scene_max[1], ctx->scene_max[2] };
            int n_edges = 0;
            lbvh_build_edges_gpu(ctx, m_total, smin, smax, &n_edges);
            ctx->last_n_edges = n_edges;
        }
        if (sub == 0) glEndQuery(GL_TIME_ELAPSED);
        acc_lbvh_ms += (cpu_now() - t0) * 1e3;

        /* ── Phase 1: LBVH wall BVH rebuild ──────────────────────────── */
        t0 = cpu_now();
        if (sub == 0) glBeginQuery(GL_TIME_ELAPSED, ctx->tq[tq_w][1]);
        /* -- Optionally rebuild wall LBVH (needed when walls contain moving nodes) -- */
        if (ctx->rebuild_wall_bvh_per_substep && ctx->n_triangles > 0) {
            float smin[3] = { ctx->scene_min[0], ctx->scene_min[1], ctx->scene_min[2] };
            float smax[3] = { ctx->scene_max[0], ctx->scene_max[1], ctx->scene_max[2] };
            lbvh_build_walls_gpu(ctx, ctx->n_triangles, smin, smax);
        }
        if (sub == 0) glEndQuery(GL_TIME_ELAPSED);
        acc_wbvh_ms += (cpu_now() - t0) * 1e3;

        /* ── Phase 2: Collision + external forces ─────────────────────── */
        t0 = cpu_now();
        if (sub == 0) glBeginQuery(GL_TIME_ELAPSED, ctx->tq[tq_w][2]);
        /* -- Edge-edge collision: 1 thread/edge, BVH traversal (O(m log m)) -- */
        if (ctx->enable_edge_edge && ctx->prog_edge_edge && m_total > 0) {
            int ee_groups = (m_total + 63) / 64;   /* 1 thread per edge (was m²/64) */
            glUseProgram(ctx->prog_edge_edge);
            glDispatchCompute((GLuint)ee_groups, 1, 1);
            /* no barrier yet — apply_corr reads velcorr at end of substep */
        }

        /* -- Sphere-triangle collision (BVH traversal) + ext_forces (independent) -- */
        if (ctx->prog_collision && ctx->n_triangles > 0) {
            glUseProgram(ctx->prog_collision);
            glDispatchCompute((GLuint)ng, 1, 1);
        }
        glUseProgram(ctx->prog_ext_forces);
        glDispatchCompute((GLuint)ng, 1, 1);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        if (sub == 0) glEndQuery(GL_TIME_ELAPSED);
        acc_col_ms += (cpu_now() - t0) * 1e3;

        if (m > 0) {
            /* ── Phase 3: build_jb + CG ──────────────────────────────── */
            t0 = cpu_now();
            if (sub == 0) glBeginQuery(GL_TIME_ELAPSED, ctx->tq[tq_w][3]);            /* -- 2+3. Build J and b (merged into one dispatch) -- */
            glUseProgram(ctx->prog_build_jb);
            glDispatchCompute((GLuint)mg, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

            /* -- 4-7. Sparse CG: Aλ = b  where A = dt² J M⁻¹ Jᵀ (implicit, CSR gather) -- */
            {
                int jsg = mg;                        /* ceil(m/64)  */
                int jtg = (3*n + 63) / 64;           /* ceil(3n/64) */

                GLuint jtg_il   = ctx->prog_cg_jt_gather_inloop ? ctx->prog_cg_jt_gather_inloop : ctx->prog_cg_jt_gather;

                int using_generic_jtg = (jtg_il == ctx->prog_cg_jt_gather);

                /* Init / warm-start: set x, r, p and compute initial r·r */
                int max_cg_iter;
                if (sub == 0 || !ctx->prog_cg_warm_init) {
                    /* Cold start (sub==0): x=0, r=p=b */
                    glUseProgram(ctx->prog_cg_init);
                    glDispatchCompute((GLuint)jsg, 1, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    max_cg_iter = (m < 20) ? m : 20;
                } else {
                    /* Warm start: x=x_prev; compute jt_vec=M^{-1}J^Tx0, then r=b-Ax0, p=r */
                    glUseProgram(ctx->prog_cg_jt_gather);
                    glUniform1i(ctx->u_cjtg_src,        1); /* read l_data (x_prev) */
                    glUniform1i(ctx->u_cjtg_apply_minv, 1);
                    glDispatchCompute((GLuint)jtg, 1, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    glUseProgram(ctx->prog_cg_warm_init);
                    glDispatchCompute((GLuint)jsg, 1, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                    max_cg_iter = (m < 10) ? m : 10;
                }
                for (int iter = 0; iter < max_cg_iter; iter++) {

                    /* jt_vec = M⁻¹ Jᵀ p  (baked: src=p, minv=1) */
                    glUseProgram(jtg_il);
                    if (using_generic_jtg) {
                        glUniform1i(ctx->u_cjtg_src,        0);
                        glUniform1i(ctx->u_cjtg_apply_minv, 1);
                    }
                    glDispatchCompute((GLuint)jtg, 1, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                    /* Ap = dt² J jt_vec + 1e-10*p; partial p·Ap → reduce_buf; finalizes α inline */
                    glUseProgram(ctx->prog_cg_j_gather);
                    glDispatchCompute((GLuint)jsg, 1, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                    /* x += α*p;  r -= α*Ap;  partial r·r → reduce_buf; finalizes β inline */
                    glUseProgram(ctx->prog_cg_update_xr);
                    glDispatchCompute((GLuint)jsg, 1, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

                    /* p = r + β*p */
                    glUseProgram(ctx->prog_cg_update_p);
                    glDispatchCompute((GLuint)jsg, 1, 1);
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
                }

            } /* end CG block */

            if (sub == 0) glEndQuery(GL_TIME_ELAPSED);
            acc_cg_ms += (cpu_now() - t0) * 1e3;

            /* ── Phase 4: apply_corr ──────────────────────────────────── */
            t0 = cpu_now();
            if (sub == 0) glBeginQuery(GL_TIME_ELAPSED, ctx->tq[tq_w][4]);
            /* -- 8. Apply corrections + integrate (J^T λ computed inline) -- */
            glUseProgram(ctx->prog_apply_corr);
            glDispatchCompute((GLuint)ng, 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            if (sub == 0) glEndQuery(GL_TIME_ELAPSED);
            acc_apply_ms += (cpu_now() - t0) * 1e3;
        } else if (sub == 0) {
            /* m==0: issue empty queries so the result slots are defined */
            glBeginQuery(GL_TIME_ELAPSED, ctx->tq[tq_w][3]); glEndQuery(GL_TIME_ELAPSED);
            glBeginQuery(GL_TIME_ELAPSED, ctx->tq[tq_w][4]); glEndQuery(GL_TIME_ELAPSED);
        }

        /* ── Phase 5: Particle sim step ──────────────────────────────── */
        t0 = cpu_now();
        if (sub == 0) glBeginQuery(GL_TIME_ELAPSED, ctx->tq[tq_w][5]);
        /* -- Particle sim step: runs AFTER apply_corr so it reads up-to-date  --
           node positions (piston at its true location for this substep).         --
           particle_mesh_forces written here are used by apply_corr next substep. */
        if (s->particles) {
            ParticleSimExtern ext;
            ext.ssbo_node_positions       = ctx->ssbo_positions;
            ext.ssbo_node_inv_mass        = ctx->ssbo_inv_mass;
            ext.ssbo_wall_bvh_nodes       = ctx->ssbo_bvh_nodes;
            ext.ssbo_wall_bvh_scratch     = ctx->ssbo_wall_bvh_scratch;
            ext.ssbo_wall_indices         = ctx->ssbo_wall_indices;
            ext.ssbo_particle_mesh_forces = ctx->ssbo_particle_mesh_forces;
            ext.n_tris                    = ctx->n_triangles;
            ext.n_nodes                   = ctx->node_count;
            particle_sim_step(s->particles, &ext, sub_dt);
            /* Rebind SSBOs that particle_sim_step may have clobbered */
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  0, ctx->ssbo_positions);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  1, ctx->ssbo_velocities);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, ctx->ssbo_collision);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 15, ctx->ssbo_bvh_nodes);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 17, ctx->ssbo_wall_indices);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 22, ctx->ssbo_wall_bvh_scratch);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 39, ctx->ssbo_particle_mesh_forces);
        }
        if (sub == 0) glEndQuery(GL_TIME_ELAPSED);
        acc_part_ms += (cpu_now() - t0) * 1e3;

    } /* end substep loop */

    /* ── GPU timer: end particle phase query if no particles (query still open) ── */
    if (!s->particles) glEndQuery(GL_TIME_ELAPSED);

    /* ── glFinish: wait for all GPU work this frame ─────────────────────── */
    glFinish();
    double t_total_end = cpu_now();

    /* ── Populate cpu timings ───────────────────────────────────────────── */
    ctx->timings.cpu_total_ms      = (t_total_end - t_frame_start) * 1e3;
    ctx->timings.cpu_lbvh_ms       = acc_lbvh_ms;
    ctx->timings.cpu_wall_bvh_ms   = acc_wbvh_ms;
    ctx->timings.cpu_collision_ms  = acc_col_ms;
    ctx->timings.cpu_cg_ms         = acc_cg_ms;
    ctx->timings.cpu_apply_ms      = acc_apply_ms;
    ctx->timings.cpu_particles_ms  = acc_part_ms;

    /* ── Read GPU timings from the PREVIOUS frame's queries ─────────────── */
    if (ctx->tq_ready) {
        int old = ctx->tq_idx ^ 1;
        const char *phase_names[SIM_TQ_PHASES] = {
            "lbvh", "wall_bvh", "collision", "cg", "apply_corr", "particles"
        };
        double *gpu_fields[SIM_TQ_PHASES] = {
            &ctx->timings.gpu_lbvh_ms,
            &ctx->timings.gpu_wall_bvh_ms,
            &ctx->timings.gpu_collision_ms,
            &ctx->timings.gpu_cg_ms,
            &ctx->timings.gpu_apply_ms,
            &ctx->timings.gpu_particles_ms
        };
        for (int ph = 0; ph < SIM_TQ_PHASES; ph++) {
            GLuint64 ns = 0;
            glGetQueryObjectui64v(ctx->tq[old][ph], GL_QUERY_RESULT, &ns);
            *gpu_fields[ph] = (double)ns * 1e-6; /* ns → ms */
            (void)phase_names[ph];
        }
    }

    /* Flip query buffer for next frame */
    ctx->tq_idx ^= 1;
    ctx->tq_ready = 1;

    /* Plant a fence so simulator_sync_positions can check (non-blocking) when the
       GPU has finished writing positions/velocities.  With GL_MAP_COHERENT_BIT the
       data is automatically visible to the CPU once the fence signals.             */
    if (ctx->readback_fence) glDeleteSync(ctx->readback_fence);
    ctx->readback_fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

// // Incremental octree rebuild: only update dirty nodes
// void incremental_octree_rebuild(OctreeNode *octree, DynArray *nodes, float threshold) {
//     if (!octree || !nodes) return;
//     for (size_t i = 0; i < dynarray_size(nodes); ++i) {
//         Node *node = (Node*)dynarray_get(nodes, i);
//         if (!node) continue;
//         mark_node_dirty_if_moved(node, threshold);
//         if (node->dirty) {
//             node->dirty = false;
//             // Remove from octree and re-add at new position
//             octree_remove(octree, node->idx, node->prev_position); // Remove using previous position
//             octree_insert_node(octree, node->idx, node->pos);      // Add at new position
//         }
//     }
// }

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
    s->solver_iters = 50;
    s->damping = 0.01f;
    s->triangle_bvh = NULL;
    s->edge_bvh = NULL;
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
    // free BVHs
    if (s->triangle_bvh) triangle_bvh_free(s->triangle_bvh);
    if (s->edge_bvh) edge_bvh_free(s->edge_bvh);
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





// Legacy 2D spatial hash collision code (deprecated, kept for reference)
// Helper: build spatial hash mapping cell_key -> DynArray of node indices (as int* allocated)
// Simple spatial hash for two 32-bit integers (ix,iy).
void simulator_step(Simulator *s) {
    if (!s) return;
    // If GPU integration is enabled, dispatch GPU step (stubbed currently).
    if (s->use_gpu) {
        simulator_gpu_step(s);
        return;
    }
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

        // compute collision forces / impulses fresh each substep
        compute_triangle_collision_forces(s, collision_forces, sub_dt);
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
            {
                // Try to solve using CSparse direct Cholesky (sparse solver).
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

/* Read back GPU positions + velocities to the CPU Node structs.
   Call once per render frame (not inside the substep loop) so that
   UI picking / drag-force code has up-to-date positions.

   Fast path  (persistent coherent map available):
     Checks the fence from the previous gpu_step non-blocking (0 ns timeout).
     If signaled → GPU writes are coherently visible → memcpy from mapped pointer.
     If not yet signaled (GPU still busy) → returns immediately; Node positions
     remain at the last successfully synced frame (1-frame stale is fine for UI).

   Slow fallback (no persistent map, e.g. glBufferStorage unavailable):
     Calls glGetBufferSubData which stalls the pipeline until the GPU is done. */
void simulator_set_rebuild_wall_bvh(Simulator *s, int enable) {
    if (!s || !s->gpu_ctx) return;
    ((GPUContext*)s->gpu_ctx)->rebuild_wall_bvh_per_substep = enable;
}

void simulator_set_edge_edge(Simulator *s, int enable) {
    if (!s || !s->gpu_ctx) return;
    ((GPUContext*)s->gpu_ctx)->enable_edge_edge = enable;
}

void simulator_upload_wall_flags(Simulator *s) {
    if (!s || !s->use_gpu || !s->gpu_ctx || !s->walls) return;
    GPUContext *ctx = (GPUContext*)s->gpu_ctx;
    int tcount = (int)dynarray_size(s->walls);
    if (tcount <= 0) return;
    /* Read back the full widx buffer, patch only the flag bytes, and re-upload */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_wall_indices);
    int *buf = (int*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                      sizeof(int) * 4 * (size_t)tcount,
                                      GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
    if (buf) {
        for (int i = 0; i < tcount; i++) {
            TriangleWall *w = (TriangleWall*)dynarray_get(s->walls, i);
            if (w) buf[i * 4 + 3] = w->translucent ? 1 : 0;
        }
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void simulator_sync_positions(Simulator *s) {
    if (!s || !s->use_gpu || !s->gpu_ctx) return;
    GPUContext *ctx = (GPUContext*)s->gpu_ctx;
    int n = ctx->node_count;
    if (n <= 0) return;

    /* ── Fast path: persistent coherent map ─────────────────────────────────── */
    if (ctx->pos_map && ctx->vel_map) {
        /* If no fence yet (before first gpu_step) just read the initial upload. */
        if (!ctx->readback_fence ||
            glClientWaitSync(ctx->readback_fence, GL_SYNC_FLUSH_COMMANDS_BIT, 0)
                != GL_TIMEOUT_EXPIRED) {
            const float *pm = ctx->pos_map;
            const float *vm = ctx->vel_map;
            for (int i = 0; i < n; i++) {
                Node *nd = (Node*)dynarray_get(s->nodes, i);
                if (!nd) continue;
                nd->pos[0] = pm[4*i];   nd->pos[1] = pm[4*i+1]; nd->pos[2] = pm[4*i+2];
                nd->vel[0] = vm[4*i];   nd->vel[1] = vm[4*i+1]; nd->vel[2] = vm[4*i+2];
            }
        }
        /* If GL_TIMEOUT_EXPIRED: GPU still running; keep last-known positions. */
        return;
    }

    /* ── Slow fallback: drain pipeline + copy ────────────────────────────────── */
    size_t pv_sz = sizeof(float) * (size_t)n * 4;
    float *pos4 = (float*)malloc(pv_sz);
    float *vel4 = (float*)malloc(pv_sz);
    if (pos4 && vel4) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_positions);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)pv_sz, pos4);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->ssbo_velocities);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)pv_sz, vel4);
        for (int i = 0; i < n; i++) {
            Node *nd = (Node*)dynarray_get(s->nodes, i);
            if (!nd) continue;
            nd->pos[0] = pos4[4*i]; nd->pos[1] = pos4[4*i+1]; nd->pos[2] = pos4[4*i+2];
            nd->vel[0] = vel4[4*i]; nd->vel[1] = vel4[4*i+1]; nd->vel[2] = vel4[4*i+2];
        }
    }
    free(pos4);
    free(vel4);
}

void simulator_draw(Simulator *s, float cam_yaw, float cam_pitch) {
    if (!s) return;

    /* ── GPU render path ─────────────────────────────────────────────────── */
    if (s->use_gpu && s->gpu_ctx) {
        GPUContext *ctx = (GPUContext*)s->gpu_ctx;
        if (ctx->prog_draw_walls && ctx->prog_draw_constraints && ctx->prog_draw_nodes) {

            /* Build MVP = Projection * ModelView (column-major) */
            float proj[16], mv[16], mvp[16];
            glGetFloatv(GL_PROJECTION_MATRIX, proj);
            glGetFloatv(GL_MODELVIEW_MATRIX,  mv);
            mat4_mul_cm(mvp, proj, mv);

            /* Disable fixed-function state that would interfere */
            glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glDisable(GL_LIGHTING);
            glDisable(GL_TEXTURE_2D);

            /* --- Draw constraints --- */
            if (ctx->m_total > 0) {
                glUseProgram(ctx->prog_draw_constraints);
                glUniformMatrix4fv(ctx->u_draw_con_mvp, 1, GL_FALSE, mvp);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  0, ctx->ssbo_positions);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  5, ctx->ssbo_constraints);
                glBindVertexArray(ctx->vao_empty);
                glDrawArrays(GL_LINES, 0, ctx->m_total * 2);
                glBindVertexArray(0);
            }

            /* --- Draw nodes (instanced spheres) --- */
            if (ctx->node_count > 0 && ctx->sphere_index_count > 0) {
                glUseProgram(ctx->prog_draw_nodes);
                glUniformMatrix4fv(ctx->u_draw_node_mvp, 1, GL_FALSE, mvp);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ctx->ssbo_positions);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, ctx->ssbo_node_flags);
                glBindVertexArray(ctx->vao_sphere);
                glDrawElementsInstanced(GL_TRIANGLES, ctx->sphere_index_count,
                                        GL_UNSIGNED_SHORT, 0, ctx->node_count);
                glBindVertexArray(0);
            }

            /* --- Draw particles (points, must be before walls for correct blending) --- */
            if (s->particles) {
                GLint vp[4]; glGetIntegerv(GL_VIEWPORT, vp);
                float screen_h = (float)vp[3];
                /* proj[5] = proj[1][1] in column-major = focal length Y */
                particle_sim_draw(s->particles, mvp, proj[5], screen_h);

                /* --- Draw producer/consumer modifier tiles (green/red quads) --- */
                if (s->particles->n_modifiers > 0) {
                    ParticleSim *_ps = s->particles;
                    glUseProgram(0);
                    glEnable(GL_BLEND);
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                    glDisable(GL_CULL_FACE);
                    glDepthMask(GL_FALSE);
                    for (int _m = 0; _m < _ps->n_modifiers; _m++) {
                        const ParticleModifier *_mod = &_ps->modifiers[_m];
                        if (!_mod->enabled) continue;
                        /* Recompute face axes from yaw + pitch */
                        float _sy = sinf(_mod->yaw),  _cy = cosf(_mod->yaw);
                        float _sp = sinf(_mod->pitch), _cp = cosf(_mod->pitch);
                        /* tanU = (cos yaw, 0, -sin yaw) */
                        float _ux = _cy,          _uy = 0.0f,  _uz = -_sy;
                        /* tanV = cross(normal, tanU) = (-sp*sy, cp, -sp*cy) */
                        float _vx = -_sp * _sy,   _vy = _cp,   _vz = -_sp * _cy;
                        float _hw = _mod->width  * 0.5f;
                        float _hl = _mod->length * 0.5f;
                        float _px = _mod->pos[0], _py = _mod->pos[1], _pz = _mod->pos[2];
                        /* Filled tile: green = producer (type 0), red = consumer (type 1) */
                        if (_mod->type == 0) glColor4f(0.1f, 0.85f, 0.1f, 0.40f);
                        else                 glColor4f(0.85f, 0.1f, 0.1f, 0.40f);
                        glBegin(GL_QUADS);
                        glVertex3f(_px - _ux*_hw - _vx*_hl, _py - _uy*_hw - _vy*_hl, _pz - _uz*_hw - _vz*_hl);
                        glVertex3f(_px + _ux*_hw - _vx*_hl, _py + _uy*_hw - _vy*_hl, _pz + _uz*_hw - _vz*_hl);
                        glVertex3f(_px + _ux*_hw + _vx*_hl, _py + _uy*_hw + _vy*_hl, _pz + _uz*_hw + _vz*_hl);
                        glVertex3f(_px - _ux*_hw + _vx*_hl, _py - _uy*_hw + _vy*_hl, _pz - _uz*_hw + _vz*_hl);
                        glEnd();
                        /* Bright border outline */
                        if (_mod->type == 0) glColor4f(0.2f, 1.0f, 0.2f, 0.90f);
                        else                 glColor4f(1.0f, 0.2f, 0.2f, 0.90f);
                        glLineWidth(2.0f);
                        glBegin(GL_LINE_LOOP);
                        glVertex3f(_px - _ux*_hw - _vx*_hl, _py - _uy*_hw - _vy*_hl, _pz - _uz*_hw - _vz*_hl);
                        glVertex3f(_px + _ux*_hw - _vx*_hl, _py + _uy*_hw - _vy*_hl, _pz + _uz*_hw - _vz*_hl);
                        glVertex3f(_px + _ux*_hw + _vx*_hl, _py + _uy*_hw + _vy*_hl, _pz + _uz*_hw + _vz*_hl);
                        glVertex3f(_px - _ux*_hw + _vx*_hl, _py - _uy*_hw + _vy*_hl, _pz - _uz*_hw + _vz*_hl);
                        glEnd();
                        glLineWidth(1.0f);
                    }
                    glDepthMask(GL_TRUE);
                    glDisable(GL_BLEND);
                }

                /* Restore bindings that particle draw may have clobbered */
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  0, ctx->ssbo_positions);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 17, ctx->ssbo_wall_indices);

                /* --- Draw heater/cooler volumes as blue wireframe AABBs --- */
                if (s->particles->n_heaters > 0) {
                    ParticleSim *_ps = s->particles;
                    glUseProgram(0);
                    glEnable(GL_BLEND);
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                    glDepthMask(GL_FALSE);
                    glLineWidth(2.0f);
                    for (int _h = 0; _h < _ps->n_heaters; _h++) {
                        const HeaterCooler *_hc = &_ps->heaters[_h];
                        if (!_hc->enabled) continue;
                        float _cx = _hc->pos[0], _cy = _hc->pos[1], _cz = _hc->pos[2];
                        float _ex = _hc->half_x, _ey = _hc->half_y, _ez = _hc->half_z;
                        /* 8 corners of the AABB */
                        float _x0 = _cx - _ex, _x1 = _cx + _ex;
                        float _y0 = _cy - _ey, _y1 = _cy + _ey;
                        float _z0 = _cz - _ez, _z1 = _cz + _ez;
                        glColor4f(0.2f, 0.5f, 1.0f, 0.85f);
                        glBegin(GL_LINE_LOOP); /* bottom face */
                        glVertex3f(_x0,_y0,_z0); glVertex3f(_x1,_y0,_z0);
                        glVertex3f(_x1,_y0,_z1); glVertex3f(_x0,_y0,_z1);
                        glEnd();
                        glBegin(GL_LINE_LOOP); /* top face */
                        glVertex3f(_x0,_y1,_z0); glVertex3f(_x1,_y1,_z0);
                        glVertex3f(_x1,_y1,_z1); glVertex3f(_x0,_y1,_z1);
                        glEnd();
                        glBegin(GL_LINES); /* 4 vertical pillars */
                        glVertex3f(_x0,_y0,_z0); glVertex3f(_x0,_y1,_z0);
                        glVertex3f(_x1,_y0,_z0); glVertex3f(_x1,_y1,_z0);
                        glVertex3f(_x1,_y0,_z1); glVertex3f(_x1,_y1,_z1);
                        glVertex3f(_x0,_y0,_z1); glVertex3f(_x0,_y1,_z1);
                        glEnd();
                    }
                    glLineWidth(1.0f);
                    glDepthMask(GL_TRUE);
                    glDisable(GL_BLEND);
                }
            }

            /* --- Draw walls LAST so translucent walls blend correctly over
                   particles and nodes that have already been written to the
                   depth buffer and framebuffer. --- */
            if (ctx->n_triangles > 0) {
                glUseProgram(ctx->prog_draw_walls);
                glUniformMatrix4fv(ctx->u_draw_wall_mvp, 1, GL_FALSE, mvp);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER,  0, ctx->ssbo_positions);
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 17, ctx->ssbo_wall_indices);
                glBindVertexArray(ctx->vao_empty);
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                glDrawArrays(GL_TRIANGLES, 0, ctx->n_triangles * 3);
                glDisable(GL_BLEND);
                glBindVertexArray(0);
            }

            glUseProgram(0);
            glPopAttrib();
            return; /* skip CPU path */
        }
    }

    /* ── CPU / fallback render path ─────────────────────────────────────── */
    // draw constraints
    for (size_t i = 0; i < dynarray_size(s->constraints); ++i) {
        Constraint *c = (Constraint*)dynarray_get(s->constraints, i);
        if (c && c->draw) c->draw(c);
    }
    // draw triangle walls as filled white triangles with black outline in full 3D
    for (size_t i = 0; i < dynarray_size(s->walls); ++i) {
        TriangleWall *w = (TriangleWall*)dynarray_get(s->walls, i);
        if (!w || !w->A || !w->B || !w->C) continue;

        // Draw filled white triangle
        glColor3f(1.0f, 1.0f, 1.0f);
        glBegin(GL_TRIANGLES);
        glVertex3f(w->A->pos[0], w->A->pos[1], w->A->pos[2]);
        glVertex3f(w->B->pos[0], w->B->pos[1], w->B->pos[2]);
        glVertex3f(w->C->pos[0], w->C->pos[1], w->C->pos[2]);
        glEnd();

        // Draw black outline
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
        node_draw(n, n->radius, cam_yaw, cam_pitch);
    }
}

// // Find the closest node to a given position within max_distance
// int simulator_find_closest_node(Simulator *s, float pos[3], float max_distance) {
//     if (!s || !s->octree || !s->nodes) return -1;
    
//     size_t n_nodes = dynarray_size(s->nodes);
//     if (n_nodes == 0) return -1;
    
//     // Convert DynArray to array of Node pointers
//     Node **node_array = (Node**)malloc(sizeof(Node*) * n_nodes);
//     for (size_t i = 0; i < n_nodes; ++i) {
//         node_array[i] = (Node*)dynarray_get(s->nodes, i);
//     }
    
//     int result = octree_find_closest_node(s->octree, node_array, n_nodes, pos, max_distance);
    
//     free(node_array);
//     return result;
// }

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
