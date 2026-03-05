#version 430
// Sphere-triangle collision using GPU-built wall BVH (SSBO 15 + 22).
// Each thread = one physics node.  Traverses the BVH instead of brute-force.
// bvh_build_walls.comp.glsl must run first (once at scene upload).
layout(local_size_x = 64) in;

layout(std430, binding =  0) buffer Positions  { vec4 pos_r[];     };  // .xyz=pos .w=radius
layout(std430, binding =  1) buffer Velocities { vec4 vel[];       };
layout(std430, binding =  4) buffer NodeFlags  { uint nflags[];    };
layout(std430, binding = 11) buffer ColOut     { vec4 col[];       };  // output forces
layout(std430, binding = 15) buffer BvhBuf     { int  bvh_nodes[]; };  // wall BVH  (10 ints/node)
layout(std430, binding = 17) buffer WallIdxBuf { int  widx[];      };  // 4 ints/tri [a,b,c,pad]
layout(std430, binding = 22) buffer ScratchBuf { int  wscratch[];  };  // 8 ints/prim; [6]=orig_tri

uniform int   u_n;           // total node count
uniform int   u_n_tris;      // number of wall triangles
uniform float u_stiffness;   // penetration stiffness  (e.g. 500)
uniform float u_damp;        // normal damping coeff   (e.g. 100)
uniform float u_friction_kt; // kinetic friction coeff (e.g. 0.4)

const uint ANCHORED_BIT         = 1u;
const uint COLLIDE_WALLS_BIT    = 2u;
const uint SIM_IGNORE_BIT       = 4u;
const uint COLLIDE_ANCHORED_BIT = 16u;

// ── Sphere–triangle contact ───────────────────────────────────────────────────
float sphere_tri(vec3 p, float r, vec3 A, vec3 B, vec3 C, out vec3 out_normal) {
    vec3 AB = B - A, AC = C - A;
    vec3 n  = cross(AB, AC);
    float nlen = length(n);
    if (nlen < 1e-9) { out_normal = vec3(0, 1, 0); return -1.0; }
    out_normal = n / nlen;

    float dist = dot(p - A, out_normal);
    if (abs(dist) > r + 1e-3) return -1.0;

    vec3  proj  = p - dist * out_normal;
    vec3  vp    = proj - A;
    float d00   = dot(AB, AB), d01 = dot(AB, AC), d11 = dot(AC, AC);
    float d20   = dot(vp, AB), d21 = dot(vp, AC);
    float denom = d00 * d11 - d01 * d01;
    if (abs(denom) < 1e-18) return -1.0;
    float inv = 1.0 / denom;
    float u   = (d11 * d20 - d01 * d21) * inv;
    float v   = (d00 * d21 - d01 * d20) * inv;
    if (u < -0.01 || v < -0.01 || u + v > 1.01) return -1.0;

    float pen = r - abs(dist);
    if (pen <= 0.0) return -1.0;
    if (dist < 0.0) out_normal = -out_normal;
    return pen;
}

// ── AABB–sphere overlap (with build PAD already baked into BVH bounds) ───────
bool aabb_sphere_overlap(vec3 mn, vec3 mx, vec3 p, float r) {
    vec3  nearest = clamp(p, mn, mx);
    float d2      = dot(p - nearest, p - nearest);
    return d2 <= r * r;
}

// ─────────────────────────────────────────────────────────────────────────────
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= u_n) return;

    col[i] = vec4(0.0);

    uint nf = nflags[i];
    if ((nf & SIM_IGNORE_BIT)    != 0u) return;
    if ((nf & COLLIDE_WALLS_BIT) == 0u) return;
    bool is_anchored       = (nf & ANCHORED_BIT)         != 0u;
    bool collide_when_anch = (nf & COLLIDE_ANCHORED_BIT) != 0u;
    if (is_anchored && !collide_when_anch) return;

    vec3  p      = pos_r[i].xyz;
    vec3  v      = vel[i].xyz;
    float radius = pos_r[i].w;
    if (radius <= 0.0 || u_n_tris <= 0) return;

    vec3 force = vec3(0.0);

    // ── BVH traversal (iterative, stack-based) ────────────────────────────────
    int stk[32];
    int top = 0;
    stk[top++] = 0;  // root node

    while (top > 0) {
        int node = stk[--top];

        vec3 mn = vec3(intBitsToFloat(bvh_nodes[node*10 + 0]),
                       intBitsToFloat(bvh_nodes[node*10 + 1]),
                       intBitsToFloat(bvh_nodes[node*10 + 2]));
        vec3 mx = vec3(intBitsToFloat(bvh_nodes[node*10 + 4]),
                       intBitsToFloat(bvh_nodes[node*10 + 5]),
                       intBitsToFloat(bvh_nodes[node*10 + 6]));

        if (!aabb_sphere_overlap(mn, mx, p, radius)) continue;

        int tri_count = bvh_nodes[node*10 + 8];
        if (tri_count > 0) {
            // ── Leaf: run contact test for each triangle ─────────────────────
            int first = bvh_nodes[node*10 + 7];
            for (int k = 0; k < tri_count; k++) {
                int orig_tri = wscratch[(first + k)*8 + 6]; // original index into widx[]

                int ai = widx[orig_tri*4 + 0];
                int bi = widx[orig_tri*4 + 1];
                int ci = widx[orig_tri*4 + 2];
                if (ai == i || bi == i || ci == i) continue;  // no self-collision

                vec3 A = pos_r[ai].xyz;
                vec3 B = pos_r[bi].xyz;
                vec3 C = pos_r[ci].xyz;

                vec3  normal;
                float pen = sphere_tri(p, radius, A, B, C, normal);
                if (pen <= 0.0) continue;

                // Contact force
                float N_mag = pen * u_stiffness;
                vec3  sep   = normal * N_mag;

                // Normal damping
                float nvel = dot(v, normal);
                if (nvel < 0.0) sep -= normal * (nvel * u_damp);

                // Coulomb friction (kinetic, capped at μN, with viscous regularisation)
                vec3  tang = v - normal * nvel;
                float tlen = length(tang);
                if (tlen > 1e-9) {
                    float coulomb_limit = u_friction_kt * N_mag;
                    float viscous_cap   = u_damp * tlen;
                    float fric_mag      = min(coulomb_limit, viscous_cap);
                    sep -= normalize(tang) * fric_mag;
                }

                force += sep;
            }
        } else {
            // ── Internal: push overlapping children ──────────────────────────
            int left  = bvh_nodes[node*10 + 3];
            int right = bvh_nodes[node*10 + 7];
            if (left  >= 0) stk[top++] = left;
            if (right >= 0) stk[top++] = right;
        }
    }

    col[i] = vec4(force, 0.0);
}
