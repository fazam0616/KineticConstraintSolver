#version 430
layout(local_size_x = 64) in;

// Brute-force sphere-triangle collision using LIVE node positions.
// Each thread = one physics node.  Iterates all wall triangles.
// Wall triangle vertices are looked up from ssbo_positions (binding 0)
// using per-triangle node indices stored in ssbo_wall_indices (binding 17).

layout(std430, binding =  0) buffer Positions  { vec4 pos_r[]; };  // .xyz=pos .w=radius
layout(std430, binding =  1) buffer Velocities { vec4 vel[];   };
layout(std430, binding =  4) buffer NodeFlags  { uint nflags[];};
layout(std430, binding = 11) buffer ColOut     { vec4 col[];   };
layout(std430, binding = 17) buffer WallIdxBuf { int  widx[];  };  // 4 ints/tri [a,b,c,pad]

uniform int   u_n;              // total node count
uniform int   u_n_tris;         // number of wall triangles
uniform float u_stiffness;      // penetration stiffness  (e.g. 500)
uniform float u_damp;           // normal damping coeff   (e.g. 100)
uniform float u_friction_kt;    // tangential friction    (e.g. 150)

// Node flag bits (match Simulator.c packing)
const uint ANCHORED_BIT         = 1u;
const uint COLLIDE_WALLS_BIT    = 2u;
const uint SIM_IGNORE_BIT       = 4u;
const uint COLLIDE_ANCHORED_BIT = 16u;  // bit 4

// ── Sphere–triangle contact ───────────────────────────────────────────────────
// Returns penetration depth > 0 if collision, else <= 0.
// out_normal points from triangle surface toward sphere center.
float sphere_tri(vec3 p, float r, vec3 A, vec3 B, vec3 C, out vec3 out_normal) {
    vec3 AB = B - A;
    vec3 AC = C - A;
    vec3 n  = cross(AB, AC);
    float nlen = length(n);
    if (nlen < 1e-9) { out_normal = vec3(0, 1, 0); return -1.0; }
    out_normal = n / nlen;

    float dist = dot(p - A, out_normal);
    if (abs(dist) > r + 1e-3) return -1.0;

    // Project p onto triangle plane and do barycentric test
    vec3 proj = p - dist * out_normal;
    vec3 vp   = proj - A;
    float d00 = dot(AB, AB), d01 = dot(AB, AC), d11 = dot(AC, AC);
    float d20 = dot(vp, AB), d21 = dot(vp, AC);
    float denom = d00 * d11 - d01 * d01;
    if (abs(denom) < 1e-18) return -1.0;
    float inv = 1.0 / denom;
    float u = (d11 * d20 - d01 * d21) * inv;
    float v = (d00 * d21 - d01 * d20) * inv;
    if (u < -0.01 || v < -0.01 || u + v > 1.01) return -1.0;

    float pen = r - abs(dist);
    if (pen <= 0.0) return -1.0;

    if (dist < 0.0) out_normal = -out_normal;
    return pen;
}

// ─────────────────────────────────────────────────────────────────────────────
void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= u_n) return;

    col[i] = vec4(0.0);  // clear output regardless

    uint nf = nflags[i];
    if ((nf & SIM_IGNORE_BIT)    != 0u) return;
    if ((nf & COLLIDE_WALLS_BIT) == 0u) return;
    bool is_anchored       = (nf & ANCHORED_BIT)         != 0u;
    bool collide_when_anch = (nf & COLLIDE_ANCHORED_BIT) != 0u;
    if (is_anchored && !collide_when_anch) return;

    vec3  p      = pos_r[i].xyz;
    vec3  v      = vel[i].xyz;
    float radius = pos_r[i].w;
    if (radius <= 0.0) return;
    if (u_n_tris <= 0) return;

    vec3 force = vec3(0.0);

    // ── Brute-force: test sphere against every wall triangle ─────────────────
    for (int t = 0; t < u_n_tris; t++) {
        int ai = widx[t * 4 + 0];
        int bi = widx[t * 4 + 1];
        int ci = widx[t * 4 + 2];

        // Skip triangles this node is a vertex of — no self-collision.
        if (ai == i || bi == i || ci == i) continue;

        // Use LIVE positions from ssbo_positions — always up to date
        vec3 A = pos_r[ai].xyz;
        vec3 B = pos_r[bi].xyz;
        vec3 C = pos_r[ci].xyz;

        vec3  normal;
        float pen = sphere_tri(p, radius, A, B, C, normal);
        if (pen <= 0.0) continue;

        // Normal force magnitude (pure stiffness — used for Coulomb bound)
        float N_mag = pen * u_stiffness;

        // Normal contact force
        vec3 sep = normal * N_mag;

        // Normal damping: oppose penetration velocity
        float nvel = dot(v, normal);
        if (nvel < 0.0) sep -= normal * (nvel * u_damp);

        // Coulomb friction: opposes tangential sliding, capped at μ × N
        // u_friction_kt is the dimensionless kinetic friction coefficient μ.
        vec3  tang = v - normal * nvel;   // tangential velocity component
        float tlen = length(tang);
        if (tlen > 1e-9) {
            // Coulomb limit: μ × normal force magnitude
            float coulomb_limit = u_friction_kt * N_mag;
            // Viscous cap: prevents overshooting when tlen is near zero
            // (equivalent to a maximum static deceleration per timestep).
            // u_damp reused as the viscous cap coefficient.
            float viscous_cap   = u_damp * tlen;
            float fric_mag      = min(coulomb_limit, viscous_cap);
            sep -= normalize(tang) * fric_mag;
        }

        force += sep;
    }

    col[i] = vec4(force, 0.0);
}
