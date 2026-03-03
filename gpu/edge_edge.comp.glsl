// edge_edge.comp.glsl
// Edge-edge capsule collision between all pairs of CT_DIST constraints.
// One thread per (edgeA, edgeB) pair using 2-D indexing over m_total × m_total.
// Velocity impulses written atomically to ssbo_velcorr (binding 18, uint[n*3]).
// apply_corr.comp.glsl reads+clears velcorr every substep.
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding =  0) readonly buffer PosBuf    { vec4  pos_r[];    }; // .w = radius
layout(std430, binding =  1) readonly buffer VelBuf    { vec4  vel4[];     };
layout(std430, binding =  2) readonly buffer InvMBuf   { float inv_mass[]; };
layout(std430, binding =  4) readonly buffer FlagsBuf  { uint  nflags[];   };
layout(std430, binding =  5) readonly buffer ConBuf    { int   cdata[];    }; // 8 ints/entry
layout(std430, binding = 18)          buffer VelCorrBuf{ uint  velcorr[];  }; // n*3 uint

uniform int   u_n;           // node count
uniform int   u_m_total;     // total constraint count
uniform float u_restitution; // e.g. 0.05
uniform float u_mu;          // kinetic friction coefficient

const int  CT_DIST        = 0;
const uint ANCHORED_BIT   = 1u;
const uint SIM_IGNORE_BIT = 4u;

// ── Atomic float add via CAS loop ─────────────────────────────────────────────
void atomicAddF(uint idx, float val) {
    uint assumed, next;
    uint old = velcorr[idx];
    do {
        assumed = old;
        next    = floatBitsToUint(uintBitsToFloat(assumed) + val);
        old     = atomicCompSwap(velcorr[idx], assumed, next);
    } while (old != assumed);
}

// ── Segment-to-segment closest points (Ericson, Real-Time Collision Detection) ─
void closest_seg_seg(vec3 A, vec3 B, vec3 C, vec3 D,
                     out float sc, out float tc) {
    vec3  u = B - A, v = D - C, w = A - C;
    float a = dot(u,u), b = dot(u,v), c = dot(v,v);
    float d = dot(u,w), e = dot(v,w);
    float det = a*c - b*b;
    float sN, sD = det, tN, tD = det;

    if (det < 1e-8) {
        sN = 0.0; sD = 1.0; tN = e; tD = c;
    } else {
        sN = b*e - c*d;
        tN = a*e - b*d;
        if      (sN < 0.0) { sN = 0.0; tN = e;     tD = c; }
        else if (sN > sD)  { sN = sD;  tN = e + b;  tD = c; }
    }
    if (tN < 0.0) {
        tN = 0.0;
        if      (-d < 0.0) sN = 0.0;
        else if (-d > a)   sN = sD;
        else { sN = -d; sD = a; }
    } else if (tN > tD) {
        tN = tD;
        if      ((-d+b) < 0.0) sN = 0.0;
        else if ((-d+b) > a)   sN = sD;
        else { sN = (-d+b); sD = a; }
    }
    sc = (abs(sN) < 1e-8) ? 0.0 : sN / sD;
    tc = (abs(tN) < 1e-8) ? 0.0 : tN / tD;
}

// ── Distribute a velocity delta to a node via atomic add ─────────────────────
void addVelCorr(int node_idx, vec3 dv) {
    uint base = uint(node_idx) * 3u;
    atomicAddF(base + 0u, dv.x);
    atomicAddF(base + 1u, dv.y);
    atomicAddF(base + 2u, dv.z);
}

// ─────────────────────────────────────────────────────────────────────────────
void main() {
    uint tid = gl_GlobalInvocationID.x;
    int  ei = int(tid) / u_m_total;   // constraint (edge) index A
    int  ej = int(tid) % u_m_total;   // constraint (edge) index B
    if (ei >= u_m_total || ej >= u_m_total) return;
    if (ei >= ej) return;  // upper-triangle only, avoid double processing

    // Both must be distance (rod/edge) constraints
    if (cdata[ei*8 + 0] != CT_DIST) return;
    if (cdata[ej*8 + 0] != CT_DIST) return;

    int ai = cdata[ei*8 + 1];   // edge i: node indices
    int bi = cdata[ei*8 + 2];
    int aj = cdata[ej*8 + 1];   // edge j: node indices
    int bj = cdata[ej*8 + 2];

    // Skip if edges share an endpoint
    if (ai == aj || ai == bj || bi == aj || bi == bj) return;

    // Live positions
    vec3 A = pos_r[ai].xyz;
    vec3 B = pos_r[bi].xyz;
    vec3 C = pos_r[aj].xyz;
    vec3 D = pos_r[bj].xyz;

    // Per-edge capsule radii from average node radii
    float ri = (pos_r[ai].w + pos_r[bi].w) * 0.5;
    float rj = (pos_r[aj].w + pos_r[bj].w) * 0.5;
    float min_dist = ri + rj;

    float sc, tc;
    closest_seg_seg(A, B, C, D, sc, tc);

    vec3  pa   = A + sc * (B - A);
    vec3  pb   = C + tc * (D - C);
    vec3  diff = pa - pb;
    float dist = length(diff);

    if (dist >= min_dist || dist < 1e-9) return;

    vec3  normal = diff / dist;          // points from edge j toward edge i
    float pen    = min_dist - dist;

    // ── Barycentric weights along each edge ──────────────────────────────────
    float wA0 = 1.0 - sc;   // weight for ai
    float wA1 = sc;          // weight for bi
    float wB0 = 1.0 - tc;   // weight for aj
    float wB1 = tc;          // weight for bj

    // ── Interpolated relative velocity at contact point ──────────────────────
    vec3 va  = vel4[ai].xyz * wA0 + vel4[bi].xyz * wA1;
    vec3 vb  = vel4[aj].xyz * wB0 + vel4[bj].xyz * wB1;
    vec3 rel = va - vb;
    float nvel = dot(rel, normal);

    // Only resolve closing contacts
    if (nvel >= 0.0) return;

    // ── Effective inverse mass ───────────────────────────────────────────────
    bool anchA0 = (nflags[ai] & (ANCHORED_BIT | SIM_IGNORE_BIT)) != 0u;
    bool anchA1 = (nflags[bi] & (ANCHORED_BIT | SIM_IGNORE_BIT)) != 0u;
    bool anchB0 = (nflags[aj] & (ANCHORED_BIT | SIM_IGNORE_BIT)) != 0u;
    bool anchB1 = (nflags[bj] & (ANCHORED_BIT | SIM_IGNORE_BIT)) != 0u;

    float imA0 = anchA0 ? 0.0 : inv_mass[ai];
    float imA1 = anchA1 ? 0.0 : inv_mass[bi];
    float imB0 = anchB0 ? 0.0 : inv_mass[aj];
    float imB1 = anchB1 ? 0.0 : inv_mass[bj];

    float weff = wA0*wA0*imA0 + wA1*wA1*imA1 + wB0*wB0*imB0 + wB1*wB1*imB1;
    if (weff <= 1e-12) return;

    // ── Normal impulse magnitude ─────────────────────────────────────────────
    float J = -(1.0 + u_restitution) * nvel / weff;
    J = min(J, 1e4);

    // ── Baumgarte penetration correction (added to normal impulse) ───────────
    const float baumgarte = 0.2;
    float Jpen = baumgarte * pen / weff;
    float Jtot = J + Jpen;

    // ── Tangential friction impulse ──────────────────────────────────────────
    vec3  tang = rel - normal * nvel;
    float tlen = length(tang);
    float jt   = 0.0;
    vec3  t_dir = vec3(0.0);
    if (tlen > 1e-9) {
        t_dir = tang / tlen;
        // Desired jt to cancel tangential relative velocity, clamped by Coulomb
        float jt_desired = -dot(rel, t_dir) / weff;
        float coulomb    = u_mu * abs(J);
        jt = clamp(jt_desired, -coulomb, coulomb);
    }

    // ── Impulse vectors ──────────────────────────────────────────────────────
    vec3 impulse = Jtot * normal + jt * t_dir;

    // Edge i nodes: receive +impulse (pushed away from edge j)
    if (!anchA0) addVelCorr(ai, ( impulse) * (wA0 * imA0));
    if (!anchA1) addVelCorr(bi, ( impulse) * (wA1 * imA1));
    // Edge j nodes: receive -impulse
    if (!anchB0) addVelCorr(aj, (-impulse) * (wB0 * imB0));
    if (!anchB1) addVelCorr(bj, (-impulse) * (wB1 * imB1));
}
