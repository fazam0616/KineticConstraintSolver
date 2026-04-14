// edge_edge.comp.glsl
// Edge-edge capsule collision using a per-substep GPU-built edge BVH.
// One thread per CT_DIST edge.  Each thread traverses the edge BVH (binding 20+21)
// to find candidate edges, then resolves impulses for each candidate with higher index.
// Velocity impulses written atomically to ssbo_velcorr (binding 18, uint[n*3]).
// apply_corr.comp.glsl reads+clears velcorr every substep.
// bvh_build_edges.comp.glsl must run first each substep.
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding =  0) readonly buffer PosBuf    { vec4  pos_r[];    }; // .w = radius
layout(std430, binding =  1) readonly buffer VelBuf    { vec4  vel4[];     };
layout(std430, binding =  2) readonly buffer InvMBuf   { float inv_mass[]; };
layout(std430, binding =  4) readonly buffer FlagsBuf  { uint  nflags[];   };
layout(std430, binding =  5) readonly buffer ConBuf    { int   cdata[];    }; // 8 ints/entry
layout(std430, binding = 18)          buffer VelCorrBuf{ uint  velcorr[];  }; // n*3 uint
layout(std430, binding = 20) readonly buffer EBvhBuf   { int   ebvh[];     }; // edge BVH (10 ints/node)
layout(std430, binding = 21) readonly buffer EScratch  { int   escratch[]; }; // 8 ints/prim; [6]=orig_con_idx

uniform int   u_n;           // node count
uniform int   u_m_total;     // total constraint count
uniform float u_restitution; // e.g. 0.05
uniform float u_mu;          // kinetic friction coefficient

const int  CT_DIST        = 1;
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

// ── AABB–AABB overlap test ────────────────────────────────────────────────────
bool aabb_overlap(vec3 mn_a, vec3 mx_a, vec3 mn_b, vec3 mx_b) {
    return all(lessThanEqual(mn_a, mx_b)) && all(lessThanEqual(mn_b, mx_a));
}

// ── Resolve one edge-edge capsule pair ───────────────────────────────────────
void resolve_pair(int ei, int ej) {
    int ai = cdata[ei*8 + 1];   // edge i node indices
    int bi = cdata[ei*8 + 2];
    int aj = cdata[ej*8 + 1];   // edge j node indices
    int bj = cdata[ej*8 + 2];

    // Skip if edges share an endpoint
    if (ai == aj || ai == bj || bi == aj || bi == bj) return;

    float ri = (pos_r[ai].w + pos_r[bi].w) * 0.5;
    float rj = (pos_r[aj].w + pos_r[bj].w) * 0.5;
    float min_dist = ri + rj;

    vec3 A = pos_r[ai].xyz, B = pos_r[bi].xyz;
    vec3 C = pos_r[aj].xyz, D = pos_r[bj].xyz;

    float sc, tc;
    closest_seg_seg(A, B, C, D, sc, tc);

    vec3  pa   = A + sc * (B - A);
    vec3  pb   = C + tc * (D - C);
    vec3  diff = pa - pb;
    float dist = length(diff);

    if (dist >= min_dist || dist < 1e-9) return;

    vec3  normal = diff / dist;
    float pen    = min_dist - dist;

    float wA0 = 1.0 - sc, wA1 = sc;
    float wB0 = 1.0 - tc, wB1 = tc;

    vec3 va  = vel4[ai].xyz * wA0 + vel4[bi].xyz * wA1;
    vec3 vb  = vel4[aj].xyz * wB0 + vel4[bj].xyz * wB1;
    vec3 rel = va - vb;
    float nvel = dot(rel, normal);
    if (nvel >= 0.0) return;

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

    float J    = -(1.0 + u_restitution) * nvel / weff;
    J          = min(J, 1e4);
    float Jpen = 0.2 * pen / weff;   // Baumgarte
    float Jtot = J + Jpen;

    vec3  tang = rel - normal * nvel;
    float tlen = length(tang);
    float jt   = 0.0;
    vec3  t_dir = vec3(0.0);
    if (tlen > 1e-9) {
        t_dir = tang / tlen;
        float jt_desired = -dot(rel, t_dir) / weff;
        float coulomb    = u_mu * abs(J);
        jt = clamp(jt_desired, -coulomb, coulomb);
    }

    vec3 impulse = Jtot * normal + jt * t_dir;

    if (!anchA0) addVelCorr(ai, ( impulse) * (wA0 * imA0));
    if (!anchA1) addVelCorr(bi, ( impulse) * (wA1 * imA1));
    if (!anchB0) addVelCorr(aj, (-impulse) * (wB0 * imB0));
    if (!anchB1) addVelCorr(bj, (-impulse) * (wB1 * imB1));
}

// ─────────────────────────────────────────────────────────────────────────────
void main() {
    uint tid = gl_GlobalInvocationID.x;
    int  ei  = int(tid);
    if (ei >= u_m_total) return;
    if (cdata[ei*8 + 0] != CT_DIST) return;

    // Compute edge i's AABB (capsule expanded by radius)
    int  ai = cdata[ei*8 + 1], bi = cdata[ei*8 + 2];
    if (ai < 0 || bi < 0) return;
    float ri  = (pos_r[ai].w + pos_r[bi].w) * 0.5;
    vec3  A   = pos_r[ai].xyz, B = pos_r[bi].xyz;
    vec3  mn_i = min(A, B) - ri;
    vec3  mx_i = max(A, B) + ri;

    // ── Traverse edge BVH to find candidate edges ej > ei ────────────────────
    int stk[32];
    int top = 0;
    stk[top++] = 0;  // root

    while (top > 0) {
        int node = stk[--top];

        vec3 mn_n = vec3(intBitsToFloat(ebvh[node*10 + 0]),
                         intBitsToFloat(ebvh[node*10 + 1]),
                         intBitsToFloat(ebvh[node*10 + 2]));
        vec3 mx_n = vec3(intBitsToFloat(ebvh[node*10 + 4]),
                         intBitsToFloat(ebvh[node*10 + 5]),
                         intBitsToFloat(ebvh[node*10 + 6]));

        if (!aabb_overlap(mn_i, mx_i, mn_n, mx_n)) continue;

        int count = ebvh[node*10 + 8];
        if (count > 0) {
            // Leaf: test each prim
            int first = ebvh[node*10 + 7];
            for (int k = 0; k < count; k++) {
                int ej = escratch[(first + k)*8 + 6]; // original constraint index
                if (ej <= ei) continue;                // process each pair once
                if (cdata[ej*8 + 0] != CT_DIST) continue;
                resolve_pair(ei, ej);
            }
        } else {
            int left  = ebvh[node*10 + 3];
            int right = ebvh[node*10 + 7];
            if (left  >= 0) stk[top++] = left;
            if (right >= 0) stk[top++] = right;
        }
    }
}
