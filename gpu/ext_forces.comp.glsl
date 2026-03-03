// ext_forces.comp.glsl
// One thread per node.  Computes gravity + spring external forces into ext_forces[].
// Collision forces are handled separately (CPU uploads to ssbo_collision, binding 11).
// Mirrors accumulation of gravity and CT_SPRING forces in simulator_step().
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly  buffer PosSSBO    { vec4  pos4[];       };
layout(std430, binding = 2) readonly  buffer InvMassSSBO{ float inv_mass[];   };
layout(std430, binding = 3) writeonly buffer ExtForceBuf{ vec4  ext_forces[]; }; // output: n*vec4
layout(std430, binding = 4) readonly  buffer FlagsSSBO  { uint  node_flags[]; };
// Full constraint list (non-spring first, springs after m_sparse)
layout(std430, binding = 5) readonly  buffer ConBuf     { int   con_data[];   };

uniform int   u_n;          // node count
uniform int   u_m_sparse;   // first u_m_sparse entries are non-spring
uniform int   u_m_total;    // total constraint count (including springs)
uniform vec3  u_gravity;    // gravity vector

#define ANCHORED_BIT   1u
#define SIM_IGNORE_BIT 4u
#define GRAVITY_BIT    8u
#define CT_SPRING      2

void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= u_n) return;

    uint flags = node_flags[i];

    // anchored / sim_ignore nodes contribute no external force
    if ((flags & ANCHORED_BIT) != 0u || (flags & SIM_IGNORE_BIT) != 0u) {
        ext_forces[i] = vec4(0.0);
        return;
    }

    vec3 f = vec3(0.0);

    // --- gravity ---
    if ((flags & GRAVITY_BIT) != 0u) {
        float im = inv_mass[i];
        float mass = (im > 1e-12) ? (1.0 / im) : 0.0;
        f += mass * u_gravity;
    }

    // --- spring forces: iterate constraint list looking for CT_SPRING ---
    // Springs are stored at indices u_m_sparse .. u_m_total-1
    for (int c = u_m_sparse; c < u_m_total; c++) {
        int ctype = con_data[c*8 + 0];
        if (ctype != CT_SPRING) continue; // only springs
        int   a_idx    = con_data[c*8 + 1];
        int   b_idx    = con_data[c*8 + 2];
        float rest     = intBitsToFloat(con_data[c*8 + 3]);
        float k_spring = intBitsToFloat(con_data[c*8 + 7]); // stiffness at slot 7

        if (a_idx != i && b_idx != i) continue; // not our node

        vec3  pa   = pos4[a_idx].xyz;
        vec3  pb   = pos4[b_idx].xyz;
        vec3  d    = pa - pb;        // a - b
        float dist = length(d);
        if (dist < 1e-9) continue;

        float mag = k_spring * (dist - rest);
        vec3  dir = d / dist;
        // force on a = -mag * dir  (pulls a toward b when stretched)
        if (a_idx == i) f += -mag * dir;
        else            f +=  mag * dir; // b_idx == i: opposite
    }

    ext_forces[i] = vec4(f, 0.0);
}
