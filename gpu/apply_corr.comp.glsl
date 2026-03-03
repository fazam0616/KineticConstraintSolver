// apply_corr.comp.glsl
// One thread per node.
// 1. Computes corr_f[node_dof] = -dt * (J^T * l)[node_dof]  by scanning all J rows.
// 2. Reads collision forces from binding 11 (CPU-uploaded per substep).
// 3. Applies: vel += inv_mass * (corr_f * N + ext_forces + collision_forces) * sub_dt
// 4. Applies: pos += vel * sub_dt
//
// Mirrors the final integration section inside the simulator_step() substep loop.
#version 430 core
layout(local_size_x = 64) in;

layout(std430, binding = 0)  buffer   PosSSBO       { vec4  pos4[];        }; // rw
layout(std430, binding = 1)  buffer   VelSSBO       { vec4  vel4[];        }; // rw
layout(std430, binding = 2)  readonly buffer InvMassSSBO { float inv_mass[]; };
layout(std430, binding = 3)  readonly buffer ExtForceBuf { vec4  ext_forces[]; }; // gravity+springs
layout(std430, binding = 4)  readonly buffer FlagsSSBO   { uint  node_flags[]; };
layout(std430, binding = 6)  readonly buffer JColsBuf    { int   J_cols[];   };
layout(std430, binding = 7)  readonly buffer JValsBuf    { float J_vals[];   };
layout(std430, binding = 10) readonly buffer LVecBuf     { float l_data[];   };
layout(std430, binding = 11) readonly buffer CollisionBuf{ vec4  coll_forces[];};// collision (CPU)
layout(std430, binding = 18)          buffer VelCorrBuf  { uint  velcorr[];    };// edge-edge impulses

uniform int   u_n;      // node count
uniform int   u_m;      // m_sparse (number of sparse constraints)
uniform float u_dt;     // full dt
uniform float u_sub_dt; // sub timestep = dt / N
uniform int   u_N;      // number of substeps (solver_iters)

#define ANCHORED_BIT   1u
#define SIM_IGNORE_BIT 4u

void main() {
    int i = int(gl_GlobalInvocationID.x);
    if (i >= u_n) return;

    // --- Read & clear edge-edge velocity correction (must happen for every node,
    //     including anchored, to keep the buffer zeroed for the next substep) ---
    uint base = uint(i) * 3u;
    vec3 ee_dv = vec3(uintBitsToFloat(velcorr[base + 0u]),
                      uintBitsToFloat(velcorr[base + 1u]),
                      uintBitsToFloat(velcorr[base + 2u]));
    velcorr[base + 0u] = 0u;
    velcorr[base + 1u] = 0u;
    velcorr[base + 2u] = 0u;

    uint flags = node_flags[i];
    if ((flags & ANCHORED_BIT) != 0u || (flags & SIM_IGNORE_BIT) != 0u) return;

    // --- Compute corr_f for this node by scanning all J rows ---
    // corr_f[dof] = -dt * sum_c { J_vals[c*6+k] * l[c]  for all k where J_cols[c*6+k] == 3*i+dof }
    vec3 corr_f = vec3(0.0);
    for (int c = 0; c < u_m; c++) {
        float lc = l_data[c];
        for (int k = 0; k < 6; k++) {
            int col = J_cols[c*6 + k];
            if (col < 0) continue;
            if (col / 3 != i) continue;          // different node
            int dof = col % 3;                   // 0=x 1=y 2=z
            corr_f[dof] += J_vals[c*6 + k] * lc;
        }
    }
    corr_f *= -u_dt; // corr_f = -dt * J^T * l  (matches CPU: cs_gaxpy + scale by -dt)

    // --- Accumulate all forces ---
    vec3 ext  = ext_forces[i].xyz;
    vec3 coll = coll_forces[i].xyz;

    // CPU mirrors: fcx = corr_f[3*i]*N, total_fx = fcx + extx, ax = invm*total_fx, vel+=ax*sub_dt
    vec3 total_f = corr_f * float(u_N) + ext + coll;

    float im     = inv_mass[i];
    vec3  accel  = im * total_f;

    // --- Symplectic-Euler integration ---
    // ee_dv: velocity impulse from edge-edge collision (computed before this shader ran)
    vec3 v = vel4[i].xyz + accel * u_sub_dt + ee_dv;
    vel4[i] = vec4(v, 0.0);
    pos4[i] = vec4(pos4[i].xyz + v * u_sub_dt, pos4[i].w);
}
