#version 430
// Simple integration compute shader (GLSL) - operates on SoA SSBOs
layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer Positions {
    vec4 positions[]; // use vec4 for alignment; w unused
};
layout(std430, binding = 1) buffer Velocities {
    vec4 velocities[];
};
layout(std430, binding = 2) buffer InvMass {
    float inv_mass[];
};
layout(std430, binding = 3) buffer ExternalForces {
    vec4 ext_forces[];
};

uniform float u_dt;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    // bounds check should be ensured by dispatch call
    vec3 pos = positions[idx].xyz;
    vec3 vel = velocities[idx].xyz;
    float invm = inv_mass[idx];
    vec3 f = ext_forces[idx].xyz;
    // acceleration = invm * f
    vec3 a = invm * f;
    vel += a * u_dt;
    pos += vel * u_dt;
    velocities[idx] = vec4(vel, 0.0);
    positions[idx] = vec4(pos, 0.0);
}
