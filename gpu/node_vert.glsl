#version 430

// Instanced sphere rendering: one sphere per node, read from SSBO.
// a_offset is a unit-sphere vertex: position on a unit sphere centered at origin.
// Also acts as the outward normal (no matrix needed for normals on a unit sphere).
layout(location = 0) in vec3 a_offset;

layout(std430, binding = 0) readonly buffer Positions { vec4 pos_r[]; }; // .xyz=pos  .w=radius
layout(std430, binding = 4) readonly buffer NodeFlags  { uint flags[];  };

uniform mat4 u_mvp;

flat out uint v_flags;
out  vec3 v_normal_ws;

void main() {
    vec3  center = pos_r[gl_InstanceID].xyz;
    float radius = pos_r[gl_InstanceID].w;
    v_flags      = flags[gl_InstanceID];
    v_normal_ws  = a_offset;  // unit-sphere: normal == vertex offset
    gl_Position  = u_mvp * vec4(center + radius * a_offset, 1.0);
}
