#version 430

// Constraint lines: glDrawArrays(GL_LINES, 0, m_total * 2)
// Vertex i -> constraint i/2, endpoint i%2 (0=nodeA, 1=nodeB)

layout(std430, binding = 0) readonly buffer Positions   { vec4 pos_r[]; };
layout(std430, binding = 5) readonly buffer Constraints { int  cdata[];  }; // 8 ints per constraint

uniform mat4 u_mvp;

flat out int v_ctype;

void main() {
    int ci    = gl_VertexID / 2;
    int which = gl_VertexID % 2;           // 0 = nodeA, 1 = nodeB
    v_ctype   = cdata[ci * 8 + 0];
    int idx   = (which == 0) ? cdata[ci * 8 + 1] : cdata[ci * 8 + 2];
    vec3 p    = (idx >= 0) ? pos_r[idx].xyz : vec3(0.0);
    gl_Position = u_mvp * vec4(p, 1.0);
}
