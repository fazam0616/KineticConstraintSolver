#version 430

// Live positions: .xyz = world pos, .w = radius (unused here)
layout(std430, binding =  0) readonly buffer PosBuf     { vec4 pos_r[]; };
// Wall node indices: 4 ints per triangle [a_idx, b_idx, c_idx, pad]
layout(std430, binding = 17) readonly buffer WallIdxBuf { int  widx[];  };

uniform mat4 u_mvp;

out vec3 v_normal_ws;

void main() {
    int ti = gl_VertexID / 3;    // triangle index
    int vi = gl_VertexID % 3;    // vertex within triangle (0=A, 1=B, 2=C)

    int ai = widx[ti * 4 + 0];
    int bi = widx[ti * 4 + 1];
    int ci = widx[ti * 4 + 2];

    vec3 A = pos_r[ai].xyz;
    vec3 B = pos_r[bi].xyz;
    vec3 C = pos_r[ci].xyz;

    v_normal_ws = normalize(cross(B - A, C - A));

    vec3 p = (vi == 0) ? A : (vi == 1) ? B : C;
    gl_Position = u_mvp * vec4(p, 1.0);
}
