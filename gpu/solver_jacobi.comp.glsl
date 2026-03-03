#version 430
// Jacobi solver compute shader: reads dense A (row-major), b, l_cur -> writes l_next
layout(local_size_x = 64) in;

layout(std430, binding = 4) buffer A_buf { float A[]; };
layout(std430, binding = 5) buffer B_buf { float b[]; };
layout(std430, binding = 6) buffer L_cur { float l_cur[]; };
layout(std430, binding = 7) buffer L_next { float l_next[]; };

uniform int u_m; // matrix size

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(u_m)) return;
    int m = u_m;
    float diag = A[i * m + i];
    float sigma = 0.0;
    for (int j = 0; j < m; ++j) {
        if (j == int(i)) continue;
        sigma += A[i * m + j] * l_cur[j];
    }
    float newv = (b[i] - sigma) / (abs(diag) < 1e-12 ? 1e-12 : diag);
    l_next[i] = newv;
}
