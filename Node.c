#include "Node.h"
#include "Shader.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <GL/gl.h>

Node* node_create(int idx, float mass, float x, float y, float z) {
    Node *n = (Node*)malloc(sizeof(Node));
    n->idx = idx;
    n->mass = mass;
    n->pos[0] = x; n->pos[1] = y; n->pos[2] = z;
    n->vel[0] = 0.0f; n->vel[1] = 0.0f; n->vel[2] = 0.0f;
    n->constraints = dynarray_create(4);
    n->isGravity = true;
    n->anchored = false;
    n->collide_when_anchored = true;
    n->collide_with_walls = true;
    n->friction = 0.5f;
    n->radius = 2*mass;
    n->sim_ignore = false;
    n->dirty = false;
    n->prev_position[0] = x;
    n->prev_position[1] = y;
    n->prev_position[2] = z;
    return n;
}

void node_free(Node *n) {
    if (!n) return;
    if (n->constraints) dynarray_free(n->constraints, NULL);
    free(n);
}

// Draw a UV-sphere centered at (cx,cy,cz) with radius r.
static void draw_sphere(float cx, float cy, float cz, float r,
                        float cam_yaw, float cam_pitch, int segments) {
    int slices = segments;
    int stacks = segments / 2;
    const float PI = 3.14159265358979323846f;
    if (g_phong_program != 0) {
        // compute MVP once from current matrices
        float mv[16], pr[16], mvp[16];
        glGetFloatv(GL_MODELVIEW_MATRIX, mv);
        glGetFloatv(GL_PROJECTION_MATRIX, pr);
        for (int ii = 0; ii < 4; ++ii) for (int jj = 0; jj < 4; ++jj) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) s += pr[ii*4 + k] * mv[k*4 + jj];
            mvp[ii*4 + jj] = s;
        }

        // set common uniforms
        glUseProgram(g_phong_program);
        GLint loc_mvp = glGetUniformLocation(g_phong_program, "u_MVP");
        GLint loc_light = glGetUniformLocation(g_phong_program, "u_lightDir");
        GLint loc_mat = glGetUniformLocation(g_phong_program, "u_materialColor");
        GLint loc_shine = glGetUniformLocation(g_phong_program, "u_shininess");
        if (loc_mvp >= 0) glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp);
        if (loc_light >= 0) {
            float lx = 0.5f, ly = -1.0f, lz = 0.5f;
            float llen = sqrtf(lx*lx + ly*ly + lz*lz);
            if (llen > 1e-8f) { lx /= llen; ly /= llen; lz /= llen; }
            glUniform3f(loc_light, lx, ly, lz);
        }
        if (loc_mat >= 0) glUniform3f(loc_mat, 1.0f, 0.0f, 0.0f);
        if (loc_shine >= 0) glUniform1f(loc_shine, 32.0f);

        // draw each stack as a triangle strip using client arrays bound to attribs
        int verts_per_stack = (slices + 1) * 2;
        float *pos = (float*)malloc(sizeof(float) * verts_per_stack * 3);
        float *nrm = (float*)malloc(sizeof(float) * verts_per_stack * 3);
        for (int i = 0; i < stacks; ++i) {
            float lat0 = PI * (-0.5f + (float)i / (float)stacks);
            float z0 = sinf(lat0);
            float zr0 = cosf(lat0);

            float lat1 = PI * (-0.5f + (float)(i+1) / (float)stacks);
            float z1 = sinf(lat1);
            float zr1 = cosf(lat1);

            int idx = 0;
            for (int j = 0; j <= slices; ++j) {
                float lng = 2.0f * PI * (float)(j) / (float)slices;
                float x = cosf(lng);
                float y = sinf(lng);
                // first latitude vertex
                nrm[idx*3 + 0] = x * zr0; nrm[idx*3 + 1] = y * zr0; nrm[idx*3 + 2] = z0;
                pos[idx*3 + 0] = cx + r * x * zr0; pos[idx*3 + 1] = cy + r * y * zr0; pos[idx*3 + 2] = cz + r * z0;
                idx++;
                // next latitude vertex
                nrm[idx*3 + 0] = x * zr1; nrm[idx*3 + 1] = y * zr1; nrm[idx*3 + 2] = z1;
                pos[idx*3 + 0] = cx + r * x * zr1; pos[idx*3 + 1] = cy + r * y * zr1; pos[idx*3 + 2] = cz + r * z1;
                idx++;
            }

            glEnableVertexAttribArray(0);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, pos);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nrm);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, verts_per_stack);
            glDisableVertexAttribArray(0);
            glDisableVertexAttribArray(1);
        }
        free(pos); free(nrm);
        glUseProgram(0);
    } else {
        for (int i = 0; i < stacks; ++i) {
            float lat0 = PI * (-0.5f + (float)i / (float)stacks);
            float z0 = sinf(lat0);
            float zr0 = cosf(lat0);

            float lat1 = PI * (-0.5f + (float)(i+1) / (float)stacks);
            float z1 = sinf(lat1);
            float zr1 = cosf(lat1);

            glBegin(GL_TRIANGLE_STRIP);
            for (int j = 0; j <= slices; ++j) {
                float lng = 2.0f * PI * (float)(j) / (float)slices;
                float x = cosf(lng);
                float y = sinf(lng);

                // Vertex on first latitude
                glNormal3f(x * zr0, y * zr0, z0);
                glVertex3f(cx + r * x * zr0, cy + r * y * zr0, cz + r * z0);

                // Vertex on next latitude
                glNormal3f(x * zr1, y * zr1, z1);
                glVertex3f(cx + r * x * zr1, cy + r * y * zr1, cz + r * z1);
            }
            glEnd();
        }
    }
}

// Draw a simple wireframe sphere (latitude + longitude lines)
static void draw_sphere_wire(float cx, float cy, float cz, float r,
                             float cam_yaw, float cam_pitch, int segments) {
    int slices = segments;
    int stacks = segments / 2;
    const float PI = 3.14159265358979323846f;
    // latitude rings
    for (int i = 0; i <= stacks; ++i) {
        float lat = PI * (-0.5f + (float)i / (float)stacks);
        float z = sinf(lat);
        float zr = cosf(lat);
        glBegin(GL_LINE_LOOP);
        for (int j = 0; j < slices; ++j) {
            float lng = 2.0f * PI * (float)j / (float)slices;
            float x = cosf(lng);
            float y = sinf(lng);
            glVertex3f(cx + r * x * zr, cy + r * y * zr, cz + r * z);
        }
        glEnd();
    }
    // longitude lines
    for (int j = 0; j < slices; ++j) {
        float lng = 2.0f * PI * (float)j / (float)slices;
        float x = cosf(lng);
        float y = sinf(lng);
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i <= stacks; ++i) {
            float lat = PI * (-0.5f + (float)i / (float)stacks);
            float z = sinf(lat);
            float zr = cosf(lat);
            glVertex3f(cx + r * x * zr, cy + r * y * zr, cz + r * z);
        }
        glEnd();
    }
}

void node_draw(Node *n, float radius, float cam_yaw, float cam_pitch) {
    if (!n) return;
    if (n->anchored) {
        // draw small black sphere for anchors
        glColor3f(0.0f, 0.0f, 0.0f);
        draw_sphere(n->pos[0], n->pos[1], n->pos[2], radius * 0.6f, cam_yaw, cam_pitch, 16);
        if (g_phong_program == 0) {
            glColor3f(0.2f, 0.2f, 0.2f);
            draw_sphere_wire(n->pos[0], n->pos[1], n->pos[2], radius * 0.6f, cam_yaw, cam_pitch, 12);
        }
    } else {
        // regular node: filled red sphere
        glColor3f(1.0f, 0.0f, 0.0f);
        draw_sphere(n->pos[0], n->pos[1], n->pos[2], radius, cam_yaw, cam_pitch, 24);
        // subtle dark outline only if shader not active
        if (g_phong_program == 0) {
            glColor3f(0.0f, 0.0f, 0.0f);
            draw_sphere_wire(n->pos[0], n->pos[1], n->pos[2], radius, cam_yaw, cam_pitch, 16);
        }
    }
}
