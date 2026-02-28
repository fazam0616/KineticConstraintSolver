#include "Node.h"
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
        glColor3f(0.2f, 0.2f, 0.2f);
        draw_sphere_wire(n->pos[0], n->pos[1], n->pos[2], radius * 0.6f, cam_yaw, cam_pitch, 12);
    } else {
        // regular node: filled red sphere
        glColor3f(1.0f, 0.0f, 0.0f);
        draw_sphere(n->pos[0], n->pos[1], n->pos[2], radius, cam_yaw, cam_pitch, 24);
        // subtle dark outline
        glColor3f(0.0f, 0.0f, 0.0f);
        draw_sphere_wire(n->pos[0], n->pos[1], n->pos[2], radius, cam_yaw, cam_pitch, 16);
    }
}
