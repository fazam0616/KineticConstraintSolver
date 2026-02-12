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
    n->radius = 6.0f;
    n->sim_ignore = false;
    return n;
}

void node_free(Node *n) {
    if (!n) return;
    if (n->constraints) dynarray_free(n->constraints, NULL);
    free(n);
}

static void draw_circle_filled(float cx, float cy, float cz, float r, int segments) {
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(cx, cy, cz);
    for (int i = 0; i <= segments; ++i) {
        float theta = 2.0f * 3.1415926f * (float)i / (float)segments;
        float x = r * cosf(theta);
        float y = r * sinf(theta);
        glVertex3f(cx + x, cy + y, cz);
    }
    glEnd();
}

void node_draw(Node *n, float radius) {
    if (!n) return;
    if (n->anchored) {
        // draw small black dot for anchors
        glColor3f(0.0f, 0.0f, 0.0f);
        draw_circle_filled(n->pos[0], n->pos[1], n->pos[2], radius * 0.6f, 20);
    } else {
        // regular node: filled red
        glColor3f(1.0f, 0.0f, 0.0f);
        draw_circle_filled(n->pos[0], n->pos[1], n->pos[2], radius, 20);
    }
    // small outline
    glColor3f(0.0f, 0.0f, 0.0f);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 20; ++i) {
        float theta = 2.0f * 3.1415926f * (float)i / 20.0f;
        float x = radius * cosf(theta);
        float y = radius * sinf(theta);
        glVertex3f(n->pos[0] + x, n->pos[1] + y, n->pos[2]);
    }
    glEnd();
}
