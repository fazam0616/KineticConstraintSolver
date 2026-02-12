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

static void draw_circle_billboard(float cx, float cy, float cz, float r, 
                                  float cam_yaw, float cam_pitch, int segments) {
    // Get camera right and up vectors for billboard
    float right_x = cosf(cam_yaw);
    float right_y = 0.0f;
    float right_z = -sinf(cam_yaw);
    
    float up_x = sinf(cam_yaw) * sinf(cam_pitch);
    float up_y = cosf(cam_pitch);
    float up_z = cosf(cam_yaw) * sinf(cam_pitch);
    
    glBegin(GL_TRIANGLE_FAN);
    glVertex3f(cx, cy, cz); // center
    for (int i = 0; i <= segments; ++i) {
        float theta = 2.0f * 3.1415926f * (float)i / (float)segments;
        float offset_x = r * (cosf(theta) * right_x + sinf(theta) * up_x);
        float offset_y = r * (cosf(theta) * right_y + sinf(theta) * up_y);
        float offset_z = r * (cosf(theta) * right_z + sinf(theta) * up_z);
        glVertex3f(cx + offset_x, cy + offset_y, cz + offset_z);
    }
    glEnd();
}

static void draw_circle_outline_billboard(float cx, float cy, float cz, float r, 
                                         float cam_yaw, float cam_pitch, int segments) {
    // Get camera right and up vectors for billboard
    float right_x = cosf(cam_yaw);
    float right_y = 0.0f;
    float right_z = -sinf(cam_yaw);
    
    float up_x = sinf(cam_yaw) * sinf(cam_pitch);
    float up_y = cosf(cam_pitch);
    float up_z = cosf(cam_yaw) * sinf(cam_pitch);
    
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < segments; ++i) {
        float theta = 2.0f * 3.1415926f * (float)i / (float)segments;
        float offset_x = r * (cosf(theta) * right_x + sinf(theta) * up_x);
        float offset_y = r * (cosf(theta) * right_y + sinf(theta) * up_y);
        float offset_z = r * (cosf(theta) * right_z + sinf(theta) * up_z);
        glVertex3f(cx + offset_x, cy + offset_y, cz + offset_z);
    }
    glEnd();
}

void node_draw(Node *n, float radius, float cam_yaw, float cam_pitch) {
    if (!n) return;
    if (n->anchored) {
        // draw small black dot for anchors
        glColor3f(0.0f, 0.0f, 0.0f);
        draw_circle_billboard(n->pos[0], n->pos[1], n->pos[2], radius * 0.6f, 
                             cam_yaw, cam_pitch, 20);
    } else {
        // regular node: filled red
        glColor3f(1.0f, 0.0f, 0.0f);
        draw_circle_billboard(n->pos[0], n->pos[1], n->pos[2], radius, 
                             cam_yaw, cam_pitch, 20);
    }
    // small outline
    glColor3f(0.0f, 0.0f, 0.0f);
    draw_circle_outline_billboard(n->pos[0], n->pos[1], n->pos[2], radius, 
                                 cam_yaw, cam_pitch, 20);
}
