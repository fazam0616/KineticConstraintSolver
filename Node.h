#ifndef NODE_H
#define NODE_H

#include <stdbool.h>
#include "datastructures.h"

typedef struct Node Node;

struct Node {
    int idx;
    float mass;
    float pos[3];
    float vel[3];
    DynArray *constraints; // Constraint*
    float radius;
    bool isGravity;
    bool anchored;
    bool collide_when_anchored;
    bool collide_with_walls;
    float friction;
    bool sim_ignore;
    bool dirty;
    float prev_position[3];
};

Node* node_create(int idx, float mass, float x, float y, float z);
void node_free(Node *n);
void node_draw(Node *n, float radius, float cam_yaw, float cam_pitch);

#endif // NODE_H
