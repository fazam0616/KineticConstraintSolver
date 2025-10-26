#ifndef NODE_H
#define NODE_H

#include <stdbool.h>
#include "datastructures.h"

typedef struct Node Node;

struct Node {
    int idx;
    float mass;
    float pos[2];
    float vel[2];
    DynArray *constraints; // Constraint*
    float radius;
    bool isGravity;
    bool anchored;
    bool collide_when_anchored;
    bool collide_with_walls;
    float friction;
    float anchor_damping_remaining;
    float anchor_damping_total;
    bool sim_ignore;
};

Node* node_create(int idx, float mass, float x, float y);
void node_free(Node *n);
void node_draw(Node *n, float radius);

#endif // NODE_H
