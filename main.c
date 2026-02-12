// Clean, corrected main.c
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Menu.h"
#include "datastructures.h"
#include "Simulator.h"
#include "Constraint.h"

// Tool selection enum for edit menu
typedef enum {
    TOOL_NONE = -1,
    TOOL_SELECT = 0,
    TOOL_ADD_NODE,
    TOOL_ADD_DIST,
    TOOL_ADD_SPRING,
    TOOL_ADD_WALL
} ToolType;

// Data used by edit menu callbacks
typedef struct {
    DynArray *menus;
    Menu **edit_menu;   // pointer to Menu* stored in main
    Menu **select_menu; // pointer to optional bottom menu
    int *win_w;         // pointers to current window size
    int *win_h;
    int *tool_select;
    int *tool_node;
    int *tool_dist;
    int *tool_spring;
    int *tool_wall;
    int *current_tool;
    int *sel_filter;    // pointer to current selection filter in main
    // +Node tool spawn parameters (allocated when Edit menu row is created)
    double *node_mass;
    double *node_friction;
    int *node_anchored;
    int *node_collide_with_walls;
    // selection & simulator hooks (set by caller)
    Simulator *sim;
    DynArray *selection;
    int *drag_enabled; // pointer to drag enable flag (allocated by menu)
    double *drag_strength; // pointer to configurable drag force gain
    double *drag_damping;  // pointer to configurable drag damping (B)
    // SPRING/WALL tool preferences (persistent)
    double *spring_stiffness;
    int *wall_add_dist;
    double *wall_friction;
    double *wall_restitution;
} EditData;

// helper: remove and free a menu from the menus list if present
static void remove_menu_from_list(DynArray *menus, Menu *m) {
    if (!menus || !m) return;
    size_t idx = (size_t)-1;
    for (size_t i = 0; i < menus->size; ++i) {
        if (menus->items[i] == m) { idx = i; break; }
    }
    if (idx == (size_t)-1) return;
    menu_free(m);
    // shift left
    for (size_t j = idx; j + 1 < menus->size; ++j) menus->items[j] = menus->items[j+1];
    menus->size -= 1;
}

// forward declarations for callbacks
static void on_edit_toggle(VariableInteraction *vi, void *user_data);
static void on_tool_change(VariableInteraction *vi, void *user_data);
// selection callbacks (defined later)
static void sel_cb_toggle_anchor(VariableInteraction *vi, void *user_data);
static void sel_cb_scale_mass(VariableInteraction *vi, void *user_data);
static void sel_cb_move_or_rotate(VariableInteraction *vi, void *user_data);
static void sel_cb_drag_toggle(VariableInteraction *vi, void *user_data);
// additional selection callbacks (defined later)
static void sel_cb_set_node_friction(VariableInteraction *vi, void *user_data);
static void sel_cb_delete_nodes(VariableInteraction *vi, void *user_data);
static void sel_cb_set_constraint_distance(VariableInteraction *vi, void *user_data);
static void sel_cb_set_spring_prop(VariableInteraction *vi, void *user_data);
static void sel_cb_set_wall_prop(VariableInteraction *vi, void *user_data);

// Create the edit palette menu (if not already created). Caller must ensure ed != NULL
static void create_edit_menu_if_needed(EditData *ed) {
    if (!ed || !ed->menus) return;
    if (*(ed->edit_menu) != NULL) return;
    // position to the right of Controls (wider so there is room for more controls)
    Menu *m = menu_create(240, 10, 440, 160, 110, "Edit", (Color){255,255,255,255}, (Color){60,40,60,255});

    // Row 1: tool radio buttons
    MenuRow *row_tools = menurow_create();
    VariableInteraction *v_sel = variableinteraction_create(ed->tool_select, "Select", 0, 1, VAR_BOOL, on_tool_change, ed);
    VariableInteraction *v_node = variableinteraction_create(ed->tool_node, "+Node", 0, 1, VAR_BOOL, on_tool_change, ed);
    VariableInteraction *v_dist = variableinteraction_create(ed->tool_dist, "+Dist", 0, 1, VAR_BOOL, on_tool_change, ed);
    VariableInteraction *v_spring = variableinteraction_create(ed->tool_spring, "+Spring", 0, 1, VAR_BOOL, on_tool_change, ed);
    VariableInteraction *v_wall = variableinteraction_create(ed->tool_wall, "+Wall", 0, 1, VAR_BOOL, on_tool_change, ed);
    menurow_add_interaction(row_tools, v_sel);
    menurow_add_interaction(row_tools, v_node);
    menurow_add_interaction(row_tools, v_dist);
    menurow_add_interaction(row_tools, v_spring);
    menurow_add_interaction(row_tools, v_wall);
    menu_add_row(m, row_tools);

    // Row 2: placeholder - populated by update_edit_row2_for_tool
    MenuRow *row_placeholder = menurow_create();
    double dummy = 0.0;
    VariableInteraction *v_dummy = variableinteraction_create(&dummy, "(no tool selected)", 0.0, 1.0, VAR_SLIDER, NULL, NULL);
    menurow_add_interaction(row_placeholder, v_dummy);
    menu_add_row(m, row_placeholder);

    dynarray_append(ed->menus, m);
    *(ed->edit_menu) = m;
}

static void destroy_edit_menu_if_present(EditData *ed) {
    if (!ed || !ed->menus) return;
    if (*(ed->edit_menu) == NULL) return;
    remove_menu_from_list(ed->menus, *(ed->edit_menu));
    *(ed->edit_menu) = NULL;
}

// Selection kinds (file-scope so helpers can reference)
enum SelKind { SEL_NONE = 0, SEL_NODE = 1, SEL_CONSTRAINT = 2, SEL_WALL = 3 };

// Helper used for click candidate lists
typedef struct {
    int type; // SEL_NODE, SEL_CONSTRAINT, SEL_WALL
    void *obj;
} ClickCandidate;

// Callback data for mass-scale slider: remember last slider value so we apply
// relative scaling (factor / last) on each change instead of repeatedly
// multiplying from absolute slider value. This avoids immediate reset/snapping
// and exponential application while dragging.
typedef struct {
    EditData *ed;
    double last;
} MassScaleCB;

// forward-declare helpers used below
static float point_segment_distance2(float px, float py, float ax, float ay, float bx, float by);
static void sel_cb_delete_selection(VariableInteraction *vi, void *user_data);

static void create_select_menu_if_needed(EditData *ed) {
    if (!ed || !ed->menus) return;
    if (*(ed->select_menu) != NULL) return;
    int w = 300, h = 160;
    int x = 10;
    int y = (*(ed->win_h) > h + 20) ? (*(ed->win_h) - h - 10) : 400;
    Menu *m = menu_create(x, y, w, h, 120, "Selection", (Color){255,255,255,255}, (Color){40,40,70,255});
    int filter = ed->sel_filter ? *(ed->sel_filter) : SEL_NODE;
    if (filter == SEL_NODE) {
        // Row: Select Size
        MenuRow *r1 = menurow_create();
        double *sel_size = malloc(sizeof(double)); *sel_size = 10.0;
        VariableInteraction *v_sel_size = variableinteraction_create(sel_size, "Select Size", 1.0, 200.0, VAR_SLIDER, NULL, NULL);
        menurow_add_interaction(r1, v_sel_size);
        menu_add_row(m, r1);

        // Row: Anchor toggle
        MenuRow *r_anchor = menurow_create();
        int *sel_anchor = malloc(sizeof(int)); *sel_anchor = 0;
        if (ed && ed->selection && dynarray_size(ed->selection) == 1) {
            Node *only = (Node*)dynarray_get(ed->selection, 0);
            if (only) *sel_anchor = only->anchored ? 1 : 0;
        }
        VariableInteraction *v_anchor = variableinteraction_create(sel_anchor, "Toggle Anchor", 0, 1, VAR_BOOL, sel_cb_toggle_anchor, ed);
        menurow_add_interaction(r_anchor, v_anchor);
        menu_add_row(m, r_anchor);

        // Row: Friction
        MenuRow *r_fric = menurow_create();
        double *fric = malloc(sizeof(double)); *fric = 0.5;
        if (ed && ed->selection && dynarray_size(ed->selection) == 1) {
            Node *only = (Node*)dynarray_get(ed->selection, 0);
            if (only) *fric = only->friction;
        }
        VariableInteraction *v_fric = variableinteraction_create(fric, "Friction", 0.0, 1.0, VAR_SLIDER, sel_cb_set_node_friction, ed);
        menurow_add_interaction(r_fric, v_fric);
        menu_add_row(m, r_fric);

        // Row: Mass scale (relative slider)
        MenuRow *r_mass = menurow_create();
        double *mass_scale = malloc(sizeof(double)); *mass_scale = 1.0;
        MassScaleCB *mscb = malloc(sizeof(MassScaleCB)); mscb->ed = ed; mscb->last = 1.0;
        VariableInteraction *v_mass = variableinteraction_create(mass_scale, "Scale Mass", 0.1, 10.0, VAR_SLIDER, sel_cb_scale_mass, mscb);
        menurow_add_interaction(r_mass, v_mass);
        menu_add_row(m, r_mass);

        // Row: Drag toggle
        MenuRow *r_drag = menurow_create();
        int *drag_enable = NULL;
        // Reuse existing toggle state if present in EditData so recreating the
        // selection menu doesn't reset the user's choice.
        if (ed && ed->drag_enabled) {
            drag_enable = ed->drag_enabled;
        } else {
            drag_enable = malloc(sizeof(int)); *drag_enable = 0;
        }
        VariableInteraction *v_drag = variableinteraction_create(drag_enable, "Drag", 0, 1, VAR_BOOL, sel_cb_drag_toggle, ed);
        menurow_add_interaction(r_drag, v_drag);
        menu_add_row(m, r_drag);

        // Row: Delete node
        MenuRow *r_deln = menurow_create();
        int *deln = malloc(sizeof(int)); *deln = 0;
        VariableInteraction *v_deln = variableinteraction_create(deln, "Delete Node", 0, 1, VAR_BOOL, sel_cb_delete_nodes, ed);
        menurow_add_interaction(r_deln, v_deln);
        menu_add_row(m, r_deln);

    // store pointers in EditData so callbacks and main loop can access
    if (ed) { ed->selection = ed->selection ? ed->selection : NULL; ed->drag_enabled = drag_enable; }
    } else if (filter == SEL_CONSTRAINT) {
        // For constraints show delete control and property sliders for single selection
        MenuRow *r_del = menurow_create();
        int *del = malloc(sizeof(int)); *del = 0;
        VariableInteraction *v_del = variableinteraction_create(del, "Delete", 0, 1, VAR_BOOL, sel_cb_delete_selection, ed);
        menurow_add_interaction(r_del, v_del);
        menu_add_row(m, r_del);

        if (ed && ed->selection && dynarray_size(ed->selection) == 1) {
            Constraint *c = (Constraint*)dynarray_get(ed->selection, 0);
            if (c && c->type == CT_DIST) {
                MenuRow *r_dist = menurow_create();
                double *distv = malloc(sizeof(double)); *distv = c->rest_length;
                VariableInteraction *v_distv = variableinteraction_create(distv, "Distance", 0.0, 200.0, VAR_SLIDER, sel_cb_set_constraint_distance, ed);
                menurow_add_interaction(r_dist, v_distv);
                menu_add_row(m, r_dist);
            } else if (c && c->type == CT_SPRING) {
                MenuRow *r_st = menurow_create();
                double *stiff = malloc(sizeof(double)); *stiff = c->stiffness;
                VariableInteraction *v_st = variableinteraction_create(stiff, "Stiffness", 0.0, 800.0, VAR_SLIDER, sel_cb_set_spring_prop, ed);
                menurow_add_interaction(r_st, v_st);
                menu_add_row(m, r_st);
                MenuRow *r_rest = menurow_create();
                double *restv = malloc(sizeof(double)); *restv = c->rest_length;
                VariableInteraction *v_restv = variableinteraction_create(restv, "Rest Length", 0.0, 200.0, VAR_SLIDER, sel_cb_set_spring_prop, ed);
                menurow_add_interaction(r_rest, v_restv);
                menu_add_row(m, r_rest);
            }
        }
    } else if (filter == SEL_WALL) {
        MenuRow *r_delw = menurow_create();
        int *delw = malloc(sizeof(int)); *delw = 0;
        VariableInteraction *v_delw = variableinteraction_create(delw, "Delete Wall", 0, 1, VAR_BOOL, sel_cb_delete_selection, ed);
        menurow_add_interaction(r_delw, v_delw);
        menu_add_row(m, r_delw);

        if (ed && ed->selection && dynarray_size(ed->selection) == 1) {
            TriangleWall *w = (TriangleWall*)dynarray_get(ed->selection, 0);
            if (w) {
                MenuRow *r_wf = menurow_create();
                double *wf = malloc(sizeof(double)); *wf = w->friction;
                VariableInteraction *v_wf = variableinteraction_create(wf, "Friction", 0.0, 1.0, VAR_SLIDER, sel_cb_set_wall_prop, ed);
                menurow_add_interaction(r_wf, v_wf);
                menu_add_row(m, r_wf);
                MenuRow *r_wr = menurow_create();
                double *wr = malloc(sizeof(double)); *wr = w->restitution;
                VariableInteraction *v_wr = variableinteraction_create(wr, "Restitution", 0.0, 1.0, VAR_SLIDER, sel_cb_set_wall_prop, ed);
                menurow_add_interaction(r_wr, v_wr);
                menu_add_row(m, r_wr);
            }
        }
    }

    // append the selection menu to registry and expose it to caller
    dynarray_append(ed->menus, m);
    *(ed->select_menu) = m;
}

static void destroy_select_menu_if_present(EditData *ed) {
    if (!ed || !ed->menus) return;
    if (*(ed->select_menu) == NULL) return;
    remove_menu_from_list(ed->menus, *(ed->select_menu));
    *(ed->select_menu) = NULL;
}

static void update_edit_row2_for_tool(EditData *ed) {
    if (!ed || !ed->edit_menu || *(ed->edit_menu) == NULL) return;
    Menu *m = *(ed->edit_menu);
    // remove existing row 2 if present
    if (m->rows->size > 1) {
        MenuRow *r = (MenuRow*)m->rows->items[1];
        // free interactions in the row
        for (size_t i = 0; i < r->interactions->size; ++i) {
            VariableInteraction *vi = (VariableInteraction*)r->interactions->items[i];
            free(vi->name);
            free(vi);
        }
        dynarray_free(r->interactions, NULL);
        free(r);
        // shift rows left
        for (size_t j = 1; j + 1 < m->rows->size; ++j) m->rows->items[j] = m->rows->items[j+1];
        m->rows->size -= 1;
    }
    // create new row based on current tool
    MenuRow *nr = menurow_create();
    if (*(ed->current_tool) == TOOL_ADD_NODE) {
        // ensure previous node param allocations freed
        if (ed->node_mass) { free(ed->node_mass); ed->node_mass = NULL; }
        if (ed->node_friction) { free(ed->node_friction); ed->node_friction = NULL; }
        if (ed->node_anchored) { free(ed->node_anchored); ed->node_anchored = NULL; }
        if (ed->node_collide_with_walls) { free(ed->node_collide_with_walls); ed->node_collide_with_walls = NULL; }
        // allocate spawn params
        ed->node_mass = malloc(sizeof(double)); *ed->node_mass = 1.0;
        ed->node_friction = malloc(sizeof(double)); *ed->node_friction = 0.0;
        ed->node_anchored = malloc(sizeof(int)); *ed->node_anchored = 0;
        ed->node_collide_with_walls = malloc(sizeof(int)); *ed->node_collide_with_walls = 1;

        VariableInteraction *vi_mass = variableinteraction_create(ed->node_mass, "Mass", 0.01, 200.0, VAR_SLIDER, NULL, NULL);
        VariableInteraction *vi_friction = variableinteraction_create(ed->node_friction, "Friction", 0.0, 1.0, VAR_SLIDER, NULL, NULL);
        VariableInteraction *vi_anchor = variableinteraction_create(ed->node_anchored, "Anchored", 0, 1, VAR_BOOL, NULL, NULL);
        VariableInteraction *vi_collide = variableinteraction_create(ed->node_collide_with_walls, "Collide Walls", 0, 1, VAR_BOOL, NULL, NULL);
        menurow_add_interaction(nr, vi_mass);
        menurow_add_interaction(nr, vi_friction);
        menurow_add_interaction(nr, vi_anchor);
        menurow_add_interaction(nr, vi_collide);
        menu_add_row(m, nr);
    } else {
        if (*(ed->current_tool) == TOOL_SELECT) {
            double dval = 0.0; VariableInteraction *vi = variableinteraction_create(&dval, "Select Mode", 0.0, 1.0, VAR_SLIDER, NULL, NULL);
            menurow_add_interaction(nr, vi);
            menu_add_row(m, nr);
        } else if (*(ed->current_tool) == TOOL_ADD_DIST) {
            double dval = 0.0; VariableInteraction *vi = variableinteraction_create(&dval, "Dist Options", 0.0, 1.0, VAR_SLIDER, NULL, NULL);
            menurow_add_interaction(nr, vi);
            menu_add_row(m, nr);
        } else if (*(ed->current_tool) == TOOL_ADD_SPRING) {
            // Spring options: preset stiffness
            if (!ed->spring_stiffness) { ed->spring_stiffness = malloc(sizeof(double)); *ed->spring_stiffness = 10.0; }
            VariableInteraction *vi_st = variableinteraction_create(ed->spring_stiffness, "Stiffness", 0.0, 800.0, VAR_SLIDER, NULL, NULL);
            menurow_add_interaction(nr, vi_st);
            menu_add_row(m, nr);
        } else if (*(ed->current_tool) == TOOL_ADD_WALL) {
            // Wall options: toggle to also add a distance constraint, plus friction/restitution presets
            if (!ed->wall_add_dist) { ed->wall_add_dist = malloc(sizeof(int)); *ed->wall_add_dist = 0; }
            if (!ed->wall_friction) { ed->wall_friction = malloc(sizeof(double)); *ed->wall_friction = 1.0; }
            if (!ed->wall_restitution) { ed->wall_restitution = malloc(sizeof(double)); *ed->wall_restitution = 0.0; }
            VariableInteraction *vi_dist_toggle = variableinteraction_create(ed->wall_add_dist, "+Dist", 0, 1, VAR_BOOL, NULL, NULL);
            menurow_add_interaction(nr, vi_dist_toggle);
            menu_add_row(m, nr);
            // friction/rest sliders on a new row
            MenuRow *nr2 = menurow_create();
            VariableInteraction *vi_wf = variableinteraction_create(ed->wall_friction, "Friction", 0.0, 1.0, VAR_SLIDER, NULL, NULL);
            VariableInteraction *vi_wr = variableinteraction_create(ed->wall_restitution, "Restitution", 0.0, 1.0, VAR_SLIDER, NULL, NULL);
            menurow_add_interaction(nr2, vi_wf);
            menurow_add_interaction(nr2, vi_wr);
            menu_add_row(m, nr2);
        } else {
            double dval = 0.0;
            VariableInteraction *vi = variableinteraction_create(&dval, "(no tool)", 0.0, 1.0, VAR_SLIDER, NULL, NULL);
            menurow_add_interaction(nr, vi);
            menu_add_row(m, nr);
        }
    }
}

static void on_edit_toggle(VariableInteraction *vi, void *user_data) {
    (void)user_data; // vi->callback_data carries EditData*
    EditData *ed = (EditData*)vi->callback_data;
    if (!ed) return;
    int on = *(int*)vi->variable;
    if (on) {
        create_edit_menu_if_needed(ed);
    } else {
        destroy_edit_menu_if_present(ed);
        destroy_select_menu_if_present(ed);
        if (ed->current_tool) *(ed->current_tool) = TOOL_NONE;
    }
}

static void on_tool_change(VariableInteraction *vi, void *user_data) {
    EditData *ed = (EditData*)vi->callback_data;
    if (!ed || !ed->edit_menu) return;
    // if the toggled button was enabled, set others off
    int *var = (int*)vi->variable;
    if (!var) return;
    if (*var) {
        // set all tool vars to 0
        *(ed->tool_select) = 0; *(ed->tool_node) = 0; *(ed->tool_dist) = 0; *(ed->tool_spring) = 0; *(ed->tool_wall) = 0;
        // enable this one
        *var = 1;
        // determine which tool
        if (var == ed->tool_select) *(ed->current_tool) = TOOL_SELECT;
        else if (var == ed->tool_node) *(ed->current_tool) = TOOL_ADD_NODE;
        else if (var == ed->tool_dist) *(ed->current_tool) = TOOL_ADD_DIST;
        else if (var == ed->tool_spring) *(ed->current_tool) = TOOL_ADD_SPRING;
        else if (var == ed->tool_wall) *(ed->current_tool) = TOOL_ADD_WALL;
    } else {
        // disabled the current tool
        *(ed->current_tool) = TOOL_NONE;
        // Clear drag when switching away from select tool
        if (ed->drag_enabled) {
            *(ed->drag_enabled) = 0;
        }
    }
    // Update row 2 contents
    update_edit_row2_for_tool(ed);
    // If select tool, create bottom selection menu; otherwise destroy it
    if (*(ed->current_tool) == TOOL_SELECT) create_select_menu_if_needed(ed);
    else destroy_select_menu_if_present(ed);
}

#define MAX_INPUT_HANDLERS 64

typedef enum {
    HANDLER_MOUSE_PRESS,
    HANDLER_MOUSE_RELEASE,
    HANDLER_MOUSE_MOVE,
    HANDLER_KEY_PRESS
} HandlerType;

typedef struct {
    int x, y, w, h;
} BoundingBox;

typedef struct InputHandler {
    HandlerType type;
    BoundingBox box; // For mouse events
    SDL_Keycode key; // For key events
    int (*should_run)(struct InputHandler*, const SDL_Event*);
    void (*run)(struct InputHandler*, const SDL_Event*);
    void *user_data;
} InputHandler;

static InputHandler *input_handlers[MAX_INPUT_HANDLERS];
static int input_handler_count = 0;

void register_input_handler(InputHandler *handler) {
    if (input_handler_count < MAX_INPUT_HANDLERS) {
        input_handlers[input_handler_count++] = handler;
    }
}

static int mouse_should_run(InputHandler *handler, const SDL_Event *event) {
    if (!event) return 0;
    if (event->type == SDL_MOUSEBUTTONDOWN || event->type == SDL_MOUSEBUTTONUP) {
        int mx = event->button.x;
        int my = event->button.y;
        BoundingBox *box = &handler->box;
        return mx >= box->x && mx < box->x + box->w && my >= box->y && my < box->y + box->h;
    } else if (event->type == SDL_MOUSEMOTION) {
        int mx = event->motion.x;
        int my = event->motion.y;
        BoundingBox *box = &handler->box;
        return mx >= box->x && mx < box->x + box->w && my >= box->y && my < box->y + box->h;
    }
    return 0;
}

static int key_should_run(InputHandler *handler, const SDL_Event *event) {
    return event && event->type == SDL_KEYDOWN && event->key.keysym.sym == handler->key;
}

static void mouse_press_run(InputHandler *handler, const SDL_Event *event) {
    (void)event;
    printf("Mouse pressed in box (%d,%d,%d,%d)\n", handler->box.x, handler->box.y, handler->box.w, handler->box.h);
}

static void key_press_run(InputHandler *handler, const SDL_Event *event) {
    (void)event;
    printf("Key %d pressed\n", (int)handler->key);
}

// helper: squared distance from point P to segment AB
static float point_segment_distance2(float px, float py, float ax, float ay, float bx, float by) {
    float vx = bx - ax, vy = by - ay;
    float wx = px - ax, wy = py - ay;
    float c1 = vx * wx + vy * wy;
    if (c1 <= 0.0f) return wx*wx + wy*wy;
    float c2 = vx*vx + vy*vy;
    if (c2 <= c1) {
        float dx = px - bx, dy = py - by; return dx*dx + dy*dy;
    }
    float t = c1 / c2;
    float projx = ax + t * vx, projy = ay + t * vy;
    float dx = px - projx, dy = py - projy;
    return dx*dx + dy*dy;
}

// Callback data struct
typedef struct { Menu *menu; int field; } MenuCallbackData;
enum {F_X, F_Y, F_W, F_H, F_COLOR};

    // Callbacks
void on_slider_change(VariableInteraction *vi, void *user_data) {
    MenuCallbackData *md = (MenuCallbackData*)user_data;
    double val = *(double*)vi->variable;
    if (!md || !md->menu) return;
    switch (md->field) {
        case F_X: md->menu->x = (int)val; break;
        case F_Y: md->menu->y = (int)val; break;
        case F_W: md->menu->width = (int)val; break;
        case F_H: md->menu->height = (int)val; break;
    }
}

void on_bool_change(VariableInteraction *vi, void *user_data) {
    MenuCallbackData *md = (MenuCallbackData*)user_data;
    int v = *(int*)vi->variable;
    if (!md || !md->menu) return;
    if (v) {
        md->menu->bgColor.r = 20;
        md->menu->bgColor.g = 160;
        md->menu->bgColor.b = 20;
    } else {
        md->menu->bgColor.r = 50;
        md->menu->bgColor.g = 50;
        md->menu->bgColor.b = 80;
    }
}

// Forward declaration for callback used when creating menus in main
static void sim_on_step_change(VariableInteraction *vi, void *user_data);

// Selection callbacks
static void sel_cb_toggle_anchor(VariableInteraction *vi, void *user_data) {
    EditData *ed = (EditData*)user_data;
    if (!ed || !ed->selection || !ed->sim) return;
    int v = *(int*)vi->variable;
    // if setting anchor on selected nodes, create AnchorConstraint for each unanchored node
    if (v) {
        for (size_t i = 0; i < dynarray_size(ed->selection); ++i) {
            Node *n = (Node*)dynarray_get(ed->selection, i);
            if (!n) continue;
            if (!n->anchored) {
                Constraint *ac = anchorconstraint_create(n, n->pos[0], n->pos[1], n->pos[2]);
                if (ac && ed->sim) simulator_add_constraint(ed->sim, ac);
                // anchorconstraint_create already sets n->anchored = true
            }
        }
    } else {
        // removing anchor: find and remove any Anchor constraints in simulator associated with these nodes
        if (!ed->sim) return;
        for (size_t si = 0; si < dynarray_size(ed->selection); ++si) {
            Node *n = (Node*)dynarray_get(ed->selection, si);
            if (!n) continue;
            // clear the flag immediately
            n->anchored = false;
            // iterate simulator constraints backwards and remove anchor constraints for this node
            for (ssize_t ci = (ssize_t)dynarray_size(ed->sim->constraints) - 1; ci >= 0; --ci) {
                Constraint *c = (Constraint*)dynarray_get(ed->sim->constraints, (size_t)ci);
                if (!c) continue;
                if (c->type == CT_ANCHOR && c->node == n) {
                    // free constraint and remove from array
                    free(c);
                    // shift left
                    for (size_t j = (size_t)ci; j + 1 < ed->sim->constraints->size; ++j) ed->sim->constraints->items[j] = ed->sim->constraints->items[j+1];
                    ed->sim->constraints->size -= 1;
                }
            }
        }
    }
}

static void sel_cb_scale_mass(VariableInteraction *vi, void *user_data) {
    MassScaleCB *ms = (MassScaleCB*)user_data;
    if (!ms || !ms->ed || !ms->ed->selection) return;
    EditData *ed = ms->ed;
    double cur = *(double*)vi->variable;
    if (cur <= 0.0) return;
    // compute relative change since last value
    double rel = 1.0;
    if (ms->last > 0.0) rel = cur / ms->last;
    if (rel == 1.0) return; // no-op
    for (size_t i = 0; i < dynarray_size(ed->selection); ++i) {
        Node *n = (Node*)dynarray_get(ed->selection, i);
        if (!n) continue;
        n->mass *= (float)rel;
    }
    // remember last applied value
    ms->last = cur;
}

static void sel_cb_move_or_rotate(VariableInteraction *vi, void *user_data) {
    EditData *ed = (EditData*)user_data;
    if (!ed || !ed->selection) return;
    const char *name = vi->name;
    if (!name) return;
    if (strcmp(name, "Move X") == 0) {
        double dx = *(double*)vi->variable;
        for (size_t i = 0; i < dynarray_size(ed->selection); ++i) {
            Node *n = (Node*)dynarray_get(ed->selection, i);
            if (!n) continue;
            n->pos[0] += (float)dx;
        }
        *(double*)vi->variable = 0.0;
    } else if (strcmp(name, "Move Y") == 0) {
        double dy = *(double*)vi->variable;
        for (size_t i = 0; i < dynarray_size(ed->selection); ++i) {
            Node *n = (Node*)dynarray_get(ed->selection, i);
            if (!n) continue;
            n->pos[1] += (float)dy;
        }
        *(double*)vi->variable = 0.0;
    } else if (strcmp(name, "Rotate (deg)") == 0) {
        double deg = *(double*)vi->variable;
        if (deg == 0.0) return;
        double rad = deg * (3.14159265358979323846 / 180.0);
        // compute COM
        double cx = 0.0, cy = 0.0;
        size_t cnt = dynarray_size(ed->selection);
        for (size_t i = 0; i < cnt; ++i) {
            Node *n = (Node*)dynarray_get(ed->selection, i);
            if (!n) continue;
            cx += n->pos[0]; cy += n->pos[1];
        }
        if (cnt == 0) return;
        cx /= (double)cnt; cy /= (double)cnt;
        // rotate each about COM
        for (size_t i = 0; i < cnt; ++i) {
            Node *n = (Node*)dynarray_get(ed->selection, i);
            if (!n) continue;
            double rx = n->pos[0] - cx;
            double ry = n->pos[1] - cy;
            double sx = rx * cos(rad) - ry * sin(rad);
            double sy = rx * sin(rad) + ry * cos(rad);
            n->pos[0] = (float)(cx + sx);
            n->pos[1] = (float)(cy + sy);
        }
        *(double*)vi->variable = 0.0;
    }
}

static void sel_cb_drag_toggle(VariableInteraction *vi, void *user_data) {
    EditData *ed = (EditData*)user_data;
    if (!ed) return;
    // drag_enabled pointer already set in ed by create_select_menu_if_needed
}

// Delete selected constraints or walls depending on current sel_filter
static void sel_cb_delete_selection(VariableInteraction *vi, void *user_data) {
    (void)vi;
    EditData *ed = (EditData*)user_data;
    if (!ed || !ed->selection || !ed->sim) return;
    int f = ed->sel_filter ? *(ed->sel_filter) : SEL_NONE;
    if (f == SEL_CONSTRAINT) {
        // iterate selected constraints and remove them from simulator
        for (size_t si = 0; si < dynarray_size(ed->selection); ++si) {
            Constraint *c = (Constraint*)dynarray_get(ed->selection, si);
            if (!c) continue;
            // remember linked nodes and type before freeing
            DynArray *cons = ed->sim->constraints;
            Node *n0 = c->node;
            Node *n1 = c->other;
            int ctype = c->type;
            // remove from simulator->constraints (search backwards for stability)
            for (ssize_t ci = (ssize_t)dynarray_size(cons) - 1; ci >= 0; --ci) {
                if ((Constraint*)dynarray_get(cons, (size_t)ci) == c) {
                    for (size_t j = (size_t)ci; j + 1 < cons->size; ++j) cons->items[j] = cons->items[j+1];
                    cons->size -= 1;
                    break;
                }
            }
            // remove from any node->constraints lists (use saved node pointers)
            if (n0 && n0->constraints) {
                DynArray *nc = n0->constraints;
                for (size_t j = 0; j < nc->size; ++j) {
                    if ((Constraint*)dynarray_get(nc, j) == c) {
                        for (size_t k = j; k + 1 < nc->size; ++k) nc->items[k] = nc->items[k+1];
                        nc->size -= 1; break;
                    }
                }
            }
            if (n1 && n1->constraints) {
                DynArray *nc = n1->constraints;
                for (size_t j = 0; j < nc->size; ++j) {
                    if ((Constraint*)dynarray_get(nc, j) == c) {
                        for (size_t k = j; k + 1 < nc->size; ++k) nc->items[k] = nc->items[k+1];
                        nc->size -= 1; break;
                    }
                }
            }
            // if this was an anchor constraint, clear anchored flag on its node
            if (ctype == CT_ANCHOR && n0) n0->anchored = false;
            // finally free constraint memory
            free(c);
        }
        // clear selection
        ed->selection->size = 0;
        // refresh menu
        destroy_select_menu_if_present(ed);
        create_select_menu_if_needed(ed);
    } else if (f == SEL_WALL) {
        for (size_t si = 0; si < dynarray_size(ed->selection); ++si) {
            TriangleWall *w = (TriangleWall*)dynarray_get(ed->selection, si);
            if (!w) continue;
            DynArray *ws = ed->sim->walls;
            for (ssize_t wi = (ssize_t)dynarray_size(ws) - 1; wi >= 0; --wi) {
                if ((TriangleWall*)dynarray_get(ws, (size_t)wi) == w) {
                    trianglewall_free(w);
                    for (size_t j = (size_t)wi; j + 1 < ws->size; ++j) ws->items[j] = ws->items[j+1];
                    ws->size -= 1; break;
                }
            }
        }
        ed->selection->size = 0;
        destroy_select_menu_if_present(ed);
        create_select_menu_if_needed(ed);
    }
}

// Helper function to get plane normal and offset based on plane type and camera orientation
static void get_plane_params(int plane_type, float offset, float cam_yaw, float cam_pitch, 
                             float cam_x, float cam_y, float cam_z,
                             float *normal_x, float *normal_y, float *normal_z, float *d) {
    // Plane equation: normal · (P - point_on_plane) = 0, or normal · P = d
    switch (plane_type) {
        case 0: // PLANE_WORLD_X: YZ plane at X = offset
            *normal_x = 1.0f; *normal_y = 0.0f; *normal_z = 0.0f;
            *d = offset;
            break;
        case 1: // PLANE_WORLD_Y: XZ plane at Y = offset
            *normal_x = 0.0f; *normal_y = 1.0f; *normal_z = 0.0f;
            *d = offset;
            break;
        case 2: // PLANE_WORLD_Z: XY plane at Z = offset
            *normal_x = 0.0f; *normal_y = 0.0f; *normal_z = 1.0f;
            *d = offset;
            break;
        case 3: // PLANE_CAM_X: perpendicular to camera right vector
            {
                float right_x = cosf(cam_yaw);
                float right_z = -sinf(cam_yaw);
                *normal_x = right_x; *normal_y = 0.0f; *normal_z = right_z;
                *d = cam_x * right_x + cam_z * right_z + offset;
            }
            break;
        case 4: // PLANE_CAM_Y: perpendicular to camera up vector (accounting for pitch)
            {
                // Camera up is perpendicular to forward in vertical plane
                float up_x = sinf(cam_yaw) * sinf(cam_pitch);
                float up_y = cosf(cam_pitch);
                float up_z = cosf(cam_yaw) * sinf(cam_pitch);
                *normal_x = up_x; *normal_y = up_y; *normal_z = up_z;
                *d = cam_x * up_x + cam_y * up_y + cam_z * up_z + offset;
            }
            break;
        case 5: // PLANE_CAM_Z: perpendicular to camera forward vector
            {
                float forward_x = -sinf(cam_yaw) * cosf(cam_pitch);
                float forward_y = -sinf(cam_pitch);
                float forward_z = -cosf(cam_yaw) * cosf(cam_pitch);
                *normal_x = forward_x; *normal_y = forward_y; *normal_z = forward_z;
                *d = cam_x * forward_x + cam_y * forward_y + cam_z * forward_z + offset;
            }
            break;
        default:
            *normal_x = 0.0f; *normal_y = 1.0f; *normal_z = 0.0f; *d = 0.0f;
            break;
    }
}

// Ray-plane intersection: returns 1 if hit, sets hit_x/y/z
static int ray_plane_intersect(float ray_ox, float ray_oy, float ray_oz,
                                float ray_dx, float ray_dy, float ray_dz,
                                float plane_nx, float plane_ny, float plane_nz, float plane_d,
                                float *hit_x, float *hit_y, float *hit_z) {
    float denom = ray_dx * plane_nx + ray_dy * plane_ny + ray_dz * plane_nz;
    if (fabsf(denom) < 1e-6f) return 0; // ray parallel to plane
    
    float t = (plane_d - (ray_ox * plane_nx + ray_oy * plane_ny + ray_oz * plane_nz)) / denom;
    // Allow intersections in both directions (remove t < 0 check) so objects behind the plane are also selectable
    
    *hit_x = ray_ox + t * ray_dx;
    *hit_y = ray_oy + t * ray_dy;
    *hit_z = ray_oz + t * ray_dz;
    return 1;
}

// Render translucent grid on the selection plane
static void render_plane_grid(int plane_type, float offset, float cam_yaw, float cam_pitch,
                              float cam_x, float cam_y, float cam_z, float grid_size, float cell_size) {
    float nx, ny, nz, d;
    get_plane_params(plane_type, offset, cam_yaw, cam_pitch, cam_x, cam_y, cam_z, &nx, &ny, &nz, &d);
    
    // Generate two basis vectors perpendicular to the normal
    float basis1_x, basis1_y, basis1_z, basis2_x, basis2_y, basis2_z;
    
    // Choose an arbitrary vector not parallel to normal
    float temp_x = 1.0f, temp_y = 0.0f, temp_z = 0.0f;
    if (fabsf(nx) > 0.9f) { temp_x = 0.0f; temp_y = 1.0f; temp_z = 0.0f; }
    
    // basis1 = normal × temp
    basis1_x = ny * temp_z - nz * temp_y;
    basis1_y = nz * temp_x - nx * temp_z;
    basis1_z = nx * temp_y - ny * temp_x;
    float len1 = sqrtf(basis1_x * basis1_x + basis1_y * basis1_y + basis1_z * basis1_z);
    basis1_x /= len1; basis1_y /= len1; basis1_z /= len1;
    
    // basis2 = normal × basis1
    basis2_x = ny * basis1_z - nz * basis1_y;
    basis2_y = nz * basis1_x - nx * basis1_z;
    basis2_z = nx * basis1_y - ny * basis1_x;
    
    // Find a point on the plane
    float point_x = nx * d, point_y = ny * d, point_z = nz * d;
    
    // Draw translucent grid
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.5f, 0.5f, 0.8f, 0.3f); // translucent blue-gray
    glBegin(GL_LINES);
    
    int num_lines = (int)(grid_size / cell_size);
    for (int i = -num_lines; i <= num_lines; ++i) {
        float t = i * cell_size;
        // Lines parallel to basis1
        float start_x = point_x + basis2_x * t - basis1_x * grid_size;
        float start_y = point_y + basis2_y * t - basis1_y * grid_size;
        float start_z = point_z + basis2_z * t - basis1_z * grid_size;
        float end_x = point_x + basis2_x * t + basis1_x * grid_size;
        float end_y = point_y + basis2_y * t + basis1_y * grid_size;
        float end_z = point_z + basis2_z * t + basis1_z * grid_size;
        glVertex3f(start_x, start_y, start_z);
        glVertex3f(end_x, end_y, end_z);
        
        // Lines parallel to basis2
        start_x = point_x + basis1_x * t - basis2_x * grid_size;
        start_y = point_y + basis1_y * t - basis2_y * grid_size;
        start_z = point_z + basis1_z * t - basis2_z * grid_size;
        end_x = point_x + basis1_x * t + basis2_x * grid_size;
        end_y = point_y + basis1_y * t + basis2_y * grid_size;
        end_z = point_z + basis1_z * t + basis2_z * grid_size;
        glVertex3f(start_x, start_y, start_z);
        glVertex3f(end_x, end_y, end_z);
    }
    glEnd();
    glDisable(GL_BLEND);
}

// Calculate distance from point to plane
static float point_plane_distance(float px, float py, float pz, 
                                   float nx, float ny, float nz, float d) {
    return fabsf(nx * px + ny * py + nz * pz - d);
}

// Convert screen coordinates to world position by ray-plane intersection
static int screen_to_world_plane(int screen_x, int screen_y, int win_w, int win_h,
                                  float cam_x, float cam_y, float cam_z,
                                  float cam_yaw, float cam_pitch,
                                  int plane_type, float plane_offset,
                                  float *world_x, float *world_y, float *world_z) {
    // Get plane parameters
    float plane_nx, plane_ny, plane_nz, plane_d;
    get_plane_params(plane_type, plane_offset, cam_yaw, cam_pitch, cam_x, cam_y, cam_z,
                    &plane_nx, &plane_ny, &plane_nz, &plane_d);
    
    // Convert screen to normalized device coordinates (-1 to 1)
    float ndc_x = (2.0f * screen_x) / win_w - 1.0f;
    float ndc_y = 1.0f - (2.0f * screen_y) / win_h; // flip Y
    
    // Unproject to get ray direction (simplified for perspective projection)
    // This assumes FOV=60 degrees
    float fov_y = 60.0f;
    float aspect = (float)win_w / (float)win_h;
    float tan_half_fov = tanf(fov_y * 3.14159265f / 360.0f);
    
    // Ray direction in camera space
    float ray_cam_x = ndc_x * aspect * tan_half_fov;
    float ray_cam_y = ndc_y * tan_half_fov;
    float ray_cam_z = -1.0f; // camera looks along -Z
    
    // Transform ray direction to world space using camera rotation
    // Apply yaw (Y-axis) then pitch (X-axis) rotation
    float cos_yaw = cosf(cam_yaw), sin_yaw = sinf(cam_yaw);
    float cos_pitch = cosf(cam_pitch), sin_pitch = sinf(cam_pitch);
    
    // Rotate by pitch around X-axis
    float ray_x_rot = ray_cam_x;
    float ray_y_rot = ray_cam_y * cos_pitch - ray_cam_z * sin_pitch;
    float ray_z_rot = ray_cam_y * sin_pitch + ray_cam_z * cos_pitch;
    
    // Rotate by yaw around Y-axis
    float ray_world_x = ray_x_rot * cos_yaw + ray_z_rot * sin_yaw;
    float ray_world_y = ray_y_rot;
    float ray_world_z = -ray_x_rot * sin_yaw + ray_z_rot * cos_yaw;
    
    // Normalize ray direction
    float ray_len = sqrtf(ray_world_x * ray_world_x + ray_world_y * ray_world_y + ray_world_z * ray_world_z);
    ray_world_x /= ray_len;
    ray_world_y /= ray_len;
    ray_world_z /= ray_len;
    
    // Intersect ray with plane
    return ray_plane_intersect(cam_x, cam_y, cam_z, ray_world_x, ray_world_y, ray_world_z,
                              plane_nx, plane_ny, plane_nz, plane_d,
                              world_x, world_y, world_z);
}


int main(int argc, char *argv[]) {
    // parse runtime options
    int override_solver_iters = 0;
    int run_mesh_test = 0;
    int mesh_k = 3;
    for (int ai = 1; ai < argc; ++ai) {
        const char *arg = argv[ai];
        // support format --N=123
        if (strncmp(arg, "--N=", 4) == 0) {
            int v = atoi(arg + 4);
            if (v > 0) override_solver_iters = v;
        } else if (strcmp(arg, "--test-octagon") == 0) {
            run_mesh_test = 1;
        } else if (strncmp(arg, "--mesh_k=", 9) == 0) {
            int v = atoi(arg + 9);
            if (v > 0) mesh_k = v;
        }
    }
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    // Request OpenGL 3.3 (compatibility profile for immediate-mode UI rendering)
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);

    SDL_Window *window = SDL_CreateWindow("SDL OpenGL Simulator",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        800, 600, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);

    if (!window) {
        fprintf(stderr, "Window could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_GLContext glContext = SDL_GL_CreateContext(window);
    if (!glContext) {
        fprintf(stderr, "OpenGL context could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_GL_SetSwapInterval(1); // Enable vsync

    // Create a test menu and interactions
    Color textC = {255,255,255,255};
    Color bgC = {50,50,80,255};
    // Menu *test = menu_create(50, 50, 300, 200, 0, "Test Menu", textC, bgC);

    // Try to set a font for menu labels
    if (menu_set_font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14) != 0) {
        // try fallback in case path differs
        menu_set_font("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 14);
    }

    int running = 1;
    SDL_Event event;
    int win_w = 800, win_h = 600;

    // FPS tracking
    Uint32 fps_last_time = SDL_GetTicks();
    int fps_frames = 0;
    int fps_value = 0;

    // 3D Camera: position, orientation (yaw=horizontal, pitch=vertical), and movement speed
    float cam_x = 0.0f, cam_y = 0.0f, cam_z = 500.0f;  // start 500 units back from origin
    float cam_yaw = 0.0f;   // horizontal rotation (radians) - looking forward
    float cam_pitch = 0.0f; // vertical rotation (radians) - looking straight ahead
    float cam_speed = 50.0f; // units per second for WASD movement
    float cam_zoom = 1.0f;   // multiplier for movement speed
    int space_down = 0;
    int rotating_camera = 0;
    int rotate_last_x = 0, rotate_last_y = 0;
    // WASD key states
    int key_w = 0, key_a = 0, key_s = 0, key_d = 0;
    int key_shift = 0, key_ctrl = 0;

    // --- Initial scenario: Box in Sleeve ---
    Simulator *sim = simulator_create(1.0f/20.0f);
    if (override_solver_iters > 0) sim->solver_iters = override_solver_iters;

     /* UI menus: registry of menus. We'll dispatch mouse events to each menu in
         order and stop when one handles the event. This allows stacking UI. */
     DynArray *menus = dynarray_create(4);

     /* Control menu: step size slider and pause toggle. The slider updates
         sim->dt via callback. */
     double step_size = sim->dt;
     int paused = 0;
    // Grid rendering controls (toggle + cell size slider)
    int show_grid = 1;
    double grid_cell_size = 10.0; /* default matches Simulator.c cell size; slider range below */

    Menu *ctrl = menu_create(10, 10, 220, 110, 100, "Controls", (Color){255,255,255,255}, (Color){40,40,60,255});
    // // Row: Drag toggle
    // MenuRow *r_drag = menurow_create();
    // int drag_enable = 0;
    // VariableInteraction *v_drag = variableinteraction_create(&drag_enable, "Drag", 0, 1, VAR_BOOL, NULL, NULL);
    // menurow_add_interaction(r_drag, v_drag);
    // menu_add_row(ctrl, r_drag);
    // Row: Pause toggle
    VariableInteraction *vi_pause = variableinteraction_create(&paused, "Paused", 0, 1, VAR_BOOL, NULL, NULL);
    MenuRow *r_pause = menurow_create();
    menurow_add_interaction(r_pause, vi_pause);
    menu_add_row(ctrl, r_pause);
    // Row: Steps per second slider (controls sim->dt = 1.0 / sps)
    double sps = 1.0 / sim->dt;
    MenuRow *r_sps = menurow_create();
    VariableInteraction *vi_sps = variableinteraction_create(&sps, "Steps/s", 15.0, 500.0, VAR_SLIDER, sim_on_step_change, sim);
    menurow_add_interaction(r_sps, vi_sps);
    menu_add_row(ctrl, r_sps);
    // Row: Grid toggle
    MenuRow *r_grid = menurow_create();
    VariableInteraction *vi_grid = variableinteraction_create(&show_grid, "Show Grid", 0, 1, VAR_BOOL, NULL, NULL);
    menurow_add_interaction(r_grid, vi_grid);
    menu_add_row(ctrl, r_grid);
    // Row: Cell size slider (0.5 .. 75)
    MenuRow *r_cell = menurow_create();
    VariableInteraction *vi_cell = variableinteraction_create(&grid_cell_size, "Cell Size", 0.5, 75.0, VAR_SLIDER, NULL, NULL);
    menurow_add_interaction(r_cell, vi_cell);
    menu_add_row(ctrl, r_cell);
    // Edit toggle will open/close the edit palette menu
    int edit_open = 0;
    VariableInteraction *vi_edit = variableinteraction_create(&edit_open, "Edit", 0, 1, VAR_BOOL, on_edit_toggle, NULL);
    MenuRow *r_edit = menurow_create();
    menurow_add_interaction(r_edit, vi_edit);
    menu_add_row(ctrl, r_edit);
    dynarray_append(menus, ctrl);

    // Edit menu state and tool selection variables (will be used by callbacks)
    Menu *edit_menu = NULL;
    Menu *select_menu = NULL;
    int current_tool = TOOL_NONE;
    int tool_select = 0, tool_node = 0, tool_dist = 0, tool_spring = 0, tool_wall = 0;
    // Selection state
    DynArray *selection = dynarray_create(8); // holds Node* / Constraint* / TriangleWall*
    int sel_filter = SEL_NONE;
    int selection_dragging = 0; // dragging selected objects
    int rect_select_active = 0; // right-button rectangle select active
    int rect_x0 = 0, rect_y0 = 0, rect_x1 = 0, rect_y1 = 0;
    // Last mouse position used for selection dragging (world-space). Using world coords
    // last pick position (world coords) used for rendering the pick cursor and drag target
    float pick_wx = 0.0f, pick_wy = 0.0f, pick_wz = 0.0f;
    int pick_active = 0;
    int mouse_left_down = 0;
    int mouse_left_down_on_ui = 0; // true if left-button down started on a menu/UI control
    
    // Placement/selection plane system
    // 6 planes: world X, Y, Z and camera-relative X, Y, Z
    enum PlaneType { PLANE_WORLD_X = 0, PLANE_WORLD_Y, PLANE_WORLD_Z, PLANE_CAM_X, PLANE_CAM_Y, PLANE_CAM_Z };
    int current_plane = PLANE_WORLD_Y;  // start with world Y plane (horizontal ground)
    float plane_offsets[6] = { 0.0f, 0.0f, 0.0f, 10.0f, 10.0f, 10.0f };  // remembered offsets for each plane (camera planes start at +10)
    float plane_offset_step = 10.0f;  // how much to adjust with left/right arrows
    float selection_distance_threshold = 20.0f;  // max distance from plane for selection
    
    EditData edata;
    edata.menus = menus; edata.edit_menu = &edit_menu; edata.select_menu = &select_menu; edata.win_w = &win_w; edata.win_h = &win_h;
    edata.tool_select = &tool_select; edata.tool_node = &tool_node; edata.tool_dist = &tool_dist; edata.tool_spring = &tool_spring; edata.tool_wall = &tool_wall; edata.current_tool = &current_tool;
    edata.sim = sim; edata.selection = selection; edata.drag_enabled = NULL;
    edata.sel_filter = &sel_filter;
    edata.node_mass = NULL; edata.node_friction = NULL; edata.node_anchored = NULL; edata.node_collide_with_walls = NULL;
    // allocate persistent storage for drag controls so menu recreation does not
    // reset or point to freed memory. These will be used by the selection menu.
    edata.drag_enabled = malloc(sizeof(int)); *edata.drag_enabled = 0;
    edata.drag_strength = malloc(sizeof(double)); *edata.drag_strength = 200.0;
    edata.drag_damping = malloc(sizeof(double)); *edata.drag_damping = 8.0;
    // allocate persistent storage for spring/wall tool preferences
    edata.spring_stiffness = malloc(sizeof(double)); *edata.spring_stiffness = 10.0;
    edata.wall_add_dist = malloc(sizeof(int)); *edata.wall_add_dist = 0;
    edata.wall_friction = malloc(sizeof(double)); *edata.wall_friction = 1.0;
    edata.wall_restitution = malloc(sizeof(double)); *edata.wall_restitution = 0.0;
    // attach user_data for vi_edit now that edata is set
    vi_edit->callback_data = &edata;

    int n = 6;

    // pending node for pair-based tool actions (dist/spring/wall)
    Node *pending_tool_node = NULL;
    // last click candidates for cycling with arrow keys
    DynArray *last_candidates = NULL; // ClickCandidate*
    int last_candidate_index = 0;

    // Optional: run octagon mesh generation test
    if (run_mesh_test) {
        DynArray *poly = dynarray_create(8);
        const float cx = 0.0f, cy = 220.0f, R = 100.0f;
        for (int i = 0; i < n; ++i) {
            float a = (float)i * 2.0f * 3.14159265f / (float)n;
            Node *pn = node_create(-1, 1.0f, cx + R * cosf(a), cy + R * sinf(a), 0.0f);
            pn->mass = 150.0f;
            // do not add to simulator; pass as polygon vertices
            dynarray_append(poly, pn);
        }
        DynArray *tris = simulator_generate_mesh_from_nodes(sim, poly, mesh_k);
        int tcount = tris ? (int)dynarray_size(tris) : 0;
        printf("Mesh test: octagon with k=%d produced %d triangles and %zu nodes in simulator\n", mesh_k, tcount, dynarray_size(sim->nodes));
        // free temporary polygon nodes (they were not added to sim)
        for (size_t i = 0; i < dynarray_size(poly); ++i) node_free((Node*)dynarray_get(poly, i));
        dynarray_free(poly, NULL);
        // free triangle list structure but do not free Node*s (they belong to simulator)
        if (tris) {
            for (size_t i = 0; i < dynarray_size(tris); ++i) free(dynarray_get(tris, i));
            dynarray_free(tris, NULL);
        }
    }
    
    /* COMMENTED OUT OLD SCENE
    const float offset = 50.0f;
    const float box_size = 200.0f;
    const float half = box_size * 0.5f;

    // We'll keep an array of node pointers so we can reference by index like the python snippet
    Node *nodes_arr[16];
    int ni = 0;

    // Box corners (friction = 0)
    nodes_arr[ni] = node_create(-1, 1.0f, -half + offset, -half, 0.0f); nodes_arr[ni]->friction = 0.0f; simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f,  half + offset, -half, 0.0f); nodes_arr[ni]->friction = 0.0f; simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f,  half + offset,  half, 0.0f); nodes_arr[ni]->friction = 0.0f; simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f, -half + offset,  half, 0.0f); nodes_arr[ni]->friction = 0.0f; simulator_add_node(sim, nodes_arr[ni++]);

    // Sleeve top and bottom panels (anchored walls)
    float sleeve_y = half + 5;
    float sleeve_length = box_size + 80.0f;
    float sleeve_left = -sleeve_length * 0.5f;
    float sleeve_right =  sleeve_length * 0.5f;

    nodes_arr[ni] = node_create(-1, 1.0f, sleeve_left + offset,  sleeve_y, 0.0f); simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f, sleeve_right + offset, sleeve_y, 0.0f); simulator_add_node(sim, nodes_arr[ni++]);
    // Create triangle wall for top boundary (A, B, C where C is midpoint offset in z)
    Node *wtop_c = node_create(-1, 0.0f, 
        (nodes_arr[4]->pos[0] + nodes_arr[5]->pos[0]) * 0.5f,
        (nodes_arr[4]->pos[1] + nodes_arr[5]->pos[1]) * 0.5f,
        10.0f);  // offset in z
    wtop_c->anchored = true; wtop_c->collide_with_walls = false;
    simulator_add_node(sim, wtop_c);
    TriangleWall *wtop = trianglewall_create(nodes_arr[4], nodes_arr[5], wtop_c, 1.0f, 0.0f); 
    simulator_add_wall(sim, wtop);

    nodes_arr[ni] = node_create(-1, 1.0f, sleeve_left + offset, -sleeve_y, 0.0f); simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f, sleeve_right + offset,-sleeve_y, 0.0f); simulator_add_node(sim, nodes_arr[ni++]);
    // Create triangle wall for bottom boundary
    Node *wbot_c = node_create(-1, 0.0f,
        (nodes_arr[6]->pos[0] + nodes_arr[7]->pos[0]) * 0.5f,
        (nodes_arr[6]->pos[1] + nodes_arr[7]->pos[1]) * 0.5f,
        10.0f);
    wbot_c->anchored = true; wbot_c->collide_with_walls = false;
    simulator_add_node(sim, wbot_c);
    TriangleWall *wbot = trianglewall_create(nodes_arr[6], nodes_arr[7], wbot_c, 1.0f, 0.0f);
    simulator_add_wall(sim, wbot);

    // Additional support / spacer nodes
    nodes_arr[ni] = node_create(-1, 1.0f, -box_size + offset, 0.0f, 0.0f); simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f, -box_size - half,    0.0f, 0.0f); simulator_add_node(sim, nodes_arr[ni++]);
    // a heavier moving node
    nodes_arr[ni] = node_create(-1, 50.0f, -box_size - half, 30.0f, 0.0f); simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f,  half * 1.25f + offset, 0.0f, 0.0f); simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f,  box_size * 1.5f + offset,  0.0f, 0.0f); simulator_add_node(sim, nodes_arr[ni++]);
    // give node index 10 an initial leftward velocity
    if (ni > 10) {
        nodes_arr[10]->vel[0] = -25.0f; nodes_arr[10]->vel[1] = 0.0f;
    }

    // Anchor constraints (anchor at current node position)
    // panels top/bot and two support nodes
    Constraint *ac;
    ac = anchorconstraint_create(nodes_arr[4], nodes_arr[4]->pos[0], nodes_arr[4]->pos[1], nodes_arr[4]->pos[2]); simulator_add_constraint(sim, ac);
    ac = anchorconstraint_create(nodes_arr[5], nodes_arr[5]->pos[0], nodes_arr[5]->pos[1], nodes_arr[5]->pos[2]); simulator_add_constraint(sim, ac);
    ac = anchorconstraint_create(nodes_arr[6], nodes_arr[6]->pos[0], nodes_arr[6]->pos[1], nodes_arr[6]->pos[2]); simulator_add_constraint(sim, ac);
    ac = anchorconstraint_create(nodes_arr[7], nodes_arr[7]->pos[0], nodes_arr[7]->pos[1], nodes_arr[7]->pos[2]); simulator_add_constraint(sim, ac);
    ac = anchorconstraint_create(nodes_arr[9], nodes_arr[9]->pos[0], nodes_arr[9]->pos[1], nodes_arr[9]->pos[2]); simulator_add_constraint(sim, ac);
    ac = anchorconstraint_create(nodes_arr[12], nodes_arr[12]->pos[0], nodes_arr[12]->pos[1], nodes_arr[12]->pos[2]); simulator_add_constraint(sim, ac);

    // Box structural constraints (edges, diagonals, and some internal links)
    {
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[0], nodes_arr[1], -1));
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[1], nodes_arr[2], -1));
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[2], nodes_arr[3], -1));
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[3], nodes_arr[0], -1));
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[0], nodes_arr[2], -1));
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[1], nodes_arr[3], -1));
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[8], nodes_arr[3], -1));
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[8], nodes_arr[0], -1));
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[9], nodes_arr[10], -1));
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[10], nodes_arr[8], -1));
        float r11_12 = sqrtf((nodes_arr[11]->pos[0]-nodes_arr[12]->pos[0])*(nodes_arr[11]->pos[0]-nodes_arr[12]->pos[0]) + (nodes_arr[11]->pos[1]-nodes_arr[12]->pos[1])*(nodes_arr[11]->pos[1]-nodes_arr[12]->pos[1]));
        simulator_add_constraint(sim, springconstraint_create(nodes_arr[11], nodes_arr[12], 10.0f, r11_12));
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[1], nodes_arr[11], -1));
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[2], nodes_arr[11], -1));
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[0], nodes_arr[11], -1));
        simulator_add_constraint(sim, distconstraint_create(nodes_arr[3], nodes_arr[11], -1));
    }

    // Sleeve inner walls along box top/bottom
    Node *w1_c = node_create(-1, 0.0f,
        (nodes_arr[0]->pos[0] + nodes_arr[1]->pos[0]) * 0.5f,
        (nodes_arr[0]->pos[1] + nodes_arr[1]->pos[1]) * 0.5f,
        10.0f);
    w1_c->anchored = true; w1_c->collide_with_walls = false;
    simulator_add_node(sim, w1_c);
    TriangleWall *w1 = trianglewall_create(nodes_arr[0], nodes_arr[1], w1_c, 1.0f, 0.0f);
    simulator_add_wall(sim, w1);
    Node *w2_c = node_create(-1, 0.0f,
        (nodes_arr[2]->pos[0] + nodes_arr[3]->pos[0]) * 0.5f,
        (nodes_arr[2]->pos[1] + nodes_arr[3]->pos[1]) * 0.5f,
        10.0f);
    w2_c->anchored = true; w2_c->collide_with_walls = false;
    simulator_add_node(sim, w2_c);
    TriangleWall *w2 = trianglewall_create(nodes_arr[2], nodes_arr[3], w2_c, 1.0f, 0.0f);
    simulator_add_wall(sim, w2);
    // Create a dense grid of free nodes below the bottom panel and attach
    // the two bottom anchors to the grid with springs. Grid nodes are
    // small-radius, and only horizontal/vertical neighbor connections are
    // created (distance constraints + walls). No diagonals.
    {
        const int GRID_COLS = 0;   // pretty dense horizontally
        const int GRID_ROWS = 0;    // several rows
        Node *grid[GRID_ROWS][GRID_COLS];
        const float spacing = 7.0f; // spacing between grid nodes (world units)
        // center the grid between the two bottom anchors (nodes_arr[6], nodes_arr[7])
        float anchor_x0 = nodes_arr[6]->pos[0];
        float anchor_x1 = nodes_arr[7]->pos[0];
        float grid_center_x = 0.5f * (anchor_x0 + anchor_x1);
        float grid_start_x = grid_center_x - ((GRID_COLS - 1) * spacing) * 0.5f;
        // place the grid below the bottom anchor line
        float grid_start_y = nodes_arr[6]->pos[1] - 28.0f;

        // create nodes
        for (int r = 0; r < GRID_ROWS; ++r) {
            for (int c = 0; c < GRID_COLS; ++c) {
                float x = grid_start_x + c * spacing;
                float y = grid_start_y - r * spacing;
                Node *n = node_create(-1, 1.0f, x, y, 0.0f);
                if (!n) continue;
                // reduce visual radius so grid looks dense and tidy
                n->radius = 1.0f;
                // some default friction for interactions with walls
                n->friction = 0.5f;
                simulator_add_node(sim, n);
                grid[r][c] = n;
            }
        }

        // connect neighbors with horizontal and vertical distance constraints + walls
        for (int r = 0; r < GRID_ROWS; ++r) {
            for (int c = 0; c < GRID_COLS; ++c) {
                Node *n = grid[r][c];
                if (!n) continue;
                // horizontal neighbor
                if (c + 1 < GRID_COLS) {
                    Node *hn = grid[r][c + 1];
                    if (hn) {
                        // distance constraint (rest = current distance)
                        float dx = n->pos[0] - hn->pos[0];
                        float dy = n->pos[1] - hn->pos[1];
                        float rest = sqrtf(dx*dx + dy*dy);
                        Constraint *dc = distconstraint_create(n, hn, rest);
                        if (dc) simulator_add_constraint(sim, dc);
                        // also add a wall segment between neighbors so collisions behave
                        Node *ws_c = node_create(-1, 0.0f,
                            (n->pos[0] + hn->pos[0]) * 0.5f,
                            (n->pos[1] + hn->pos[1]) * 0.5f,
                            10.0f);
                        ws_c->anchored = true; ws_c->collide_with_walls = false;
                        simulator_add_node(sim, ws_c);
                        TriangleWall *ws = trianglewall_create(n, hn, ws_c, 1.0f, 0.0f);
                        if (ws) simulator_add_wall(sim, ws);
                    }
                }
                // vertical neighbor
                if (r + 1 < GRID_ROWS) {
                    Node *vn = grid[r + 1][c];
                    if (vn) {
                        float dx = n->pos[0] - vn->pos[0];
                        float dy = n->pos[1] - vn->pos[1];
                        float rest = sqrtf(dx*dx + dy*dy);
                        Constraint *dc = distconstraint_create(n, vn, rest);
                        if (dc) simulator_add_constraint(sim, dc);
                        Node *ws_c = node_create(-1, 0.0f,
                            (n->pos[0] + vn->pos[0]) * 0.5f,
                            (n->pos[1] + vn->pos[1]) * 0.5f,
                            10.0f);
                        ws_c->anchored = true; ws_c->collide_with_walls = false;
                        simulator_add_node(sim, ws_c);
                        TriangleWall *ws = trianglewall_create(n, vn, ws_c, 1.0f, 0.0f);
                        if (ws) simulator_add_wall(sim, ws);
                    }
                }
            }
        }

        // attach the two bottom anchors to the top row of the grid with springs
        // (one spring from each anchor to the nearest grid edge). Use a moderate
        // stiffness so the grid is influenced but still free.
        float spring_stiff = 5000.0f;
        if (GRID_COLS > 0 && GRID_ROWS > 0) {
            Node *left_top = grid[0][0];
            Node *right_top = grid[0][GRID_COLS - 1];
            if (left_top) {
                float dx = left_top->pos[0] - nodes_arr[6]->pos[0];
                float dy = left_top->pos[1] - nodes_arr[6]->pos[1];
                float rest = sqrtf(dx*dx + dy*dy);
                Constraint *s = springconstraint_create(nodes_arr[6], left_top, spring_stiff, rest);
                if (s) simulator_add_constraint(sim, s);
            }
            if (right_top) {
                float dx = right_top->pos[0] - nodes_arr[7]->pos[0];
                float dy = right_top->pos[1] - nodes_arr[7]->pos[1];
                float rest = sqrtf(dx*dx + dy*dy);
                Constraint *s = springconstraint_create(nodes_arr[7], right_top, spring_stiff, rest);
                if (s) simulator_add_constraint(sim, s);
            }
        }
    }
    END OLD SCENE */

    // NEW SCENE: Ground + Tetrahedron
    // Position in front of camera (0,0,500), slightly below (y < 0)
    // Camera looks along -Z axis, so objects should be at negative Z
    
    // Ground plane: 2 large anchored triangles forming a square
    float ground_size = 200.0f;
    float ground_z = -300.0f;  // in front of camera
    float ground_y = -50.0f;   // below camera
    
    // Ground corners
    Node *g1 = node_create(-1, 1.0f, -ground_size, ground_y, ground_z - ground_size);
    Node *g2 = node_create(-1, 1.0f,  ground_size, ground_y, ground_z - ground_size);
    Node *g3 = node_create(-1, 1.0f,  ground_size, ground_y, ground_z + ground_size);
    Node *g4 = node_create(-1, 1.0f, -ground_size, ground_y, ground_z + ground_size);
    g1->anchored = true; g2->anchored = true; g3->anchored = true; g4->anchored = true;
    simulator_add_node(sim, g1);
    simulator_add_node(sim, g2);
    simulator_add_node(sim, g3);
    simulator_add_node(sim, g4);
    
    // Ground triangle walls
    TriangleWall *ground1 = trianglewall_create(g1, g2, g3, 1.0f, 0.0f);
    TriangleWall *ground2 = trianglewall_create(g1, g3, g4, 1.0f, 0.0f);
    simulator_add_wall(sim, ground1);
    simulator_add_wall(sim, ground2);
    
    // Tetrahedron above ground, pointy end down
    float tet_size = 30.0f;
    float tet_center_z = ground_z;  // same Z as ground center
    float tet_bottom_y = ground_y + 20.0f;  // 20 units above ground
    float tet_height = tet_size * sqrtf(2.0f / 3.0f);  // height of regular tetrahedron
    
    // Bottom vertex (pointy end)
    Node *t_bottom = node_create(-1, 1.0f, 0.0f, tet_bottom_y, tet_center_z);
    simulator_add_node(sim, t_bottom);
    
    // Top 3 vertices forming equilateral triangle
    float top_y = tet_bottom_y + tet_height;
    float angle_offset = 3.14159265f / 2.0f;  // start at top
    Node *t1 = node_create(-1, 1.1f, 
        tet_size * cosf(angle_offset), 
        top_y, 
        tet_center_z + tet_size * sinf(angle_offset));
    Node *t2 = node_create(-1, 1.0f, 
        tet_size * cosf(angle_offset + 2.0f * 3.14159265f / 3.0f), 
        top_y, 
        tet_center_z + tet_size * sinf(angle_offset + 2.0f * 3.14159265f / 3.0f));
    Node *t3 = node_create(-1, 1.05f, 
        tet_size * cosf(angle_offset + 4.0f * 3.14159265f / 3.0f), 
        top_y, 
        tet_center_z + tet_size * sinf(angle_offset + 4.0f * 3.14159265f / 3.0f));
    simulator_add_node(sim, t1);
    simulator_add_node(sim, t2);
    simulator_add_node(sim, t3);
    
    // Distance constraints for all 6 edges of tetrahedron
    simulator_add_constraint(sim, distconstraint_create(t_bottom, t1, -1));
    simulator_add_constraint(sim, distconstraint_create(t_bottom, t2, -1));
    simulator_add_constraint(sim, distconstraint_create(t_bottom, t3, -1));
    simulator_add_constraint(sim, distconstraint_create(t1, t2, -1));
    simulator_add_constraint(sim, distconstraint_create(t2, t3, -1));
    simulator_add_constraint(sim, distconstraint_create(t3, t1, -1));
    printf("Generated initial scenario with %zu nodes, %zu constraints, %zu walls\n",
        dynarray_size(sim->nodes), dynarray_size(sim->constraints), dynarray_size(sim->walls));

    printf("Entering main loop. Close window to exit.\n");
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = 0;
            }

            // Generalized input handler dispatch
            for (int i = 0; i < input_handler_count; ++i) {
                InputHandler *handler = input_handlers[i];
                if (handler && handler->should_run && handler->should_run(handler, &event)) {
                    if (handler->run) handler->run(handler, &event);
                }
            }

            // Menu input dispatch: run each registered menu in order, stop when one handles the event
            if (event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_MOUSEBUTTONUP) {
                int state = (event.type == SDL_MOUSEBUTTONDOWN) ? SDL_PRESSED : SDL_RELEASED;
                int mx = event.button.x, my = event.button.y, button = event.button.button;
                if (state == SDL_PRESSED && button == SDL_BUTTON_LEFT) mouse_left_down = 1;
                else if (state == SDL_RELEASED && button == SDL_BUTTON_LEFT) mouse_left_down = 0;
                int menu_handled = 0;
                // iterate menus in reverse order (higher z-index drawn last -> receive events first)
                for (int mi = (int)menus->size - 1; mi >= 0; --mi) {
                    Menu *m = (Menu*)menus->items[mi];
                    if (!m) continue;
                    if (menu_handle_mouse_button(m, button, state, mx, my)) { menu_handled = 1; break; }
                }

                // Track whether the left-button event began on the UI. If so, mark
                // mouse_left_down_on_ui so per-frame drag forces are suppressed while
                // the user is interacting with menus/sliders.
                if (button == SDL_BUTTON_LEFT) {
                    if (state == SDL_PRESSED) {
                        mouse_left_down_on_ui = menu_handled ? 1 : 0;
                    } else if (state == SDL_RELEASED) {
                        mouse_left_down_on_ui = 0;
                    }
                }

                // If not handled by UI menus, handle current tool actions (select, add-node, etc.)
                if (!menu_handled) {
                    // Convert screen to world coords using ray-plane intersection
                    float wx = 0.0f, wy = 0.0f, wz = 0.0f;
                    int hit = screen_to_world_plane(mx, my, win_w, win_h,
                                                    cam_x, cam_y, cam_z, cam_yaw, cam_pitch,
                                                    current_plane, plane_offsets[current_plane],
                                                    &wx, &wy, &wz);
                    if (!hit) {
                        // Fallback if ray doesn't hit plane (shouldn't happen normally)
                        wx = (float)mx; wy = (float)my; wz = 0.0f;
                    }
                    // store last pick position so we can render it on screen
                    pick_wx = wx; pick_wy = wy; pick_wz = wz; pick_active = 1;

                    if (current_tool == TOOL_ADD_NODE) {
                        // In +Node mode, left-click spawns a node at world coords with parameters from edit menu (if present)
                        if (event.type == SDL_MOUSEBUTTONDOWN && button == SDL_BUTTON_LEFT) {
                            double mass = 1.0;
                            double friction_v = 0.0;
                            int anchored_v = 0;
                            int collide_v = 1;
                            if (edata.node_mass) mass = *edata.node_mass;
                            if (edata.node_friction) friction_v = *edata.node_friction;
                            if (edata.node_anchored) anchored_v = *edata.node_anchored;
                            if (edata.node_collide_with_walls) collide_v = *edata.node_collide_with_walls;
                            Node *nn = node_create(-1, (float)mass, wx, wy, wz);
                            if (nn) {
                                nn->friction = (float)friction_v;
                                nn->anchored = anchored_v ? true : false;
                                nn->collide_with_walls = collide_v ? true : false;
                                simulator_add_node(sim, nn);
                                // optionally select the newly created node
                                sel_filter = SEL_NODE;
                                dynarray_append(selection, nn);
                            }
                        }
                    } else if (current_tool == TOOL_SELECT) {
                        const float pick_px = 8.0f;
                        const float pick_world = pick_px; // TODO: scale based on camera distance in 3D

                        // Get plane parameters for distance filtering
                        float plane_nx, plane_ny, plane_nz, plane_d;
                        get_plane_params(current_plane, plane_offsets[current_plane], cam_yaw, cam_pitch,
                                        cam_x, cam_y, cam_z, &plane_nx, &plane_ny, &plane_nz, &plane_d);

                        // LEFT click: build candidate list at pick point and choose first
                        if (event.type == SDL_MOUSEBUTTONDOWN && button == SDL_BUTTON_LEFT) {
                            if (last_candidates) { dynarray_free(last_candidates, free); last_candidates = NULL; }
                            last_candidates = dynarray_create(8);
                            int allow_nodes = 0, allow_constraints = 0, allow_walls = 0;
                            size_t sel_sz = dynarray_size(selection);
                            if (sel_sz == 0) {
                                allow_nodes = allow_constraints = allow_walls = 1;
                            } else {
                                if (sel_filter == SEL_NODE) allow_nodes = 1;
                                else if (sel_filter == SEL_CONSTRAINT) allow_constraints = 1;
                                else if (sel_filter == SEL_WALL) allow_walls = 1;
                                else { allow_nodes = allow_constraints = allow_walls = 1; }
                            }

                            if (allow_nodes) {
                                for (size_t ii = 0; ii < dynarray_size(sim->nodes); ++ii) {
                                    Node *nn = (Node*)dynarray_get(sim->nodes, ii);
                                    if (!nn) continue;
                                    // Check absolute distance from plane (both sides)
                                    float dist_to_plane = fabsf(point_plane_distance(nn->pos[0], nn->pos[1], nn->pos[2],
                                                                               plane_nx, plane_ny, plane_nz, plane_d));
                                    if (dist_to_plane > selection_distance_threshold) continue;
                                    
                                    // Check distance from pick position (2D screen space approximation)
                                    float dx = wx - nn->pos[0]; float dy = wy - nn->pos[1]; float dz = wz - nn->pos[2];
                                    float d2 = dx*dx + dy*dy + dz*dz;
                                    float r = nn->radius + pick_world;
                                    if (d2 <= r*r) {
                                        ClickCandidate *cc = malloc(sizeof(ClickCandidate)); cc->type = SEL_NODE; cc->obj = nn; dynarray_append(last_candidates, cc);
                                    }
                                }
                            }
                            if (allow_constraints && sim->constraints) {
                                for (size_t ii = 0; ii < dynarray_size(sim->constraints); ++ii) {
                                    Constraint *c = (Constraint*)dynarray_get(sim->constraints, ii);
                                    if (!c) continue;
                                    if (c->type == CT_DIST || c->type == CT_SPRING) {
                                        if (!c->node || !c->other) continue;
                                        // Check if constraint endpoints are near plane (both sides)
                                        float dist1 = fabsf(point_plane_distance(c->node->pos[0], c->node->pos[1], c->node->pos[2],
                                                                          plane_nx, plane_ny, plane_nz, plane_d));
                                        float dist2 = fabsf(point_plane_distance(c->other->pos[0], c->other->pos[1], c->other->pos[2],
                                                                          plane_nx, plane_ny, plane_nz, plane_d));
                                        if (dist1 > selection_distance_threshold && dist2 > selection_distance_threshold) continue;
                                        
                                        float d2 = point_segment_distance2(wx, wy, c->node->pos[0], c->node->pos[1], c->other->pos[0], c->other->pos[1]);
                                        if (d2 <= pick_world * pick_world) {
                                            ClickCandidate *cc = malloc(sizeof(ClickCandidate)); cc->type = SEL_CONSTRAINT; cc->obj = c; dynarray_append(last_candidates, cc);
                                        }
                                    }
                                }
                            }
                            if (allow_walls && sim->walls) {
                                for (size_t ii = 0; ii < dynarray_size(sim->walls); ++ii) {
                                    TriangleWall *w = (TriangleWall*)dynarray_get(sim->walls, ii);
                                    if (!w || !w->A || !w->B) continue;
                                    // Check if wall endpoints are near plane (both sides)
                                    float dist1 = fabsf(point_plane_distance(w->A->pos[0], w->A->pos[1], w->A->pos[2],
                                                                       plane_nx, plane_ny, plane_nz, plane_d));
                                    float dist2 = fabsf(point_plane_distance(w->B->pos[0], w->B->pos[1], w->B->pos[2],
                                                                       plane_nx, plane_ny, plane_nz, plane_d));
                                    if (dist1 > selection_distance_threshold && dist2 > selection_distance_threshold) continue;
                                    
                                    float d2 = point_segment_distance2(wx, wy, w->A->pos[0], w->A->pos[1], w->B->pos[0], w->B->pos[1]);
                                    if (d2 <= pick_world * pick_world) {
                                        ClickCandidate *cc = malloc(sizeof(ClickCandidate)); cc->type = SEL_WALL; cc->obj = w; dynarray_append(last_candidates, cc);
                                    }
                                }
                            }

                            if (dynarray_size(last_candidates) == 0) {
                                // nothing under cursor; normally clear selection. However,
                                // if Drag is enabled we keep the current selection so the
                                // user can click anywhere to pull the previously-selected
                                // nodes. Update pick position instead.
                                if (!(edata.drag_enabled && edata.drag_enabled[0])) {
                                    selection->size = 0;
                                    // refresh selection menu
                                    destroy_select_menu_if_present(&edata);
                                    create_select_menu_if_needed(&edata);
                                } else {
                                    // keep selection; update pick so drag has a target
                                    pick_wx = wx; pick_wy = wy; pick_wz = wz; pick_active = 1;
                                }
                            } else {
                                // pick first candidate and set selection/filter
                                last_candidate_index = 0;
                                ClickCandidate *cc = (ClickCandidate*)dynarray_get(last_candidates, 0);
                                selection->size = 0;
                                if (cc->type == SEL_NODE) { sel_filter = SEL_NODE; dynarray_append(selection, cc->obj); }
                                else if (cc->type == SEL_CONSTRAINT) { sel_filter = SEL_CONSTRAINT; dynarray_append(selection, cc->obj); }
                                else if (cc->type == SEL_WALL) { sel_filter = SEL_WALL; dynarray_append(selection, cc->obj); }
                                // Note: dragging is controlled by the Selection menu toggle. Do not
                                // auto-enter a special "selection_dragging" mode here — force-drag
                                // will be applied whenever the user is clicking and the toggle is enabled.
                                // refresh selection menu rows for this type
                                destroy_select_menu_if_present(&edata);
                                create_select_menu_if_needed(&edata);
                            }
                        }

                        // RIGHT mouse: rectangle select start/finish handled below (button up/down)
                        if (event.type == SDL_MOUSEBUTTONDOWN && button == SDL_BUTTON_RIGHT) {
                            rect_select_active = 1; rect_x0 = mx; rect_y0 = my; rect_x1 = mx; rect_y1 = my;
                            // TODO: proper 3D picking
                            pick_wx = (float)mx;
                            pick_wy = (float)my;
                            pick_active = 1;
                        } else if (event.type == SDL_MOUSEBUTTONUP && button == SDL_BUTTON_LEFT) {
                            // leave dragging toggle state unchanged; mouse_left_down is cleared
                            // by the top-level handler above. Keep pick active for rendering.
                            pick_active = 1;
                        } else if (event.type == SDL_MOUSEBUTTONUP && button == SDL_BUTTON_RIGHT) {
                            if (rect_select_active) {
                                rect_select_active = 0;
                                int rx0 = rect_x0 < rect_x1 ? rect_x0 : rect_x1;
                                int ry0 = rect_y0 < rect_y1 ? rect_y0 : rect_y1;
                                int rx1 = rect_x0 > rect_x1 ? rect_x0 : rect_x1;
                                int ry1 = rect_y0 > rect_y1 ? rect_y0 : rect_y1;
                                // TODO: proper 3D rectangle selection
                                float wx0 = (float)rx0;
                                float wy1 = (float)ry0;
                                float wx1 = (float)rx1;
                                float wy0 = (float)ry1;
                                // choose filter by finding first object inside rect: nodes -> constraints -> walls
                                int found_type = SEL_NONE;
                                size_t n_nodes = dynarray_size(sim->nodes);
                                for (size_t ii = 0; ii < n_nodes; ++ii) {
                                    Node *nn = (Node*)dynarray_get(sim->nodes, ii);
                                    if (!nn) continue;
                                    if (nn->pos[0] >= wx0 && nn->pos[0] <= wx1 && nn->pos[1] >= wy0 && nn->pos[1] <= wy1) { found_type = SEL_NODE; break; }
                                }
                                if (found_type == SEL_NONE) found_type = SEL_NODE; // default
                                sel_filter = found_type;
                                if (sel_filter == SEL_NODE) {
                                    selection->size = 0;
                                    for (size_t ii = 0; ii < dynarray_size(sim->nodes); ++ii) {
                                        Node *nn = (Node*)dynarray_get(sim->nodes, ii);
                                        if (!nn) continue;
                                        if (nn->pos[0] >= wx0 && nn->pos[0] <= wx1 && nn->pos[1] >= wy0 && nn->pos[1] <= wy1) dynarray_append(selection, nn);
                                    }
                                }
                                // do not clear the drag toggle here; leave it to the user to toggle
                                // drag mode in the Selection menu
                                pick_active = 1;
                                // refresh selection menu
                                destroy_select_menu_if_present(&edata);
                                create_select_menu_if_needed(&edata);
                            }
                        }
                    }
                    else if (current_tool == TOOL_ADD_DIST || current_tool == TOOL_ADD_SPRING || current_tool == TOOL_ADD_WALL) {
                        const float pick_px = 8.0f;
                        const float pick_world = pick_px; // TODO: scale based on camera distance
                        if (event.type == SDL_MOUSEBUTTONDOWN && button == SDL_BUTTON_LEFT) {
                            // find node under mouse
                            Node *found = NULL;
                            for (size_t ii = 0; ii < dynarray_size(sim->nodes); ++ii) {
                                Node *nn = (Node*)dynarray_get(sim->nodes, ii);
                                if (!nn) continue;
                                float dx = wx - nn->pos[0]; float dy = wy - nn->pos[1];
                                float d2 = dx*dx + dy*dy;
                                float r = nn->radius + pick_world;
                                if (d2 <= r*r) { found = nn; break; }
                            }
                            if (found) {
                                if (pending_tool_node == NULL) {
                                    // set the first endpoint
                                    pending_tool_node = found;
                                } else if (pending_tool_node == found) {
                                    // clicked same node again: clear pending
                                    pending_tool_node = NULL;
                                } else {
                                    // create the requested constraint/wall between pending_tool_node and found
                                    if (current_tool == TOOL_ADD_DIST) {
                                        Constraint *c = distconstraint_create(pending_tool_node, found, -1);
                                        if (c) simulator_add_constraint(sim, c);
                                    } else if (current_tool == TOOL_ADD_SPRING) {
                                        // rest length = current distance; stiffness from edit prefs
                                        float dx = pending_tool_node->pos[0] - found->pos[0];
                                        float dy = pending_tool_node->pos[1] - found->pos[1];
                                        float rest = sqrtf(dx*dx + dy*dy);
                                        float stiff = edata.spring_stiffness ? (float)(*(edata.spring_stiffness)) : 10.0f;
                                        Constraint *c = springconstraint_create(pending_tool_node, found, stiff, rest);
                                        if (c) simulator_add_constraint(sim, c);
                                    } else if (current_tool == TOOL_ADD_WALL) {
                                        float wf = edata.wall_friction ? (float)(*(edata.wall_friction)) : 1.0f;
                                        float wr = edata.wall_restitution ? (float)(*(edata.wall_restitution)) : 0.0f;
                                        Node *w_c = node_create(-1, 0.0f,
                                            (pending_tool_node->pos[0] + found->pos[0]) * 0.5f,
                                            (pending_tool_node->pos[1] + found->pos[1]) * 0.5f,
                                            10.0f);
                                        w_c->anchored = true; w_c->collide_with_walls = false;
                                        simulator_add_node(sim, w_c);
                                        TriangleWall *w = trianglewall_create(pending_tool_node, found, w_c, wf, wr);
                                        if (w) simulator_add_wall(sim, w);
                                        // if +Dist toggle is set, also add a distance constraint between the nodes
                                        if (edata.wall_add_dist && edata.wall_add_dist[0]) {
                                            Constraint *dc = distconstraint_create(pending_tool_node, found, -1);
                                            if (dc) simulator_add_constraint(sim, dc);
                                        }
                                    }
                                    // clear pending after creation
                                    pending_tool_node = NULL;
                                }
                            }
                        }
                    }
                }

            } else if (event.type == SDL_MOUSEMOTION) {
                int mx = event.motion.x, my = event.motion.y;
                int menu_handled = 0;
                for (int mi = (int)menus->size - 1; mi >= 0; --mi) {
                    Menu *m = (Menu*)menus->items[mi];
                    if (!m) continue;
                    if (menu_handle_mouse_motion(m, mx, my)) { menu_handled = 1; break; }
                }
                // update rectangle selection while dragging
                if (!menu_handled && rect_select_active) {
                    rect_x1 = mx; rect_y1 = my;
                }
                // update pick position while moving mouse (show where selection will search)
                if (!menu_handled && (*(edata.current_tool) == TOOL_SELECT || *(edata.current_tool) == TOOL_ADD_NODE ||
                                      *(edata.current_tool) == TOOL_ADD_DIST || *(edata.current_tool) == TOOL_ADD_SPRING ||
                                      *(edata.current_tool) == TOOL_ADD_WALL)) {
                    // Use ray-plane intersection for accurate 3D position
                    float wx = 0.0f, wy = 0.0f, wz = 0.0f;
                    int hit = screen_to_world_plane(mx, my, win_w, win_h,
                                                    cam_x, cam_y, cam_z, cam_yaw, cam_pitch,
                                                    current_plane, plane_offsets[current_plane],
                                                    &wx, &wy, &wz);
                    if (hit) {
                        pick_wx = wx;
                        pick_wy = wy;
                        pick_wz = wz;
                        pick_active = 1;
                    }
                }
                // Force-drag is applied continuously in the per-frame update (see below).
            }

            // Camera controls: WASD for movement, SHIFT for up, CTRL for down, P to pause, space+drag to rotate, scroll for zoom/speed
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_SPACE) space_down = 1;
                else if (event.key.keysym.sym == SDLK_w) key_w = 1;
                else if (event.key.keysym.sym == SDLK_a) key_a = 1;
                else if (event.key.keysym.sym == SDLK_s) key_s = 1;
                else if (event.key.keysym.sym == SDLK_d) key_d = 1;
                else if (event.key.keysym.sym == SDLK_LSHIFT || event.key.keysym.sym == SDLK_RSHIFT) key_shift = 1;
                else if (event.key.keysym.sym == SDLK_LCTRL || event.key.keysym.sym == SDLK_RCTRL) key_ctrl = 1;
                else if (event.key.keysym.sym == SDLK_p) {
                    paused = !paused; // Toggle pause
                }
                // Plane selection with up/down arrows
                else if (event.key.keysym.sym == SDLK_UP) {
                    current_plane = (current_plane + 1) % 6;
                }
                else if (event.key.keysym.sym == SDLK_DOWN) {
                    current_plane = (current_plane - 1 + 6) % 6;
                }
                // Plane offset adjustment with left/right arrows (unless in select tool with candidates)
                else if (event.key.keysym.sym == SDLK_LEFT) {
                    if (current_tool == TOOL_SELECT && last_candidates && dynarray_size(last_candidates) > 0) {
                        // Selection candidate cycling
                        int dir = -1;
                        int count = (int)dynarray_size(last_candidates);
                        int ni = (last_candidate_index + dir) % count;
                        if (ni < 0) ni += count;
                        last_candidate_index = ni;
                        ClickCandidate *cc = (ClickCandidate*)dynarray_get(last_candidates, (size_t)last_candidate_index);
                        if (cc) {
                            selection->size = 0;
                            if (cc->type == SEL_NODE) { sel_filter = SEL_NODE; dynarray_append(selection, cc->obj); }
                            else if (cc->type == SEL_CONSTRAINT) { sel_filter = SEL_CONSTRAINT; dynarray_append(selection, cc->obj); }
                            else if (cc->type == SEL_WALL) { sel_filter = SEL_WALL; dynarray_append(selection, cc->obj); }
                            destroy_select_menu_if_present(&edata);
                            create_select_menu_if_needed(&edata);
                        }
                    } else {
                        // Adjust plane offset
                        plane_offsets[current_plane] -= plane_offset_step;
                    }
                }
                else if (event.key.keysym.sym == SDLK_RIGHT) {
                    if (current_tool == TOOL_SELECT && last_candidates && dynarray_size(last_candidates) > 0) {
                        // Selection candidate cycling
                        int dir = 1;
                        int count = (int)dynarray_size(last_candidates);
                        int ni = (last_candidate_index + dir) % count;
                        if (ni < 0) ni += count;
                        last_candidate_index = ni;
                        ClickCandidate *cc = (ClickCandidate*)dynarray_get(last_candidates, (size_t)last_candidate_index);
                        if (cc) {
                            selection->size = 0;
                            if (cc->type == SEL_NODE) { sel_filter = SEL_NODE; dynarray_append(selection, cc->obj); }
                            else if (cc->type == SEL_CONSTRAINT) { sel_filter = SEL_CONSTRAINT; dynarray_append(selection, cc->obj); }
                            else if (cc->type == SEL_WALL) { sel_filter = SEL_WALL; dynarray_append(selection, cc->obj); }
                            destroy_select_menu_if_present(&edata);
                            create_select_menu_if_needed(&edata);
                        }
                    } else {
                        // Adjust plane offset
                        plane_offsets[current_plane] += plane_offset_step;
                    }
                }
            } else if (event.type == SDL_KEYUP) {
                if (event.key.keysym.sym == SDLK_SPACE) { space_down = 0; rotating_camera = 0; }
                else if (event.key.keysym.sym == SDLK_w) key_w = 0;
                else if (event.key.keysym.sym == SDLK_a) key_a = 0;
                else if (event.key.keysym.sym == SDLK_s) key_s = 0;
                else if (event.key.keysym.sym == SDLK_d) key_d = 0;
                else if (event.key.keysym.sym == SDLK_LSHIFT || event.key.keysym.sym == SDLK_RSHIFT) key_shift = 0;
                else if (event.key.keysym.sym == SDLK_LCTRL || event.key.keysym.sym == SDLK_RCTRL) key_ctrl = 0;
            } else if (event.type == SDL_MOUSEWHEEL) {
                // Scroll to adjust zoom (affects movement speed)
                float zoom_factor = 1.1f;
                if (event.wheel.y > 0) cam_zoom *= zoom_factor;
                else if (event.wheel.y < 0) cam_zoom /= zoom_factor;
                if (cam_zoom < 0.1f) cam_zoom = 0.1f;
                if (cam_zoom > 10.0f) cam_zoom = 10.0f;
            } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (space_down && event.button.button == SDL_BUTTON_LEFT) {
                    rotating_camera = 1;
                    rotate_last_x = event.button.x;
                    rotate_last_y = event.button.y;
                }
            } else if (event.type == SDL_MOUSEBUTTONUP) {
                if (event.button.button == SDL_BUTTON_LEFT) rotating_camera = 0;
            } else if (event.type == SDL_MOUSEMOTION) {
                if (rotating_camera) {
                    int dx = event.motion.x - rotate_last_x;
                    int dy = event.motion.y - rotate_last_y;
                    // Horizontal mouse movement rotates yaw (left/right)
                    cam_yaw += (float)dx * 0.005f;
                    // Vertical mouse movement rotates pitch (up/down)
                    cam_pitch += (float)dy * 0.005f;
                    // Clamp pitch to avoid gimbal lock
                    if (cam_pitch > 1.5f) cam_pitch = 1.5f;
                    if (cam_pitch < -1.5f) cam_pitch = -1.5f;
                    rotate_last_x = event.motion.x;
                    rotate_last_y = event.motion.y;
                }
            }
        }

        // update window size in case of resize
        SDL_GetWindowSize(window, &win_w, &win_h);

        // Update camera position based on WASD keys, SHIFT for up, CTRL for down
        float dt_cam = sim->dt; // use simulation timestep for camera movement
        float move_speed = cam_speed * cam_zoom * dt_cam;
        if (key_w || key_a || key_s || key_d || key_shift || key_ctrl) {
            // Compute forward and right vectors from yaw and pitch (full 3D camera movement)
            // Forward is the direction the camera is looking (including vertical component)
            float forward_x = -sinf(cam_yaw) * cosf(cam_pitch);
            float forward_y = sinf(cam_pitch);
            float forward_z = -cosf(cam_yaw) * cosf(cam_pitch);
            // Right is perpendicular to forward in the horizontal plane (ignores pitch)
            float right_x = cosf(cam_yaw);
            float right_z = -sinf(cam_yaw);
            
            if (key_w) { cam_x += forward_x * move_speed; cam_y += forward_y * move_speed; cam_z += forward_z * move_speed; }
            if (key_s) { cam_x -= forward_x * move_speed; cam_y -= forward_y * move_speed; cam_z -= forward_z * move_speed; }
            if (key_a) { cam_x -= right_x * move_speed; cam_z -= right_z * move_speed; }
            if (key_d) { cam_x += right_x * move_speed; cam_z += right_z * move_speed; }
            if (key_shift) { cam_y += move_speed; } // Move up
            if (key_ctrl) { cam_y -= move_speed; }  // Move down
        }

        // Rendering with 3D perspective
        glViewport(0,0,win_w,win_h);
        glClearColor(0.6f, 0.7f, 0.9f, 1.0f);  // Sky blue background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);  // Don't cull backfaces so triangles are visible from both sides

        // Render simulator with 3D perspective projection
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        // Perspective projection: FOV=60 degrees, aspect ratio, near=10, far=10000
        float aspect = (float)win_w / (float)win_h;
        float fov_y = 60.0f;
        float near_plane = 10.0f;
        float far_plane = 10000.0f;
        // gluPerspective(fov_y, aspect, near_plane, far_plane);
        // Manual perspective calculation (equivalent to gluPerspective)
        float f = 1.0f / tanf(fov_y * 3.14159265f / 360.0f);
        glFrustum(-near_plane * aspect / f, near_plane * aspect / f, 
                  -near_plane / f, near_plane / f, near_plane, far_plane);
        
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        // Apply camera transformation: rotate then translate
        glRotatef(-cam_pitch * 180.0f / 3.14159265f, 1.0f, 0.0f, 0.0f); // pitch around X
        glRotatef(-cam_yaw * 180.0f / 3.14159265f, 0.0f, 1.0f, 0.0f);   // yaw around Y
        glTranslatef(-cam_x, -cam_y, -cam_z); // translate to camera position

        // Apply force-drag: when the user is holding the left mouse button and the
        // Selection menu's Drag toggle is enabled, apply a proportional force to
        // each selected node toward the current pick position. This does not
        // require the click to have started on the node and remains active until
        // the user disables the toggle.
    float K_p = edata.drag_strength ? (float)(*(edata.drag_strength)) : 3.0f;  // proportional (stiffness)
    float K_d = 12.0f;  // derivative (damping) - tune this ratio as needed
    
    const float MAX_FORCE = 1e3f;
    if (mouse_left_down && !mouse_left_down_on_ui && edata.drag_enabled && edata.drag_enabled[0] && edata.drag_strength && sel_filter == SEL_NODE && dynarray_size(selection) > 0) {
            float K = (float)(*(edata.drag_strength));
            for (size_t si = 0; si < dynarray_size(selection); ++si) {
                Node *n = (Node*)dynarray_get(selection, si);
                if (!n) continue;
                // Error vector: direction and distance to target (3D)
                float rx = pick_wx - n->pos[0];
                float ry = pick_wy - n->pos[1];
                float rz = pick_wz - n->pos[2];
                float dist2 = rx*rx + ry*ry + rz*rz;
                if (dist2 < 1e-8f) continue;
                
                float dist = sqrtf(dist2);
                float ux = rx / dist;
                float uy = ry / dist;
                float uz = rz / dist;
                
                // Proportional term: spring-like force based on distance
                float force_p = K_p * dist;
                // Derivative term: damping based on velocity toward target
                // Project velocity onto the direction toward target
                float vel_dot_u = n->vel[0] * ux + n->vel[1] * uy + n->vel[2] * uz;
                float force_d = K_d * vel_dot_u;  // opposes motion toward/away from target
                
                // Combined force magnitude
                float force_mag = force_p - force_d;  // subtract damping to resist motion
                
                // Clamp to max force
                if (force_mag > MAX_FORCE) force_mag = MAX_FORCE;
                if (force_mag < -MAX_FORCE) force_mag = -MAX_FORCE;
                
                // Apply force in target direction
                float fx = force_mag * ux;
                float fy = force_mag * uy;
                float fz = force_mag * uz;
                
                float mass = fmaxf(n->mass, 1e-6f);
                float dvx = fx * sim->dt;
                float dvy = fy * sim->dt;
                float dvz = fz * sim->dt;

                n->vel[0] *= 0.99f;
                n->vel[1] *= 0.99f;
                n->vel[2] *= 0.99f;

                n->vel[0] += dvx;
                n->vel[1] += dvy;
                n->vel[2] += dvz;
            }
        }
        if (!paused) {
            simulator_step(sim);
        }

        // Draw 3D reference grid at z=0 plane (XY plane)
        if (show_grid) {
            float cell_size = (float)grid_cell_size;
            int grid_extent = 20; // draw ±20 cells from origin
            glColor3f(0.7f, 0.7f, 0.75f);
            glLineWidth(1.0f);
            glBegin(GL_LINES);
            for (int ix = -grid_extent; ix <= grid_extent; ++ix) {
                float x = (float)ix * cell_size;
                glVertex3f(x, -grid_extent * cell_size, 0.0f);
                glVertex3f(x, grid_extent * cell_size, 0.0f);
            }
            for (int iy = -grid_extent; iy <= grid_extent; ++iy) {
                float y = (float)iy * cell_size;
                glVertex3f(-grid_extent * cell_size, y, 0.0f);
                glVertex3f(grid_extent * cell_size, y, 0.0f);
            }
            glEnd();
            // Draw axes
            glLineWidth(3.0f);
            glBegin(GL_LINES);
            glColor3f(1.0f, 0.0f, 0.0f); glVertex3f(0,0,0); glVertex3f(200,0,0); // X red
            glColor3f(0.0f, 1.0f, 0.0f); glVertex3f(0,0,0); glVertex3f(0,200,0); // Y green
            glColor3f(0.0f, 0.0f, 1.0f); glVertex3f(0,0,0); glVertex3f(0,0,200); // Z blue
            glEnd();
            glLineWidth(1.0f);
        }

        // Draw selection/placement plane grid (translucent) after reference grid
        if (current_tool == TOOL_SELECT || current_tool == TOOL_ADD_NODE || 
            current_tool == TOOL_ADD_DIST || current_tool == TOOL_ADD_SPRING || current_tool == TOOL_ADD_WALL) {
            render_plane_grid(current_plane, plane_offsets[current_plane], cam_yaw, cam_pitch,
                             cam_x, cam_y, cam_z, 400.0f, (float)grid_cell_size);
        }

        simulator_draw(sim, cam_yaw, cam_pitch);
        // draw selection highlights (in world coordinates, 3D)
        if (dynarray_size(selection) > 0) {
            glColor3f(0.0f, 1.0f, 0.0f);
            for (size_t si = 0; si < dynarray_size(selection); ++si) {
                Node *n = (Node*)dynarray_get(selection, si);
                if (!n) continue;
                float r = n->radius + 2.0f;
                glBegin(GL_LINE_LOOP);
                for (int k = 0; k < 20; ++k) {
                    float theta = 2.0f * 3.14159265f * (float)k / 20.0f;
                    glVertex3f(n->pos[0] + r * cosf(theta), n->pos[1] + r * sinf(theta), n->pos[2]);
                }
                glEnd();
            }
        }
        // draw pending-tool-node marker (magenta) if waiting for second endpoint
        if (pending_tool_node != NULL) {
            glColor3f(1.0f, 0.0f, 1.0f);
            float r = pending_tool_node->radius + 3.0f;
            glBegin(GL_LINE_LOOP);
            for (int k = 0; k < 20; ++k) {
                float theta = 2.0f * 3.14159265f * (float)k / 20.0f;
                glVertex2f(pending_tool_node->pos[0] + r * cosf(theta), pending_tool_node->pos[1] + r * sinf(theta));
            }
            glEnd();
        }
        
        // Draw pale blue camera-facing circle at pick position (drag target indicator)
        if (pick_active && (current_tool == TOOL_SELECT || current_tool == TOOL_ADD_NODE ||
                           current_tool == TOOL_ADD_DIST || current_tool == TOOL_ADD_SPRING ||
                           current_tool == TOOL_ADD_WALL)) {
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glColor4f(0.5f, 0.7f, 1.0f, 0.6f); // pale blue, translucent
            
            float pick_radius = 8.0f;
            
            // Create camera-facing billboard at pick position
            // Get camera right and up vectors
            float right_x = cosf(cam_yaw);
            float right_y = 0.0f;
            float right_z = -sinf(cam_yaw);
            
            float up_x = sinf(cam_yaw) * sinf(cam_pitch);
            float up_y = cosf(cam_pitch);
            float up_z = cosf(cam_yaw) * sinf(cam_pitch);
            
            glBegin(GL_TRIANGLE_FAN);
            glVertex3f(pick_wx, pick_wy, pick_wz); // center
            for (int k = 0; k <= 20; ++k) {
                float theta = 2.0f * 3.14159265f * (float)k / 20.0f;
                float offset_x = pick_radius * (cosf(theta) * right_x + sinf(theta) * up_x);
                float offset_y = pick_radius * (cosf(theta) * right_y + sinf(theta) * up_y);
                float offset_z = pick_radius * (cosf(theta) * right_z + sinf(theta) * up_z);
                glVertex3f(pick_wx + offset_x, pick_wy + offset_y, pick_wz + offset_z);
            }
            glEnd();
            glDisable(GL_BLEND);
        }
        
        glPopMatrix();

        // restore projection/modelview
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);

        // Disable depth test and set up 2D orthographic projection for UI
        glDisable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, win_w, win_h, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        // Render menu UI (draw all registered menus)
        for (size_t mi = 0; mi < menus->size; ++mi) {
            Menu *m = (Menu*)menus->items[mi];
            if (m) menu_render(m, win_w, win_h);
        }
        
        // draw rectangle selection overlay in screen space if active
        if (rect_select_active) {
            glColor4f(0.0f, 1.0f, 0.0f, 0.25f);
            glBegin(GL_LINE_LOOP);
            glVertex2i(rect_x0, rect_y0);
            glVertex2i(rect_x1, rect_y0);
            glVertex2i(rect_x1, rect_y1);
            glVertex2i(rect_x0, rect_y1);
            glEnd();
        }

        // Update FPS counter
        fps_frames += 1;
        Uint32 fps_now = SDL_GetTicks();
        Uint32 fps_dt = fps_now - fps_last_time;
        if (fps_dt >= 250) { // update every 250 ms for responsiveness
            fps_value = (int)((fps_frames * 1000) / (fps_dt));
            fps_frames = 0;
            fps_last_time = fps_now;
        }

        // Draw FPS in bottom-right with white color
        char fps_text[64];
        snprintf(fps_text, sizeof(fps_text), "FPS: %d", fps_value > 0 ? fps_value : 0);
        int tw = 0, th = 0;
        if (menu_measure_text(fps_text, &tw, &th) == 0) {
            int pad = 8;
            int tx = win_w - pad - tw;
            int ty = win_h - pad - th;
            Color white = {255,255,255,255};
            menu_draw_text_at(fps_text, tx, ty, white);
        }
        
        // Draw plane info in top-left when in placement/selection modes
        if (current_tool == TOOL_SELECT || current_tool == TOOL_ADD_NODE || 
            current_tool == TOOL_ADD_DIST || current_tool == TOOL_ADD_SPRING || current_tool == TOOL_ADD_WALL) {
            const char* plane_names[] = {"World X", "World Y", "World Z", "Camera X", "Camera Y", "Camera Z"};
            char plane_text[128];
            snprintf(plane_text, sizeof(plane_text), "Plane: %s (offset: %.1f)\nUp/Down: change plane | Left/Right: adjust offset", 
                    plane_names[current_plane], plane_offsets[current_plane]);
            Color cyan = {100,200,255,255};
            menu_draw_text_at(plane_text, 8, 8, cyan);
        }
        
        // Pop 2D UI matrices
        glPopMatrix(); // modelview
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);

        SDL_GL_SwapWindow(window);
    }

    // Free input handlers
    for (int i = 0; i < input_handler_count; ++i) {
        free(input_handlers[i]);
    }

    // Free simulator
    simulator_free(sim);

    // Free menus
    for (size_t mi = 0; mi < menus->size; ++mi) {
        Menu *m = (Menu*)menus->items[mi];
        if (m) menu_free(m);
    }
    dynarray_free(menus, NULL);

    // // Free menu and callback data
    // menu_free(test);
    // free(d_x);
    // free(d_y);
    // free(d_w);
    // free(d_h);
    // free(d_col);
    // cleanup font
    menu_clear_font();

    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}

// Callback to update simulator dt when slider changes (top-level)
static void sim_on_step_change(VariableInteraction *vi, void *user_data) {
    Simulator *s = (Simulator*)user_data;
    if (!s) return;
    double v = *(double*)vi->variable;
    // Variable represents steps-per-second (SPS). Convert to timestep dt = 1.0 / SPS
    if (v > 0.0) s->dt = 1.0f / (float)v;
}

// Delete selected nodes and all incident constraints/walls
static void sel_cb_delete_nodes(VariableInteraction *vi, void *user_data) {
    (void)vi;
    EditData *ed = (EditData*)user_data;
    if (!ed || !ed->selection || !ed->sim) return;
    Simulator *sim = ed->sim;
    size_t sel_n = dynarray_size(ed->selection);
    if (sel_n == 0) return;
    // collect nodes to delete
    Node **to_del = (Node**)malloc(sizeof(Node*) * sel_n);
    size_t td = 0;
    for (size_t i = 0; i < sel_n; ++i) {
        Node *n = (Node*)dynarray_get(ed->selection, i);
        if (n) to_del[td++] = n;
    }

    // For each node, remove incident constraints and walls
    for (size_t d = 0; d < td; ++d) {
        Node *n = to_del[d];
        if (!n) continue;
        // remove constraints referencing this node
        DynArray *cons = sim->constraints;
        for (ssize_t ci = (ssize_t)dynarray_size(cons) - 1; ci >= 0; --ci) {
            Constraint *c = (Constraint*)dynarray_get(cons, (size_t)ci);
            if (!c) continue;
            if (c->node == n || c->other == n) {
                // remove from other endpoint's node->constraints
                Node *other = (c->node == n) ? c->other : c->node;
                if (other && other->constraints) {
                    DynArray *nc = other->constraints;
                    for (size_t j = 0; j < nc->size; ++j) {
                        if ((Constraint*)dynarray_get(nc, j) == c) {
                            for (size_t k = j; k + 1 < nc->size; ++k) nc->items[k] = nc->items[k+1];
                            nc->size -= 1; break;
                        }
                    }
                }
                // remove from simulator constraints array
                for (size_t j = (size_t)ci; j + 1 < cons->size; ++j) cons->items[j] = cons->items[j+1];
                cons->size -= 1;
                free(c);
            }
        }
        // remove walls referencing this node
        DynArray *ws = sim->walls;
        for (ssize_t wi = (ssize_t)dynarray_size(ws) - 1; wi >= 0; --wi) {
            TriangleWall *w = (TriangleWall*)dynarray_get(ws, (size_t)wi);
            if (!w) continue;
            if (w->A == n || w->B == n || w->C == n) {
                trianglewall_free(w);
                for (size_t j = (size_t)wi; j + 1 < ws->size; ++j) ws->items[j] = ws->items[j+1];
                ws->size -= 1;
            }
        }
        // remove node from sim->nodes and free
        DynArray *na = sim->nodes;
        for (size_t ni = 0; ni < na->size; ++ni) {
            if ((Node*)dynarray_get(na, ni) == n) {
                node_free(n);
                for (size_t j = ni; j + 1 < na->size; ++j) na->items[j] = na->items[j+1];
                na->size -= 1;
                break;
            }
        }
    }

    // reindex remaining nodes
    for (size_t i = 0; i < dynarray_size(sim->nodes); ++i) {
        Node *n = (Node*)dynarray_get(sim->nodes, i);
        if (n) n->idx = (int)i;
    }

    free(to_del);
    // clear selection and refresh menu
    ed->selection->size = 0;
    destroy_select_menu_if_present(ed);
    create_select_menu_if_needed(ed);
}

static void sel_cb_set_node_friction(VariableInteraction *vi, void *user_data) {
    EditData *ed = (EditData*)user_data;
    if (!ed || !ed->selection) return;
    double v = *(double*)vi->variable;
    for (size_t i = 0; i < dynarray_size(ed->selection); ++i) {
        Node *n = (Node*)dynarray_get(ed->selection, i);
        if (!n) continue;
        n->friction = (float)v;
    }
}

static void sel_cb_set_constraint_distance(VariableInteraction *vi, void *user_data) {
    EditData *ed = (EditData*)user_data;
    if (!ed || !ed->selection) return;
    double v = *(double*)vi->variable;
    for (size_t i = 0; i < dynarray_size(ed->selection); ++i) {
        Constraint *c = (Constraint*)dynarray_get(ed->selection, i);
        if (!c) continue;
        c->rest_length = (float)v;
    }
}

static void sel_cb_set_spring_prop(VariableInteraction *vi, void *user_data) {
    EditData *ed = (EditData*)user_data;
    if (!ed || !ed->selection) return;
    const char *name = vi->name;
    double v = *(double*)vi->variable;
    for (size_t i = 0; i < dynarray_size(ed->selection); ++i) {
        Constraint *c = (Constraint*)dynarray_get(ed->selection, i);
        if (!c) continue;
        if (c->type != CT_SPRING) continue;
        if (strcmp(name, "Stiffness") == 0) {
            c->stiffness = (float)v;
        } else if (strcmp(name, "Rest Length") == 0) {
            c->rest_length = (float)v;
        }
    }
}

static void sel_cb_set_wall_prop(VariableInteraction *vi, void *user_data) {
    EditData *ed = (EditData*)user_data;
    if (!ed || !ed->selection) return;
    const char *name = vi->name;
    double v = *(double*)vi->variable;
    for (size_t i = 0; i < dynarray_size(ed->selection); ++i) {
        TriangleWall *w = (TriangleWall*)dynarray_get(ed->selection, i);
        if (!w) continue;
        if (strcmp(name, "Friction") == 0) w->friction = (float)v;
        else if (strcmp(name, "Restitution") == 0) w->restitution = (float)v;
    }
}