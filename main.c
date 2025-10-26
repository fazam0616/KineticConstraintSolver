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
    // +Node tool spawn parameters (allocated when Edit menu row is created)
    double *node_mass;
    double *node_friction;
    int *node_anchored;
    int *node_collide_with_walls;
    // selection & simulator hooks (set by caller)
    Simulator *sim;
    DynArray *selection;
    int *drag_enabled; // pointer to drag enable flag (allocated by menu)
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

static void create_select_menu_if_needed(EditData *ed) {
    if (!ed || !ed->menus) return;
    if (*(ed->select_menu) != NULL) return;
    int w = 300, h = 160;
    int x = 10;
    int y = (*(ed->win_h) > h + 20) ? (*(ed->win_h) - h - 10) : 400;
    Menu *m = menu_create(x, y, w, h, 120, "Selection", (Color){255,255,255,255}, (Color){40,40,70,255});

    // Row: Select Size
    MenuRow *r1 = menurow_create();
    double *sel_size = malloc(sizeof(double)); *sel_size = 10.0;
    VariableInteraction *v_sel_size = variableinteraction_create(sel_size, "Select Size", 1.0, 200.0, VAR_SLIDER, NULL, NULL);
    menurow_add_interaction(r1, v_sel_size);
    menu_add_row(m, r1);

    // Row: Anchor toggle
    MenuRow *r_anchor = menurow_create();
    int *sel_anchor = malloc(sizeof(int));
    // Default anchor toggle to reflect single-node selection state when possible
    *sel_anchor = 0;
    if (ed && ed->selection && dynarray_size(ed->selection) == 1) {
        Node *only = (Node*)dynarray_get(ed->selection, 0);
        if (only) *sel_anchor = only->anchored ? 1 : 0;
    }
    VariableInteraction *v_anchor = variableinteraction_create(sel_anchor, "Toggle Anchor", 0, 1, VAR_BOOL, sel_cb_toggle_anchor, ed);
    menurow_add_interaction(r_anchor, v_anchor);
    menu_add_row(m, r_anchor);

    // Row: Mass scale
    MenuRow *r_mass = menurow_create();
    double *mass_scale = malloc(sizeof(double)); *mass_scale = 1.0;
    VariableInteraction *v_mass = variableinteraction_create(mass_scale, "Scale Mass", 0.1, 10.0, VAR_SLIDER, sel_cb_scale_mass, ed);
    menurow_add_interaction(r_mass, v_mass);
    menu_add_row(m, r_mass);

    // Row: Move X / Move Y / Rotate
    MenuRow *r_move = menurow_create();
    double *move_x = malloc(sizeof(double)); *move_x = 0.0;
    double *move_y = malloc(sizeof(double)); *move_y = 0.0;
    double *rotate = malloc(sizeof(double)); *rotate = 0.0;
    VariableInteraction *v_move_x = variableinteraction_create(move_x, "Move X", -500.0, 500.0, VAR_SLIDER, sel_cb_move_or_rotate, ed);
    VariableInteraction *v_move_y = variableinteraction_create(move_y, "Move Y", -500.0, 500.0, VAR_SLIDER, sel_cb_move_or_rotate, ed);
    VariableInteraction *v_rotate = variableinteraction_create(rotate, "Rotate (deg)", -180.0, 180.0, VAR_SLIDER, sel_cb_move_or_rotate, ed);
    menurow_add_interaction(r_move, v_move_x); menurow_add_interaction(r_move, v_move_y); menurow_add_interaction(r_move, v_rotate);
    menu_add_row(m, r_move);

    // Row: Drag toggle
    MenuRow *r_drag = menurow_create();
    int *drag_enable = malloc(sizeof(int)); *drag_enable = 0;
    VariableInteraction *v_drag = variableinteraction_create(drag_enable, "Drag", 0, 1, VAR_BOOL, sel_cb_drag_toggle, ed);
    menurow_add_interaction(r_drag, v_drag);
    menu_add_row(m, r_drag);

    // store pointers in EditData so callbacks can access
    if (ed) { ed->selection = ed->selection ? ed->selection : NULL; ed->drag_enabled = drag_enable; }
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

        VariableInteraction *vi_mass = variableinteraction_create(ed->node_mass, "Mass", 0.01, 1000.0, VAR_SLIDER, NULL, NULL);
        VariableInteraction *vi_friction = variableinteraction_create(ed->node_friction, "Friction", 0.0, 1.0, VAR_SLIDER, NULL, NULL);
        VariableInteraction *vi_anchor = variableinteraction_create(ed->node_anchored, "Anchored", 0, 1, VAR_BOOL, NULL, NULL);
        VariableInteraction *vi_collide = variableinteraction_create(ed->node_collide_with_walls, "Collide Walls", 0, 1, VAR_BOOL, NULL, NULL);
        menurow_add_interaction(nr, vi_mass);
        menurow_add_interaction(nr, vi_friction);
        menurow_add_interaction(nr, vi_anchor);
        menurow_add_interaction(nr, vi_collide);
        menu_add_row(m, nr);
    } else {
        double dval = 0.0;
        const char *label = "(no tool)";
        if (*(ed->current_tool) == TOOL_SELECT) label = "Select Mode";
        else if (*(ed->current_tool) == TOOL_ADD_DIST) label = "Dist Options";
        else if (*(ed->current_tool) == TOOL_ADD_SPRING) label = "Spring Options";
        else if (*(ed->current_tool) == TOOL_ADD_WALL) label = "Wall Options";
        VariableInteraction *vi = variableinteraction_create(&dval, label, 0.0, 1.0, VAR_SLIDER, NULL, NULL);
        menurow_add_interaction(nr, vi);
        menu_add_row(m, nr);
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
    if (!ed || !ed->selection) return;
    int v = *(int*)vi->variable;
    // if setting anchor on selected nodes, create AnchorConstraint for each unanchored node
    if (v) {
        for (size_t i = 0; i < dynarray_size(ed->selection); ++i) {
            Node *n = (Node*)dynarray_get(ed->selection, i);
            if (!n) continue;
            if (!n->anchored) {
                Constraint *ac = anchorconstraint_create(n, n->pos[0], n->pos[1]);
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
    EditData *ed = (EditData*)user_data;
    if (!ed || !ed->selection) return;
    double factor = *(double*)vi->variable;
    if (factor == 0.0) return;
    for (size_t i = 0; i < dynarray_size(ed->selection); ++i) {
        Node *n = (Node*)dynarray_get(ed->selection, i);
        if (!n) continue;
        n->mass *= (float)factor;
    }
    // reset slider to 1.0 so future adjustments are relative
    *(double*)vi->variable = 1.0;
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

    // // Variables to drive sliders (double) and toggle (int)
    // double var_x = test->x;
    // double var_y = test->y;
    // double var_w = test->width;
    // double var_h = test->height;
    // int var_color = 0;

    


    // // Create rows and interactions
    // MenuRow *row1 = menurow_create();
    // MenuCallbackData *d_x = malloc(sizeof(MenuCallbackData)); d_x->menu = test; d_x->field = F_X;
    // MenuCallbackData *d_y = malloc(sizeof(MenuCallbackData)); d_y->menu = test; d_y->field = F_Y;
    // VariableInteraction *vi_x = variableinteraction_create(&var_x, "X", 0, 700, VAR_SLIDER, on_slider_change, d_x);
    // VariableInteraction *vi_y = variableinteraction_create(&var_y, "Y", 0, 500, VAR_SLIDER, on_slider_change, d_y);
    // menurow_add_interaction(row1, vi_x);
    // menurow_add_interaction(row1, vi_y);
    // menu_add_row(test, row1);

    // MenuRow *row2 = menurow_create();
    // MenuCallbackData *d_w = malloc(sizeof(MenuCallbackData)); d_w->menu = test; d_w->field = F_W;
    // MenuCallbackData *d_h = malloc(sizeof(MenuCallbackData)); d_h->menu = test; d_h->field = F_H;
    // VariableInteraction *vi_w = variableinteraction_create(&var_w, "W", 50, 1000, VAR_SLIDER, on_slider_change, d_w);
    // VariableInteraction *vi_h = variableinteraction_create(&var_h, "H", 20, 800, VAR_SLIDER, on_slider_change, d_h);
    // menurow_add_interaction(row2, vi_w);
    // menurow_add_interaction(row2, vi_h);
    // menu_add_row(test, row2);

    // MenuRow *row3 = menurow_create();
    // MenuCallbackData *d_col = malloc(sizeof(MenuCallbackData)); d_col->menu = test; d_col->field = F_COLOR;
    // VariableInteraction *vi_col = variableinteraction_create(&var_color, "ToggleColor", 0, 1, VAR_BOOL, on_bool_change, d_col);
    // menurow_add_interaction(row3, vi_col);
    // menu_add_row(test, row3);

    int running = 1;
    SDL_Event event;
    int win_w = 800, win_h = 600;

    // FPS tracking
    Uint32 fps_last_time = SDL_GetTicks();
    int fps_frames = 0;
    int fps_value = 0;

    // Camera for pan/zoom
    float cam_x = 0.0f, cam_y = 0.0f;
    float cam_scale = 1.0f;
    int space_down = 0;
    int panning = 0;
    int pan_last_x = 0, pan_last_y = 0;

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
    DynArray *selection = dynarray_create(8); // holds Node* / Constraint* / WallSegment*
    enum { SEL_NONE=0, SEL_NODE=1, SEL_CONSTRAINT=2, SEL_WALL=3 } sel_filter = SEL_NODE;
    int selection_dragging = 0; // dragging selected objects
    int rect_select_active = 0; // right-button rectangle select active
    int rect_x0 = 0, rect_y0 = 0, rect_x1 = 0, rect_y1 = 0;
    // Last mouse position used for selection dragging (world-space). Using world coords
    // avoids errors if cam_scale or cam_x/cam_y change during a drag.
    float sel_last_wx = 0.0f, sel_last_wy = 0.0f;
    // last pick position (world coords) used for rendering the pick cursor
    float pick_wx = 0.0f, pick_wy = 0.0f;
    int pick_active = 0;
    EditData edata;
    edata.menus = menus; edata.edit_menu = &edit_menu; edata.select_menu = &select_menu; edata.win_w = &win_w; edata.win_h = &win_h;
    edata.tool_select = &tool_select; edata.tool_node = &tool_node; edata.tool_dist = &tool_dist; edata.tool_spring = &tool_spring; edata.tool_wall = &tool_wall; edata.current_tool = &current_tool;
    edata.sim = sim; edata.selection = selection; edata.drag_enabled = NULL;
    edata.node_mass = NULL; edata.node_friction = NULL; edata.node_anchored = NULL; edata.node_collide_with_walls = NULL;
    // attach user_data for vi_edit now that edata is set
    vi_edit->callback_data = &edata;

    int n = 6;

    // pending node for pair-based tool actions (dist/spring/wall)
    Node *pending_tool_node = NULL;

    // Optional: run octagon mesh generation test
    if (run_mesh_test) {
        DynArray *poly = dynarray_create(8);
        const float cx = 0.0f, cy = 220.0f, R = 100.0f;
        for (int i = 0; i < n; ++i) {
            float a = (float)i * 2.0f * 3.14159265f / (float)n;
            Node *pn = node_create(-1, 1.0f, cx + R * cosf(a), cy + R * sinf(a));
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
    const float offset = 50.0f;
    const float box_size = 200.0f;
    const float half = box_size * 0.5f;

    // We'll keep an array of node pointers so we can reference by index like the python snippet
    Node *nodes_arr[16];
    int ni = 0;

    // Box corners (friction = 0)
    nodes_arr[ni] = node_create(-1, 1.0f, -half + offset, -half); nodes_arr[ni]->friction = 0.0f; simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f,  half + offset, -half); nodes_arr[ni]->friction = 0.0f; simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f,  half + offset,  half); nodes_arr[ni]->friction = 0.0f; simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f, -half + offset,  half); nodes_arr[ni]->friction = 0.0f; simulator_add_node(sim, nodes_arr[ni++]);

    // Sleeve top and bottom panels (anchored walls)
    float sleeve_y = half + 5;
    float sleeve_length = box_size + 80.0f;
    float sleeve_left = -sleeve_length * 0.5f;
    float sleeve_right =  sleeve_length * 0.5f;

    nodes_arr[ni] = node_create(-1, 1.0f, sleeve_left + offset,  sleeve_y); simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f, sleeve_right + offset, sleeve_y); simulator_add_node(sim, nodes_arr[ni++]);
    WallSegment *wtop = wallsegment_create(nodes_arr[4], nodes_arr[5], 1.0f, 0.0f); simulator_add_wall(sim, wtop);

    nodes_arr[ni] = node_create(-1, 1.0f, sleeve_left + offset, -sleeve_y); simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f, sleeve_right + offset,-sleeve_y); simulator_add_node(sim, nodes_arr[ni++]);
    WallSegment *wbot = wallsegment_create(nodes_arr[6], nodes_arr[7], 1.0f, 0.0f); simulator_add_wall(sim, wbot);

    // Additional support / spacer nodes
    nodes_arr[ni] = node_create(-1, 1.0f, -box_size + offset, 0.0f); simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f, -box_size - half,    0.0f); simulator_add_node(sim, nodes_arr[ni++]);
    // a heavier moving node
    nodes_arr[ni] = node_create(-1, 50.0f, -box_size - half, 30.0f); simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f,  half * 1.25f + offset, 0.0f); simulator_add_node(sim, nodes_arr[ni++]);
    nodes_arr[ni] = node_create(-1, 1.0f,  box_size * 1.5f + offset,  0.0f); simulator_add_node(sim, nodes_arr[ni++]);
    // give node index 10 an initial leftward velocity
    if (ni > 10) {
        nodes_arr[10]->vel[0] = -25.0f; nodes_arr[10]->vel[1] = 0.0f;
    }

    // Anchor constraints (anchor at current node position)
    // panels top/bot and two support nodes
    Constraint *ac;
    ac = anchorconstraint_create(nodes_arr[4], nodes_arr[4]->pos[0], nodes_arr[4]->pos[1]); simulator_add_constraint(sim, ac);
    ac = anchorconstraint_create(nodes_arr[5], nodes_arr[5]->pos[0], nodes_arr[5]->pos[1]); simulator_add_constraint(sim, ac);
    ac = anchorconstraint_create(nodes_arr[6], nodes_arr[6]->pos[0], nodes_arr[6]->pos[1]); simulator_add_constraint(sim, ac);
    ac = anchorconstraint_create(nodes_arr[7], nodes_arr[7]->pos[0], nodes_arr[7]->pos[1]); simulator_add_constraint(sim, ac);
    ac = anchorconstraint_create(nodes_arr[9], nodes_arr[9]->pos[0], nodes_arr[9]->pos[1]); simulator_add_constraint(sim, ac);
    ac = anchorconstraint_create(nodes_arr[12], nodes_arr[12]->pos[0], nodes_arr[12]->pos[1]); simulator_add_constraint(sim, ac);

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
    WallSegment *w1 = wallsegment_create(nodes_arr[0], nodes_arr[1], 1.0f, 0.0f); simulator_add_wall(sim, w1);
    WallSegment *w2 = wallsegment_create(nodes_arr[2], nodes_arr[3], 1.0f, 0.0f); simulator_add_wall(sim, w2);

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
                int menu_handled = 0;
                // iterate menus in reverse order (higher z-index drawn last -> receive events first)
                for (int mi = (int)menus->size - 1; mi >= 0; --mi) {
                    Menu *m = (Menu*)menus->items[mi];
                    if (!m) continue;
                    if (menu_handle_mouse_button(m, button, state, mx, my)) { menu_handled = 1; break; }
                }

                // If not handled by UI menus, handle current tool actions (select, add-node, etc.)
                if (!menu_handled) {
                    // convert screen to world coords (account for Y-flip and camera)
                    // screen -> world: world = (screen / cam_scale) + cam_x
                    float wx = ((float)mx + cam_x) / cam_scale ;
                    float wy = ((float)(win_h - my) + cam_y) / cam_scale;
                    // store last pick position so we can render it on screen
                    pick_wx = wx; pick_wy = wy; pick_active = 1;

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
                            Node *nn = node_create(-1, (float)mass, wx, wy);
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
                        const float pick_world = pick_px / cam_scale; // convert pixel radius to world units

                        if (event.type == SDL_MOUSEBUTTONDOWN && button == SDL_BUTTON_LEFT) {
                            // find nearest node within pick radius
                            Node *found = NULL;
                            size_t n_nodes = dynarray_size(sim->nodes);
                            for (size_t ii = 0; ii < n_nodes; ++ii) {
                                Node *nn = (Node*)dynarray_get(sim->nodes, ii);
                                if (!nn) continue;
                                float dx = wx - nn->pos[0]; float dy = wy - nn->pos[1];
                                float d2 = dx*dx + dy*dy;
                                float r = nn->radius + pick_world;
                                if (d2 <= r*r) { found = nn; break; }
                            }
                            if (found) {
                                // toggle selection of this node; enforces node-only selection
                                sel_filter = SEL_NODE;
                                // check if already selected
                                int already = 0;
                                for (size_t si = 0; si < dynarray_size(selection); ++si) {
                                    if ((Node*)dynarray_get(selection, si) == found) { already = 1;
                                        for (size_t sj = si; sj + 1 < dynarray_size(selection); ++sj) selection->items[sj] = selection->items[sj+1];
                                        selection->size -= 1;
                                        break;
                                    }
                                }
                                if (!already) dynarray_append(selection, found);
                                // start dragging immediately if drag mode is enabled and the found node is selected
                                if (edata.drag_enabled && edata.drag_enabled[0]) {
                                    // Start dragging regardless of whether the node was already
                                    // selected; store the current mouse position in world
                                    // coordinates so subsequent motion uses consistent units.
                                    selection_dragging = 1;
                                    sel_last_wx = wx; sel_last_wy = wy;
                                }
                            }
                        } else if (event.type == SDL_MOUSEBUTTONDOWN && button == SDL_BUTTON_RIGHT) {
                            // start rectangle selection
                            rect_select_active = 1; rect_x0 = mx; rect_y0 = my; rect_x1 = mx; rect_y1 = my;
                            // also mark pick at start of rect (use same screen->world formula used above)
                            pick_wx = ((float)mx + cam_x) / cam_scale;
                            pick_wy = ((float)(win_h - my) + cam_y) / cam_scale;
                            pick_active = 1;
                        } else if (event.type == SDL_MOUSEBUTTONUP && button == SDL_BUTTON_LEFT) {
                            // stop any in-progress selection dragging
                            selection_dragging = 0;
                    // keep pick visible on release
                    pick_active = 1;
                        } else if (event.type == SDL_MOUSEBUTTONUP && button == SDL_BUTTON_RIGHT) {
                            if (rect_select_active) {
                                rect_select_active = 0;
                                // compute world-space rectangle
                                int rx0 = rect_x0 < rect_x1 ? rect_x0 : rect_x1;
                                int ry0 = rect_y0 < rect_y1 ? rect_y0 : rect_y1;
                                int rx1 = rect_x0 > rect_x1 ? rect_x0 : rect_x1;
                                int ry1 = rect_y0 > rect_y1 ? rect_y0 : rect_y1;
                                float wx0 = ((float)rx0 + cam_x) / cam_scale ;
                                float wy1 = ((float)(win_h - ry0) + cam_y) / cam_scale;
                                float wx1 = ((float)rx1 + cam_x) / cam_scale;
                                float wy0 = ((float)(win_h - ry1) + cam_y) / cam_scale;
                                // choose filter by finding first object inside rect: nodes -> constraints -> walls
                                int found_type = SEL_NONE;
                                // nodes
                                size_t n_nodes = dynarray_size(sim->nodes);
                                for (size_t ii = 0; ii < n_nodes; ++ii) {
                                    Node *nn = (Node*)dynarray_get(sim->nodes, ii);
                                    if (!nn) continue;
                                    if (nn->pos[0] >= wx0 && nn->pos[0] <= wx1 && nn->pos[1] >= wy0 && nn->pos[1] <= wy1) { found_type = SEL_NODE; break; }
                                }
                                if (found_type == SEL_NONE) found_type = SEL_NODE; // default
                                sel_filter = found_type;
                                // now select all of that type inside rect (for now only nodes)
                                if (sel_filter == SEL_NODE) {
                                    // clear existing selection
                                    selection->size = 0;
                                    for (size_t ii = 0; ii < dynarray_size(sim->nodes); ++ii) {
                                        Node *nn = (Node*)dynarray_get(sim->nodes, ii);
                                        if (!nn) continue;
                                        if (nn->pos[0] >= wx0 && nn->pos[0] <= wx1 && nn->pos[1] >= wy0 && nn->pos[1] <= wy1) dynarray_append(selection, nn);
                                    }
                                }
                                // finished rectangle selection; no dragging
                                selection_dragging = 0;
                                // keep pick visible at release
                                pick_active = 1;
                            }
                        }
                    }
                    else if (current_tool == TOOL_ADD_DIST || current_tool == TOOL_ADD_SPRING || current_tool == TOOL_ADD_WALL) {
                        const float pick_px = 8.0f;
                        const float pick_world = pick_px / cam_scale;
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
                                        // default stiffness and rest length = current distance
                                        float dx = pending_tool_node->pos[0] - found->pos[0];
                                        float dy = pending_tool_node->pos[1] - found->pos[1];
                                        float rest = sqrtf(dx*dx + dy*dy);
                                        Constraint *c = springconstraint_create(pending_tool_node, found, 10.0f, rest);
                                        if (c) simulator_add_constraint(sim, c);
                                    } else if (current_tool == TOOL_ADD_WALL) {
                                        WallSegment *w = wallsegment_create(pending_tool_node, found, 1.0f, 0.0f);
                                        if (w) simulator_add_wall(sim, w);
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
                if (!menu_handled && (*(edata.current_tool) == TOOL_SELECT)) {
                    // keep consistent with the screen->world conversion used above
                    pick_wx = ((float)mx + cam_x) / cam_scale;
                    pick_wy = ((float)(win_h - my) + cam_y) / cam_scale;
                    pick_active = 1;
                }
                // handle dragging of selected nodes. Compute current mouse world
                // coordinates and apply the world-space delta relative to the
                // last stored world coords. This is robust if cam_scale or
                // camera pan change during the drag.
                if (!menu_handled && selection_dragging && edata.drag_enabled && edata.drag_enabled[0]) {
                    // convert current pixel to world using same formula as picking
                    float cur_wx = ((float)mx + cam_x) / cam_scale;
                    float cur_wy = ((float)(win_h - my) + cam_y) / cam_scale;
                    float dx = cur_wx - sel_last_wx;
                    float dy = cur_wy - sel_last_wy;
                    if (dx != 0.0f || dy != 0.0f) {
                        for (size_t si = 0; si < dynarray_size(selection); ++si) {
                            Node *n = (Node*)dynarray_get(selection, si);
                            if (!n) continue;
                            n->pos[0] += dx; n->pos[1] += dy;
                        }
                        sel_last_wx = cur_wx; sel_last_wy = cur_wy;
                    }
                }
            }

            // Camera controls: space + drag to pan; mouse wheel to zoom
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_SPACE) space_down = 1;
            } else if (event.type == SDL_KEYUP) {
                if (event.key.keysym.sym == SDLK_SPACE) space_down = 0;
            } else if (event.type == SDL_MOUSEWHEEL) {
                // zoom about screen origin; scale factor step
                if (event.wheel.y > 0) cam_scale *= 1.1f; else if (event.wheel.y < 0) cam_scale *= 0.9f;
                if (cam_scale < 0.05f) cam_scale = 0.05f;
                if (cam_scale > 20.0f) cam_scale = 20.0f;
            } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (space_down && event.button.button == SDL_BUTTON_LEFT) {
                    panning = 1;
                    pan_last_x = event.button.x;
                    pan_last_y = event.button.y;
                }
            } else if (event.type == SDL_MOUSEBUTTONUP) {
                if (event.button.button == SDL_BUTTON_LEFT) panning = 0;
            } else if (event.type == SDL_MOUSEMOTION) {
                if (panning) {
                    int dx = event.motion.x - pan_last_x;
                    int dy = event.motion.y - pan_last_y;
                    // convert screen delta to world delta (account for scale and Y-flip)
                    float world_dx = (float)dx;
                    float world_dy = -(float)dy;
                    cam_x -= world_dx;
                    cam_y -= world_dy;
                    pan_last_x = event.motion.x;
                    pan_last_y = event.motion.y;
                }
            }
        }

        // update window size in case of resize
        SDL_GetWindowSize(window, &win_w, &win_h);

        // Rendering (clear only)
        glViewport(0,0,win_w,win_h);
        glClearColor(0.2f, 0.3f, 0.6f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Render simulator (use pixel orthographic projection)
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, win_w, win_h, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        if (!paused) simulator_step(sim);
        // Draw simulator with a Y-flip so world +Y (up) maps to screen Y downwards
        glPushMatrix();
        glTranslatef(0.0f, (float)win_h, 0.0f);
        glScalef(1.0f, -1.0f, 1.0f);
        // apply camera pan & zoom (in world coordinates)
        glTranslatef(-cam_x, -cam_y, 0.0f);
        glScalef(cam_scale, cam_scale, 1.0f);
        simulator_draw(sim);
        // draw selection highlights (in world coordinates)
        if (dynarray_size(selection) > 0) {
            glColor3f(0.0f, 1.0f, 0.0f);
            for (size_t si = 0; si < dynarray_size(selection); ++si) {
                Node *n = (Node*)dynarray_get(selection, si);
                if (!n) continue;
                float r = n->radius + 2.0f;
                glBegin(GL_LINE_LOOP);
                for (int k = 0; k < 20; ++k) {
                    float theta = 2.0f * 3.14159265f * (float)k / 20.0f;
                    glVertex2f(n->pos[0] + r * cosf(theta), n->pos[1] + r * sinf(theta));
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
        glPopMatrix();

        // restore projection/modelview
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);

        // Render menu UI (draw all registered menus)
        for (size_t mi = 0; mi < menus->size; ++mi) {
            Menu *m = (Menu*)menus->items[mi];
            if (m) menu_render(m, win_w, win_h);
        }
        // draw rectangle selection overlay in screen space if active
        if (rect_select_active) {
            glMatrixMode(GL_PROJECTION);
            glPushMatrix(); glLoadIdentity(); glOrtho(0, win_w, win_h, 0, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix(); glLoadIdentity();
            glColor4f(0.0f, 1.0f, 0.0f, 0.25f);
            glBegin(GL_LINE_LOOP);
            glVertex2i(rect_x0, rect_y0);
            glVertex2i(rect_x1, rect_y0);
            glVertex2i(rect_x1, rect_y1);
            glVertex2i(rect_x0, rect_y1);
            glEnd();
            glPopMatrix();
            glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW);
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
            // Ensure we have a pixel-orthographic projection for text drawing
            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            glOrtho(0, win_w, win_h, 0, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glLoadIdentity();

            menu_draw_text_at(fps_text, tx, ty, white);

            // restore matrices
            glPopMatrix();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);
        }

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
    if (v > 0.0) s->dt = (float)v;
}