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

    int n = 6;

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

            // Menu input handling
            // if (event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_MOUSEBUTTONUP) {
            //     int state = (event.type == SDL_MOUSEBUTTONDOWN) ? SDL_PRESSED : SDL_RELEASED;
            //     menu_handle_mouse_button(test, event.button.button, state, event.button.x, event.button.y);
            // } else if (event.type == SDL_MOUSEMOTION) {
            //     menu_handle_mouse_motion(test, event.motion.x, event.motion.y);
            // }

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
                    float world_dx = (float)dx / cam_scale;
                    float world_dy = -(float)dy / cam_scale;
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
        glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Render simulator (use pixel orthographic projection)
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, win_w, win_h, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

    simulator_step(sim);
    // Draw simulator with a Y-flip so world +Y (up) maps to screen Y downwards
    glPushMatrix();
    glTranslatef(0.0f, (float)win_h, 0.0f);
    glScalef(1.0f, -1.0f, 1.0f);
    // apply camera pan & zoom (in world coordinates)
    glTranslatef(-cam_x, -cam_y, 0.0f);
    glScalef(cam_scale, cam_scale, 1.0f);
    simulator_draw(sim);
    glPopMatrix();

        // restore projection/modelview
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);

        // Render menu UI
        // menu_render(test, win_w, win_h);

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