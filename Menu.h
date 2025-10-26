#ifndef MENU_H
#define MENU_H

#include "datastructures.h"
#include <stdint.h>

// Variable interaction types
typedef enum {
    VAR_BOOL,
    VAR_SLIDER
} VariableType;

// Color struct
typedef struct {
    uint8_t r, g, b, a;
} Color;

// Forward declare for callback
typedef struct VariableInteraction VariableInteraction;
typedef void (*InteractionCallback)(VariableInteraction *interaction, void *user_data);

// Variable interaction
struct VariableInteraction {
    void *variable;
    char *name;
    double min, max;
    VariableType type;
    InteractionCallback on_change; // called when value changes
    void *callback_data;
};

// Menu row: a list of VariableInteractions
typedef struct {
    DynArray *interactions; // Array of VariableInteraction*
} MenuRow;

// Menu struct
typedef struct {
    int x, y; // Top-left position
    int width, height; // Size in pixels
    int z_index;
    char *title;
    DynArray *rows; // Array of MenuRow*
    Color textColor;
    Color bgColor;
    // Interaction state
    struct VariableInteraction *active_interaction;
    int dragging;
} Menu;

// Menu API
Menu* menu_create(int x, int y, int width, int height, int z_index, const char *title, Color textColor, Color bgColor);
void menu_add_row(Menu *menu, MenuRow *row);
MenuRow* menurow_create();
void menurow_add_interaction(MenuRow *row, VariableInteraction *interaction);
VariableInteraction* variableinteraction_create(void *variable, const char *name, double min, double max, VariableType type, InteractionCallback on_change, void *callback_data);
// Input handling for menus
// mouse_button: button state SDL_PRESSED/SDL_RELEASED
// These return 1 if the menu consumed/handled the event, 0 otherwise.
int menu_handle_mouse_button(Menu *menu, int button, int state, int mx, int my);
int menu_handle_mouse_motion(Menu *menu, int mx, int my);
void menu_render(Menu *menu, int window_w, int window_h);
// Font handling (global for Menu module)
// Returns 0 on success, -1 on failure
int menu_set_font(const char *font_path, int pt_size);
void menu_clear_font(void);
void menu_free(Menu *menu);
// Measure text in pixels using the current font. Returns 0 on success, -1 if no font.
int menu_measure_text(const char *text, int *out_w, int *out_h);
// Draw text at pixel coordinates (x,y) with explicit color. Uses the global font if set.
void menu_draw_text_at(const char *text, int x, int y, Color color);

#endif // MENU_H
