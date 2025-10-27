#include "Menu.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL_ttf.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

Menu* menu_create(int x, int y, int width, int height, int z_index, const char *title, Color textColor, Color bgColor) {
    Menu *menu = (Menu*)malloc(sizeof(Menu));
    menu->x = x;
    menu->y = y;
    menu->width = width;
    menu->height = height;
    menu->z_index = z_index;
    menu->title = strdup(title);
    menu->rows = dynarray_create(4);
    menu->textColor = textColor;
    menu->bgColor = bgColor;
    return menu;
}

void menu_add_row(Menu *menu, MenuRow *row) {
    dynarray_append(menu->rows, row);
}

MenuRow* menurow_create() {
    MenuRow *row = (MenuRow*)malloc(sizeof(MenuRow));
    row->interactions = dynarray_create(2);
    return row;
}

void menurow_add_interaction(MenuRow *row, VariableInteraction *interaction) {
    dynarray_append(row->interactions, interaction);
}

VariableInteraction* variableinteraction_create(void *variable, const char *name, double min, double max, VariableType type, InteractionCallback on_change, void *callback_data) {
    VariableInteraction *vi = (VariableInteraction*)malloc(sizeof(VariableInteraction));
    vi->variable = variable;
    vi->name = strdup(name);
    vi->min = min;
    vi->max = max;
    vi->type = type;
    vi->on_change = on_change;
    vi->callback_data = callback_data;
    return vi;
}

// Helper: check if point in rect
static int point_in_rect(int px, int py, int rx, int ry, int rw, int rh) {
    return px >= rx && px < rx + rw && py >= ry && py < ry + rh;
}

// Compute layout and handle mouse interactions
// Returns 1 if the event was handled (consumed) by this menu, 0 otherwise.
int menu_handle_mouse_button(Menu *menu, int button, int state, int mx, int my) {
    if (!menu) return 0;
    int title_h = 24;
    size_t rows = menu->rows->size;
    if (rows == 0) return 0;
    int avail_h = menu->height - title_h;

    for (size_t r = 0; r < rows; ++r) {
        MenuRow *row = (MenuRow*)menu->rows->items[r];
        size_t cols = row->interactions->size;
        if (cols == 0) continue;
        int row_h = avail_h / (int)rows;
        int ry = menu->y + title_h + (int)r * row_h;
        int slot_w = menu->width / (int)cols;

        for (size_t c = 0; c < cols; ++c) {
            VariableInteraction *vi = (VariableInteraction*)row->interactions->items[c];
            int rx = menu->x + (int)c * slot_w;
            int pad = 8;

            if (vi->type == VAR_BOOL) {
                int square_size = 16;
                int label_w = 0, label_h = 0;
                if (menu_measure_text(vi->name, &label_w, &label_h) != 0) label_w = 0;
                int gap = 6;
                int sx = rx + pad + label_w + gap;
                int sy = ry + (row_h - square_size) / 2;

                if (state == SDL_PRESSED && point_in_rect(mx, my, sx, sy, square_size, square_size)) {
                    int *val = (int*)vi->variable;
                    *val = !(*val);
                    if (vi->on_change) vi->on_change(vi, vi->callback_data);
                    return 1;
                }
            } else if (vi->type == VAR_SLIDER) {
                /* Compose a transient label that includes the slider's current value formatted to 2 decimals.
                   This avoids any heap allocation per-frame and ensures measurement/drawing match. */
                char labelbuf[64];
                double cur = *(double*)vi->variable;
                if (vi->max > vi->min) {
                    /* clamp display value into range for nicer formatting */
                    if (cur < vi->min) cur = vi->min;
                    if (cur > vi->max) cur = vi->max;
                }
                snprintf(labelbuf, sizeof(labelbuf), "%s: %.2f", vi->name, cur);

                int label_w = 0, label_h = 0;
                if (menu_measure_text(labelbuf, &label_w, &label_h) != 0) label_w = 0;
                int gap = 8;
                int bar_x = rx + pad + label_w + gap;
                int bar_w = slot_w - (pad + label_w + gap + pad);
                if (bar_w < 32) bar_w = 32;
                int bar_y = ry + row_h / 2 - 6;
                int bar_h = 12;

                if (state == SDL_PRESSED && point_in_rect(mx, my, bar_x, bar_y, bar_w, bar_h)) {
                    double t = (double)(mx - bar_x) / (double)bar_w;
                    if (t < 0) t = 0; if (t > 1) t = 1;
                    double val = vi->min + t * (vi->max - vi->min);
                    *(double*)vi->variable = val;
                    if (vi->on_change) vi->on_change(vi, vi->callback_data);
                    menu->active_interaction = vi;
                    menu->dragging = 1;
                    return 1;
                }
            }
        }
    }

    if (state == SDL_RELEASED) {
        int was_dragging = menu->dragging;
        menu->dragging = 0;
        menu->active_interaction = NULL;
        return was_dragging ? 1 : 0;
    }
    return 0;
}

// --- Font / text rendering ---
static TTF_Font *g_font = NULL;

int menu_set_font(const char *font_path, int pt_size) {
    if (!font_path) return -1;
    if (g_font) TTF_CloseFont(g_font);
    if (TTF_WasInit() == 0) {
        if (TTF_Init() != 0) {
            fprintf(stderr, "TTF_Init failed: %s\n", TTF_GetError());
            return -1;
        }
    }
    g_font = TTF_OpenFont(font_path, pt_size);
    if (!g_font) {
        fprintf(stderr, "TTF_OpenFont failed for '%s': %s\n", font_path, TTF_GetError());
        return -1;
    }
    return 0;
}

void menu_clear_font(void) {
    if (g_font) {
        TTF_CloseFont(g_font);
        g_font = NULL;
    }
    if (TTF_WasInit()) TTF_Quit();
}

// Render text at pixel coords (x,y) with menu's text color. Uses immediate texture creation.
static void draw_text(Menu *menu, const char *text, int x, int y) {
    if (!g_font || !text || !menu) return;
    SDL_Color sc = { menu->textColor.r, menu->textColor.g, menu->textColor.b, menu->textColor.a };
    SDL_Surface *surf = TTF_RenderUTF8_Blended(g_font, text, sc);
    if (!surf) return;
    SDL_Surface *conv = SDL_ConvertSurfaceFormat(surf, SDL_PIXELFORMAT_ABGR8888, 0);
    SDL_FreeSurface(surf);
    if (!conv) return;

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, conv->w, conv->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, conv->pixels);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_TEXTURE_2D);

    glColor4ub(255, 255, 255, 255);
    glBegin(GL_QUADS);
    glTexCoord2f(0,0); glVertex2i(x, y);
    glTexCoord2f(1,0); glVertex2i(x + conv->w, y);
    glTexCoord2f(1,1); glVertex2i(x + conv->w, y + conv->h);
    glTexCoord2f(0,1); glVertex2i(x, y + conv->h);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);

    glDeleteTextures(1, &tex);
    SDL_FreeSurface(conv);
}

// Public: measure text using the loaded font. Returns 0 on success, -1 on failure.
int menu_measure_text(const char *text, int *out_w, int *out_h) {
    if (!g_font || !text) return -1;
    int w = 0, h = 0;
    if (TTF_SizeUTF8(g_font, text, &w, &h) != 0) return -1;
    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return 0;
}

// Public: draw text at pixel coordinates using a specified color. Uses immediate texture creation like draw_text.
void menu_draw_text_at(const char *text, int x, int y, Color color) {
    if (!g_font || !text) return;
    SDL_Color sc = { color.r, color.g, color.b, color.a };
    SDL_Surface *surf = TTF_RenderUTF8_Blended(g_font, text, sc);
    if (!surf) return;
    SDL_Surface *conv = SDL_ConvertSurfaceFormat(surf, SDL_PIXELFORMAT_ABGR8888, 0);
    SDL_FreeSurface(surf);
    if (!conv) return;

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, conv->w, conv->h, 0, GL_RGBA, GL_UNSIGNED_BYTE, conv->pixels);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_TEXTURE_2D);

    glColor4ub(255,255,255,255);
    // The caller expects pixel coordinates with origin top-left; match menu draw_text behaviour
    glBegin(GL_QUADS);
    glTexCoord2f(0,0); glVertex2i(x, y);
    glTexCoord2f(1,0); glVertex2i(x + conv->w, y);
    glTexCoord2f(1,1); glVertex2i(x + conv->w, y + conv->h);
    glTexCoord2f(0,1); glVertex2i(x, y + conv->h);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);

    glDeleteTextures(1, &tex);
    SDL_FreeSurface(conv);
}

// Helper: draw filled rectangle in pixel coordinates
static void draw_filled_rect(int x, int y, int w, int h, Color c) {
    glColor4ub(c.r, c.g, c.b, c.a);
    glBegin(GL_QUADS);
    glVertex2i(x, y);
    glVertex2i(x + w, y);
    glVertex2i(x + w, y + h);
    glVertex2i(x, y + h);
    glEnd();
}

static void draw_rect_border(int x, int y, int w, int h, Color c) {
    glColor4ub(c.r, c.g, c.b, c.a);
    glBegin(GL_LINE_LOOP);
    glVertex2i(x, y);
    glVertex2i(x + w, y);
    glVertex2i(x + w, y + h);
    glVertex2i(x, y + h);
    glEnd();
}
int menu_handle_mouse_motion(Menu *menu, int mx, int my) {
    if (!menu) return 0;
    if (!menu->dragging || !menu->active_interaction) return 0;
    // find active interaction location
    int title_h = 24;
    size_t rows = menu->rows->size;
    if (rows == 0) return 0;
    int avail_h = menu->height - title_h;

    for (size_t r = 0; r < rows; ++r) {
        MenuRow *row = (MenuRow*)menu->rows->items[r];
        size_t cols = row->interactions->size;
        if (cols == 0) continue;
        int row_h = avail_h / (int)rows;
        int ry = menu->y + title_h + (int)r * row_h;
        int slot_w = menu->width / (int)cols;

        for (size_t c = 0; c < cols; ++c) {
            VariableInteraction *vi = (VariableInteraction*)row->interactions->items[c];
            if (vi != menu->active_interaction) continue;
            int rx = menu->x + (int)c * slot_w;
            int label_w = 0, label_h = 0;
            int pad = 8, gap = 8;
            char labelbuf[64];
            const char *labelptr = vi->name;
            if (vi->type == VAR_SLIDER) {
                double cur = *(double*)vi->variable;
                if (vi->max > vi->min) {
                    if (cur < vi->min) cur = vi->min;
                    if (cur > vi->max) cur = vi->max;
                }
                snprintf(labelbuf, sizeof(labelbuf), "%s: %.2f", vi->name, cur);
                labelptr = labelbuf;
            }
            if (menu_measure_text(labelptr, &label_w, &label_h) != 0) label_w = 0;
            int bar_x = rx + pad + label_w + gap;
            int bar_w = slot_w - (pad + label_w + gap + pad);
            if (bar_w < 32) bar_w = 32;
            // clamp and compute t
            double t = (double)(mx - bar_x) / (double)bar_w;
            if (t < 0) t = 0; if (t > 1) t = 1;
            double val = vi->min + t * (vi->max - vi->min);
            *(double*)vi->variable = val;
            if (vi->on_change) vi->on_change(vi, vi->callback_data);
            return 1;
        }
    }
    return 0;
}

void menu_render(Menu *menu, int window_w, int window_h) {
    if (!menu) return;
    // setup 2D orthographic projection matching window pixels
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, window_w, window_h, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // draw background
    draw_filled_rect(menu->x, menu->y, menu->width, menu->height, menu->bgColor);
    // title bar (left-top area)
    Color titleColor = { (uint8_t) (menu->bgColor.r + 30), (uint8_t)(menu->bgColor.g + 30), (uint8_t)(menu->bgColor.b + 30), menu->bgColor.a };
    int title_h = 24;
    draw_filled_rect(menu->x, menu->y, menu->width, title_h, titleColor);
    // draw title text
    draw_text(menu, menu->title, menu->x + 8, menu->y + 6);
    draw_rect_border(menu->x, menu->y, menu->width, menu->height, menu->textColor);

    // layout rows and items
    size_t rows = menu->rows->size;
    if (rows == 0) {
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        return;
    }
    int avail_h = menu->height - title_h;
    for (size_t r = 0; r < rows; ++r) {
        MenuRow *row = (MenuRow*)menu->rows->items[r];
        size_t cols = row->interactions->size;
        if (cols == 0) continue;
        int row_h = avail_h / (int)rows;
        int ry = menu->y + title_h + (int)r * row_h;
        int slot_w = menu->width / (int)cols;
        for (size_t c = 0; c < cols; ++c) {
            VariableInteraction *vi = (VariableInteraction*)row->interactions->items[c];
            int rx = menu->x + (int)c * slot_w;
            int pad = 8; int gap = 8;
            // draw interaction control with label on the left
            int label_w = 0, label_h = 0;
            char labelbuf[64];
            const char *labelptr = vi->name;
            if (vi->type == VAR_SLIDER) {
                double cur = *(double*)vi->variable;
                if (vi->max > vi->min) {
                    if (cur < vi->min) cur = vi->min;
                    if (cur > vi->max) cur = vi->max;
                }
                snprintf(labelbuf, sizeof(labelbuf), "%s: %.2f", vi->name, cur);
                labelptr = labelbuf;
            }
            if (menu_measure_text(labelptr, &label_w, &label_h) != 0) label_w = 0;
            int label_x = rx + pad;
            int label_y = ry + (row_h - 12) / 2;
            if (vi->type == VAR_BOOL) {
                int square_size = 16;
                int sx = rx + pad + label_w + gap;
                int sy = ry + (row_h - square_size)/2;
                Color fill = { (uint8_t)(menu->textColor.r), (uint8_t)(menu->textColor.g), (uint8_t)(menu->textColor.b), 255 };
                if (*(int*)vi->variable) {
                    draw_filled_rect(sx, sy, square_size, square_size, fill);
                } else {
                    draw_rect_border(sx, sy, square_size, square_size, fill);
                }
                // draw label to the left of the checkbox
                draw_text(menu, labelptr, label_x, label_y);
            } else if (vi->type == VAR_SLIDER) {
                int bar_x = rx + pad + label_w + gap;
                int bar_w = slot_w - (pad + label_w + gap + pad);
                if (bar_w < 32) bar_w = 32;
                int bar_y = ry + row_h/2 - 6;
                int bar_h = 12;
                Color barBg = {200,200,200,255};
                Color barFg = {100,180,240,255};
                draw_filled_rect(bar_x, bar_y, bar_w, bar_h, barBg);
                // compute thumb position
                double cur = *(double*)vi->variable;
                double t = 0.0;
                if (vi->max > vi->min) t = (cur - vi->min) / (vi->max - vi->min);
                if (t < 0) t = 0; if (t > 1) t = 1;
                int thumb_w = 8;
                int thumb_x = bar_x + (int)(t * (bar_w - thumb_w));
                int thumb_y = bar_y - 2;
                draw_filled_rect(thumb_x, thumb_y, thumb_w, bar_h + 4, barFg);
                // draw the label to the left of the control
                draw_text(menu, labelptr, label_x, label_y);
            }
        }
    }

    // restore matrices
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}


void menu_free(Menu *menu) {
    free(menu->title);
    for (size_t i = 0; i < menu->rows->size; ++i) {
        MenuRow *row = (MenuRow*)menu->rows->items[i];
        for (size_t j = 0; j < row->interactions->size; ++j) {
            VariableInteraction *vi = (VariableInteraction*)row->interactions->items[j];
            free(vi->name);
            free(vi);
        }
        dynarray_free(row->interactions, NULL);
        free(row);
    }
    dynarray_free(menu->rows, NULL);
    free(menu);
}
