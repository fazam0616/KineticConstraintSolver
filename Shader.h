#ifndef SHADER_H
#define SHADER_H

#include <GL/glew.h>

extern GLuint g_phong_program;

// Load and compile a vertex+fragment shader program. Returns program or 0 on error.
GLuint shader_load_program(const char *vert_path, const char *frag_path);

// Utility to free a program
void shader_free_program(GLuint prog);

#endif
