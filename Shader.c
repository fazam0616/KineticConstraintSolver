#include "Shader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <GL/glew.h>
#include <GL/gl.h>

GLuint g_phong_program = 0;

static char *read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char*)malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    buf[sz] = '\0';
    fclose(f);
    return buf;
}

static GLuint compile_shader(const char *src, GLenum type) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024]; GLsizei l = 0; glGetShaderInfoLog(s, sizeof(log), &l, log);
        fprintf(stderr, "Shader compile error: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint shader_load_program(const char *vert_path, const char *frag_path) {
    char *vsrc = read_text_file(vert_path);
    char *fsrc = read_text_file(frag_path);
    if (!vsrc || !fsrc) {
        fprintf(stderr, "Failed to read shader files: %s , %s\n", vert_path, frag_path);
        free(vsrc); free(fsrc); return 0;
    }
    GLuint vs = compile_shader(vsrc, GL_VERTEX_SHADER);
    GLuint fs = compile_shader(fsrc, GL_FRAGMENT_SHADER);
    free(vsrc); free(fsrc);
    if (!vs || !fs) { if (vs) glDeleteShader(vs); if (fs) glDeleteShader(fs); return 0; }
    GLuint prog = glCreateProgram();
    // bind attribute locations for consistency
    glBindAttribLocation(prog, 0, "a_position");
    glBindAttribLocation(prog, 1, "a_normal");
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok = 0; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024]; GLsizei l = 0; glGetProgramInfoLog(prog, sizeof(log), &l, log);
        fprintf(stderr, "Shader link error: %s\n", log);
        glDeleteProgram(prog);
        glDeleteShader(vs); glDeleteShader(fs);
        return 0;
    }
    glDetachShader(prog, vs); glDetachShader(prog, fs);
    glDeleteShader(vs); glDeleteShader(fs);
    return prog;
}

void shader_free_program(GLuint prog) {
    if (prog) glDeleteProgram(prog);
    if (g_phong_program == prog) g_phong_program = 0;
}
