#version 430

flat in int v_ctype;
out vec4 frag_color;

// Must match Constraint.h enum
const int CT_DIST   = 0;
const int CT_ANCHOR = 1;
const int CT_SPRING = 2;

void main() {
    vec3 col = (v_ctype == CT_SPRING) ? vec3(0.20, 0.60, 1.00) :
               (v_ctype == CT_ANCHOR) ? vec3(0.90, 0.80, 0.10) :
                                        vec3(0.85, 0.85, 0.30);
    frag_color = vec4(col, 1.0);
}
