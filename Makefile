# Simple Makefile for SDL2 + OpenGL C project

# Compiler
CC = gcc

# Source files
SRC = main.c Menu.c datastructures.c Node.c Constraint.c Simulator.c Octree.c

# Compiler flags
## CSparse / SuiteSparse settings
# You can set CSPARSE_INC and CSPARSE_LIB when calling make, for example:
# make CSPARSE_INC=./SuiteSparse/CSparse/Include CSPARSE_LIB=./SuiteSparse/CSparse/build/libcsparse.so
CSPARSE_INC ?= /usr/include/suitesparse
CSPARSE_LIB ?= -L/usr/lib/x86_64-linux-gnu -lcxsparse

# Compiler flags (always enable debug jacobian)
CFLAGS = -Wall -Wextra -g -DDEBUG_JACOBIAN $(if $(CSPARSE_INC),-I$(CSPARSE_INC))

# Linker flags for CSparse (can be a -L... -lcsparse or exact archive path)
CSPARSE_LDFLAGS = $(CSPARSE_LIB)


# Output binary
OUT = simulator

# SDL2 and OpenGL flags (Linux/WSL)
SDL_CFLAGS = $(shell sdl2-config --cflags 2>/dev/null || echo "")
SDL_LDFLAGS = $(shell sdl2-config --libs 2>/dev/null || echo "-lSDL2")
GL_LDFLAGS = -lGL
TTF_LDFLAGS = -lSDL2_ttf
MATH_LDFLAGS = -lm

# Build rule
all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(SDL_CFLAGS) $(SDL_LDFLAGS) $(GL_LDFLAGS) $(TTF_LDFLAGS) $(MATH_LDFLAGS) $(CSPARSE_LDFLAGS)

clean:
	rm -f $(OUT)
