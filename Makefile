# Simple Makefile for SDL2 + OpenGL C project

# Compiler
CC = gcc

# Source files
SRC = main.c Menu.c datastructures.c Node.c Constraint.c Simulator.c Octree.c BVH.c TriangleBVH.c EdgeBVH.c Shader.c

# Compiler flags
## CSparse / SuiteSparse settings
# You can set CSPARSE_INC and CSPARSE_LIB when calling make, for example:
# make CSPARSE_INC=./SuiteSparse/CSparse/Include CSPARSE_LIB=./SuiteSparse/CSparse/build/libcsparse.so
## Default to the bundled SuiteSparse/CSparse in the repository (works on Windows/MSYS/WSL)
CSPARSE_INC ?= ./SuiteSparse/CSparse/Include
# If building in MSYS2 MinGW64 and suitesparse headers are installed there,
# prefer the /mingw64 include/lib paths so the compiler finds cs.h automatically.
## If building in MSYS2 MinGW64, try several likely header locations for cs.h
ifneq ($(wildcard /mingw64/include/cs.h),)
	CSPARSE_INC := /mingw64/include
	CSPARSE_LIB ?= -L/mingw64/lib -lcsparse
else ifneq ($(wildcard /mingw64/include/suitesparse/cs.h),)
	CSPARSE_INC := /mingw64/include/suitesparse
	CSPARSE_LIB ?= -L/mingw64/lib -lcsparse
else ifneq ($(wildcard /mingw64/include/csparse/cs.h),)
	CSPARSE_INC := /mingw64/include/csparse
	CSPARSE_LIB ?= -L/mingw64/lib -lcsparse
else
	# Link against libcsparse in the repo build directory by default. Override when calling make.
	# Try common Linux include locations before falling back to the bundled copy
	ifneq ($(wildcard /usr/include/cs.h),)
		CSPARSE_INC := /usr/include
	else ifneq ($(wildcard /usr/include/suitesparse/cs.h),)
		CSPARSE_INC := /usr/include/suitesparse
	else ifneq ($(wildcard /usr/include/csparse/cs.h),)
		CSPARSE_INC := /usr/include/csparse
	endif
	CSPARSE_LIB ?= -L./SuiteSparse/CSparse/build -lcsparse
endif

# If MSYS2 installed suitesparse under different library names, prefer them when present
ifneq ($(wildcard /mingw64/lib/libcxsparse.a),)
	CSPARSE_LIB := /mingw64/lib/libcxsparse.a -lsuitesparseconfig
else ifneq ($(wildcard /mingw64/lib/libcxsparse.dll.a),)
	CSPARSE_LIB := /mingw64/lib/libcxsparse.dll.a -lsuitesparseconfig
else ifneq ($(wildcard /mingw64/lib/libsuitesparseconfig.a),)
	CSPARSE_LIB := /mingw64/lib/libsuitesparseconfig.a
else ifneq ($(wildcard /mingw64/lib/libsuitesparseconfig.dll.a),)
	CSPARSE_LIB := /mingw64/lib/libsuitesparseconfig.dll.a
endif

# Prefer common Linux library filenames/locations when available
ifneq ($(wildcard /usr/lib/libcxsparse.a),)
	CSPARSE_LIB := /usr/lib/libcxsparse.a -lsuitesparseconfig
else ifneq ($(wildcard /usr/lib/libcxsparse.so),)
	CSPARSE_LIB := -lcxsparse -lsuitesparseconfig
else ifneq ($(wildcard /usr/lib/x86_64-linux-gnu/libcxsparse.a),)
	CSPARSE_LIB := /usr/lib/x86_64-linux-gnu/libcxsparse.a -lsuitesparseconfig
else ifneq ($(wildcard /usr/lib/x86_64-linux-gnu/libcxsparse.so),)
	CSPARSE_LIB := -lcxsparse -lsuitesparseconfig
else ifneq ($(wildcard /usr/lib/libsuitesparseconfig.a),)
	CSPARSE_LIB := /usr/lib/libsuitesparseconfig.a
else ifneq ($(wildcard /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a),)
	CSPARSE_LIB := /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.a
endif

# Compiler flags (always enable debug jacobian)
CFLAGS = -Wall -Wextra -O3 -DDEBUG_JACOBIAN $(if $(CSPARSE_INC),-I$(CSPARSE_INC))

# Linker flags for CSparse (can be a -L... -lcsparse or exact archive path)
CSPARSE_LDFLAGS = $(CSPARSE_LIB)


# Output binary
OUT = simulator

# SDL2 and OpenGL flags
# Prefer pkg-config on MSYS2 to avoid sdl2-config returning Linux -lGL.
ifeq ($(OS),Windows_NT)
	SDL_CFLAGS  = $(shell pkg-config --cflags sdl2 2>/dev/null || echo "-IC:/msys64/mingw64/include/SDL2 -Dmain=SDL_main")
	# Build as console subsystem (no -mwindows) so stdout/stderr are available in terminals
	# Ensure any -mwindows or Windows-subsystem flags coming from pkg-config are stripped
	SDL_LDFLAGS = $(shell pkg-config --libs sdl2 2>/dev/null | sed 's/-lGL//g; s/-lGLU//g; s/-mwindows//g; s/-Wl,--subsystem,windows//g' || echo "-lmingw32 -lSDL2main -lSDL2")
	# Link against GLEW and OpenGL on Windows (MSYS2 MinGW-w64 provides libglew32)
	override GL_LDFLAGS := -lglew32 -lopengl32
else
	SDL_CFLAGS  = $(shell sdl2-config --cflags 2>/dev/null || echo "")
	SDL_LDFLAGS = $(shell sdl2-config --libs 2>/dev/null || echo "-lSDL2")
	override GL_LDFLAGS := -lGL -lGLEW
endif
TTF_LDFLAGS = -lSDL2_ttf
MATH_LDFLAGS = -lm

## Construct final link flags: remove any -lGL/-lGLU from auto flags, then append the chosen GL_LDFLAGS
AUTO_LDFLAGS := $(SDL_LDFLAGS) $(TTF_LDFLAGS) $(MATH_LDFLAGS) $(CSPARSE_LDFLAGS)
FILTERED_LDFLAGS := $(shell echo '$(AUTO_LDFLAGS)' | sed 's/-lGL//g; s/-lGLU//g; s/-mwindows//g; s/-Wl,--subsystem,windows//g')
LINK_LDFLAGS := $(FILTERED_LDFLAGS) $(GL_LDFLAGS)

all:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(SDL_CFLAGS) $(LINK_LDFLAGS)

clean:
	rm -f $(OUT)

show:
	@echo CFLAGS=$(CFLAGS)
	@echo SDL_CFLAGS=$(SDL_CFLAGS)
	@echo SDL_LDFLAGS=$(SDL_LDFLAGS)
	@echo GL_LDFLAGS=$(GL_LDFLAGS)
	@echo TTF_LDFLAGS=$(TTF_LDFLAGS)
	@echo CSPARSE_LDFLAGS=$(CSPARSE_LDFLAGS)
	@echo LDFLAGS=$(LDFLAGS)
	@echo LDLIBS=$(LDLIBS)
