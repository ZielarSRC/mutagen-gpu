# Makefile — fixed tabs and formatting (Linux + Windows)
# Build: make            (Linux)
#        mingw32-make    (Windows / MSYS2)

# Detect OS
UNAME_S := $(shell uname -s)

# Enable static linking by default (Windows only toggle below)
STATIC_LINKING = yes

# ---------------- Linux ----------------
ifeq ($(UNAME_S),Linux)

CXX := g++

CXXFLAGS := -m64 -std=c++20 -Ofast -mssse3 -Wall -Wextra \
            -Wno-write-strings -Wno-unused-variable -Wno-deprecated-copy \
            -Wno-unused-parameter -Wno-sign-compare -Wno-strict-aliasing \
            -Wno-unused-but-set-variable \
            -funroll-loops -ftree-vectorize -fstrict-aliasing -fno-semantic-interposition \
            -fvect-cost-model=unlimited -fno-trapping-math -fipa-ra -flto \
            -fassociative-math -fopenmp -mavx2 -mbmi2 -madx -fwrapv

SRCS := mutagen.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp \
        Point.cpp ripemd160_avx2.cpp sha256_avx2.cpp

OBJS := $(SRCS:.cpp=.o)

TARGET := mutagen

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
	@chmod +x $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo "Cleaning..."
	@rm -f $(OBJS) $(TARGET)

.PHONY: all clean

else

# ---------------- Windows (MinGW-w64) ----------------
CXX := g++

# Try to hint MSYS path if g++ not in PATH (informational only)
CHECK_COMPILER := $(shell which $(CXX) 2>NUL)

CXXFLAGS := -m64 -std=c++20 -Ofast -mssse3 -Wall -Wextra \
            -Wno-write-strings -Wno-unused-variable -Wno-deprecated-copy \
            -Wno-unused-parameter -Wno-sign-compare -Wno-strict-aliasing \
            -Wno-unused-but-set-variable -funroll-loops -ftree-vectorize \
            -fstrict-aliasing -fno-semantic-interposition -fvect-cost-model=unlimited \
            -fno-trapping-math -fipa-ra -fassociative-math -fopenmp \
            -mavx2 -mbmi2 -madx -fwrapv

ifeq ($(STATIC_LINKING),yes)
CXXFLAGS += -static -static-libgcc -static-libstdc++
endif

SRCS := mutagen.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp \
        Point.cpp ripemd160_avx2.cpp sha256_avx2.cpp

OBJS := $(SRCS:.cpp=.o)

TARGET := mutagen.exe

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
	-del /q $(OBJS) 2>NUL || exit 0

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@echo Cleaning...
	-del /q $(OBJS) $(TARGET) 2>NUL || exit 0

.PHONY: all clean

endif
