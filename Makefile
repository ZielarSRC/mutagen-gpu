# Detect OS
UNAME_S := $(shell uname -s)

# Enable static linking by default (change to 'no' to use dynamic linking)
STATIC_LINKING = yes

# Compiler settings based on OS
ifeq ($(UNAME_S),Linux)
# Linux settings

# Compiler
CXX = g++

# Detect CUDA compiler
NVCC := $(shell command -v nvcc 2>/dev/null)
CUDA_PATH ?= /usr/local/cuda

# Compiler flags
CXXFLAGS = -m64 -std=c++17 -Ofast -mssse3 -Wall -Wextra \
           -Wno-write-strings -Wno-unused-variable -Wno-deprecated-copy \
           -Wno-unused-parameter -Wno-sign-compare -Wno-strict-aliasing \
           -Wno-unused-but-set-variable \
           -funroll-loops -ftree-vectorize -fstrict-aliasing -fno-semantic-interposition \
           -fvect-cost-model=unlimited -fno-trapping-math -fipa-ra -flto \
           -fassociative-math -fopenmp -mavx2 -mbmi2 -madx -fwrapv

# Source files
SRCS = mutagen.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp \
       Point.cpp ripemd160_avx2.cpp sha256_avx2.cpp gpu_hash.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# CUDA settings
CUDA_OBJS =
LIBS =
ifeq ($(NVCC),)
  $(info CUDA compiler not found - building CPU-only binary.)
else
  CUDA_OBJS = gpu_hash_cuda.o
  CXXFLAGS += -DMUTAGEN_HAS_CUDA
  NVCCFLAGS = -O3 -std=c++17 -Xcompiler "-m64"
  NVCCFLAGS += -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86
  LIBS += -L$(CUDA_PATH)/lib64 -lcudart
endif

# Target executable
TARGET = mutagen

# Default target
all: $(TARGET)

# Link the object files to create the executable and then delete .o files
$(TARGET): $(OBJS) $(CUDA_OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(CUDA_OBJS) $(LIBS)
	rm -f $(OBJS) $(CUDA_OBJS) && chmod +x $(TARGET)

# Compile each source file into an object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

gpu_hash_cuda.o: gpu_hash_cuda.cu
ifeq ($(NVCC),)
	@echo "Skipping CUDA build; nvcc not available."
else
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif

# Clean up build files
clean:
	@echo "Cleaning..."
	rm -f $(OBJS) $(CUDA_OBJS) $(TARGET)

# Phony targets
.PHONY: all clean

else
# Windows settings (MinGW-w64)

# Compiler
CXX = g++

# Check if compiler is found
CHECK_COMPILER := $(shell which $(CXX))

# Add MSYS path if the compiler is not found
ifeq ($(CHECK_COMPILER),)
  $(info Compiler not found. Adding MSYS path to the environment...)
  SHELL := powershell
  PATH := C:\\msys64\\mingw64\\bin;$(PATH)
endif

# Compiler flags (without LTO)
CXXFLAGS = -m64 -std=c++17 -Ofast -mssse3 -Wall -Wextra \
           -Wno-write-strings -Wno-unused-variable -Wno-deprecated-copy \
           -Wno-unused-parameter -Wno-sign-compare -Wno-strict-aliasing \
           -Wno-unused-but-set-variable -funroll-loops -ftree-vectorize \
           -fstrict-aliasing -fno-semantic-interposition -fvect-cost-model=unlimited \
           -fno-trapping-math -fipa-ra -fassociative-math -fopenmp \
           -mavx2 -mbmi2 -madx -fwrapv

# Add -static flag if STATIC_LINKING is enabled
ifeq ($(STATIC_LINKING), yes)
    CXXFLAGS += -static
else
    $(info Dynamic linking will be used. Ensure required DLLs are distributed)
endif

# Source files
SRCS = mutagen.cpp SECP256K1.cpp Int.cpp IntGroup.cpp IntMod.cpp \
       Point.cpp ripemd160_avx2.cpp sha256_avx2.cpp gpu_hash.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = mutagen.exe

# Default target
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
	del /q $(OBJS)

# Compile each source file into an object file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	@echo Cleaning...
	del /q $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
endif
