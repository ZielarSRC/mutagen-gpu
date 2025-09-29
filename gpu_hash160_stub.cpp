#include "gpu_hash160.h"

#ifndef HAVE_CUDA
bool gpu_hash160_is_available() { return false; }

bool gpu_hash160_hash(const std::uint8_t*, std::size_t, std::uint8_t*) { return false; }
#endif
