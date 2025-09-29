#pragma once

#include <cstddef>
#include <cstdint>

// Initializes the CUDA hashing backend on first use and reports whether a
// compatible NVIDIA GPU is available.
bool gpu_hash160_is_available();

// Computes HASH160 (SHA-256 followed by RIPEMD-160) for a contiguous array of
// compressed public keys (33 bytes each). The output buffer must provide
// space for 20 * count bytes. Returns false if the computation failed or the
// GPU backend is unavailable.
bool gpu_hash160_hash(const std::uint8_t* pub_keys, std::size_t count,
                      std::uint8_t* hash_out);
