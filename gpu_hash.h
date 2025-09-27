#pragma once

#include <cstddef>
#include <cstdint>

namespace gpu_hash {

bool initialize();
bool is_available();
int preferred_batch_size();
bool compute_hash160_batch(int numKeys, const uint8_t (*pubKeys)[33],
                           uint8_t (*hashResults)[20]);
void shutdown();

}  // namespace gpu_hash
