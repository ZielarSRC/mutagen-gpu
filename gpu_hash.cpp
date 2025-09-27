#include "gpu_hash.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace gpu_hash {
namespace {
std::once_flag init_flag;
std::atomic<bool> initialized{false};
std::atomic<bool> available{false};
std::mutex compute_mutex;

constexpr int kPreferredBatch = 131072;
}  // namespace

#ifdef MUTAGEN_HAS_CUDA
namespace detail {
bool cuda_initialize();
void cuda_shutdown();
bool cuda_compute_hash160_batch(int numKeys, const uint8_t (*pubKeys)[33],
                                uint8_t (*hashResults)[20]);
}  // namespace detail
#endif

bool initialize() {
  std::call_once(init_flag, []() {
#ifdef MUTAGEN_HAS_CUDA
    if (detail::cuda_initialize()) {
      available.store(true);
    } else {
      std::cerr << "[GPU] CUDA init failed.\n";
      available.store(false);
    }
#else
    available.store(false);
#endif
    initialized.store(true);
  });
  return available.load();
}

bool is_available() {
  if (!initialized.load()) (void)initialize();
  return available.load();
}

int preferred_batch_size() { return kPreferredBatch; }

bool compute_hash160_batch(int numKeys, const uint8_t (*pubKeys)[33],
                           uint8_t (*hashResults)[20]) {
#ifdef MUTAGEN_HAS_CUDA
  if (!is_available()) {
    throw std::runtime_error("[GPU] Not initialized.");
  }

  std::lock_guard<std::mutex> lock(compute_mutex);
  const bool ok = detail::cuda_compute_hash160_batch(numKeys, pubKeys, hashResults);
  if (!ok) {
    available.store(false);
    detail::cuda_shutdown();
    throw std::runtime_error("[GPU] Compute failed.");
  }
  return true;
#else
  (void)numKeys;
  (void)pubKeys;
  (void)hashResults;
  return false;
#endif
}

void shutdown() {
#ifdef MUTAGEN_HAS_CUDA
  if (available.load()) detail::cuda_shutdown();
#endif
  available.store(false);
  initialized.store(false);
}

}  // namespace gpu_hash
