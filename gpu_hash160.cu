#include "gpu_hash160.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace {
std::once_flag init_flag;
bool cuda_available = false;
std::mutex gpu_mutex;
std::uint8_t* d_inputs = nullptr;
std::uint8_t* d_outputs = nullptr;
std::size_t input_capacity = 0;
std::size_t output_capacity = 0;

bool check_cuda(cudaError_t err) { return err == cudaSuccess; }

bool ensure_device_capacity(std::size_t count) {
  const std::size_t required_inputs = count * 33;
  const std::size_t required_outputs = count * 20;

  if (required_inputs > input_capacity) {
    if (d_inputs) {
      cudaFree(d_inputs);
      d_inputs = nullptr;
      input_capacity = 0;
    }
    if (!check_cuda(cudaMalloc(&d_inputs, required_inputs))) {
      return false;
    }
    input_capacity = required_inputs;
  }

  if (required_outputs > output_capacity) {
    if (d_outputs) {
      cudaFree(d_outputs);
      d_outputs = nullptr;
      output_capacity = 0;
    }
    if (!check_cuda(cudaMalloc(&d_outputs, required_outputs))) {
      return false;
    }
    output_capacity = required_outputs;
  }

  return true;
}

bool initialize_cuda() {
  int device_count = 0;
  if (!check_cuda(cudaGetDeviceCount(&device_count))) {
    return false;
  }
  if (device_count <= 0) {
    return false;
  }
  if (!check_cuda(cudaSetDevice(0))) {
    return false;
  }
  return true;
}
}  // namespace

__device__ inline std::uint32_t rotr32(std::uint32_t x, int s) {
  return (x >> s) | (x << (32 - s));
}

__device__ inline std::uint32_t ch(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  return (x & y) ^ ((~x) & z);
}

__device__ inline std::uint32_t maj(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  return (x & y) ^ (x & z) ^ (y & z);
}

__device__ inline std::uint32_t bsig0(std::uint32_t x) {
  return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}

__device__ inline std::uint32_t bsig1(std::uint32_t x) {
  return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}

__device__ inline std::uint32_t ssig0(std::uint32_t x) {
  return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}

__device__ inline std::uint32_t ssig1(std::uint32_t x) {
  return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

__device__ void sha256_block(const std::uint8_t block[64], std::uint32_t out_state[8]) {
  const std::uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u,
      0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
      0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u,
      0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
      0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
      0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
      0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
      0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au,
      0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

  std::uint32_t w[64];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    w[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
           (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
           (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
           (static_cast<std::uint32_t>(block[i * 4 + 3]));
  }
#pragma unroll
  for (int i = 16; i < 64; ++i) {
    w[i] = ssig1(w[i - 2]) + w[i - 7] + ssig0(w[i - 15]) + w[i - 16];
  }

  std::uint32_t a = 0x6a09e667u;
  std::uint32_t b = 0xbb67ae85u;
  std::uint32_t c = 0x3c6ef372u;
  std::uint32_t d = 0xa54ff53au;
  std::uint32_t e = 0x510e527fu;
  std::uint32_t f = 0x9b05688cu;
  std::uint32_t g = 0x1f83d9abu;
  std::uint32_t h = 0x5be0cd19u;

#pragma unroll
  for (int i = 0; i < 64; ++i) {
    std::uint32_t temp1 = h + bsig1(e) + ch(e, f, g) + k[i] + w[i];
    std::uint32_t temp2 = bsig0(a) + maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }

  out_state[0] = a + 0x6a09e667u;
  out_state[1] = b + 0xbb67ae85u;
  out_state[2] = c + 0x3c6ef372u;
  out_state[3] = d + 0xa54ff53au;
  out_state[4] = e + 0x510e527fu;
  out_state[5] = f + 0x9b05688cu;
  out_state[6] = g + 0x1f83d9abu;
  out_state[7] = h + 0x5be0cd19u;
}

__device__ inline std::uint32_t rol32(std::uint32_t x, int s) {
  return (x << s) | (x >> (32 - s));
}

__device__ std::uint32_t f_ripemd(int j, std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  if (j <= 15) return x ^ y ^ z;
  if (j <= 31) return (x & y) | (~x & z);
  if (j <= 47) return (x | ~y) ^ z;
  if (j <= 63) return (x & z) | (y & ~z);
  return x ^ (y | ~z);
}

__device__ std::uint32_t K_ripemd(int j) {
  if (j <= 15) return 0x00000000u;
  if (j <= 31) return 0x5a827999u;
  if (j <= 47) return 0x6ed9eba1u;
  if (j <= 63) return 0x8f1bbcdcu;
  return 0xa953fd4eu;
}

__device__ std::uint32_t Kp_ripemd(int j) {
  if (j <= 15) return 0x50a28be6u;
  if (j <= 31) return 0x5c4dd124u;
  if (j <= 47) return 0x6d703ef3u;
  if (j <= 63) return 0x7a6d76e9u;
  return 0x00000000u;
}

__device__ const int r_ripemd[80] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                                     13, 14, 15, 7,  4,  13, 1,  10, 6,  15, 3,  12,
                                     0,  9,  5,  2,  14, 11, 8,  3,  10, 14, 4,  9,  15,
                                     8,  1,  2,  7,  0,  6,  13, 11, 5,  12, 1,  9,  11,
                                     10, 0,  8,  12, 4,  13, 3,  7,  15, 14, 5,  6,  2,
                                     4,  0,  5,  9,  7,  12, 2,  10, 14, 1,  3,  8,  11,
                                     6,  15, 13};

__device__ const int rp_ripemd[80] = {5,  14, 7,  0,  9,  2,  11, 4,  13, 6,  15, 8,  1,
                                      10, 3,  12, 6,  11, 3,  7,  0,  13, 5,  10, 14, 15,
                                      8,  12, 4,  9,  1,  2,  15, 5,  1,  3,  7,  14, 6,
                                      9,  11, 8,  12, 2,  10, 0,  13, 4,  8,  6,  4,  1,
                                      3,  11, 15, 0,  5,  12, 2,  13, 9,  7,  10, 14, 12,
                                      15, 10, 4,  1,  5,  8,  7,  6,  2,  13, 14, 0,  3,
                                      9,  11};

__device__ const int s_ripemd[80] = {11, 14, 15, 12, 5,  8,  7,  9,  11, 13, 14, 15, 6,
                                     7,  9,  8,  7,  6,  8,  13, 11, 9,  7,  15, 7,  12,
                                     15, 9,  11, 7,  13, 12, 11, 13, 6,  7,  14, 9,  13,
                                     15, 14, 8,  13, 6,  5,  12, 7,  5,  11, 12, 14, 15,
                                     14, 15, 9,  8,  9,  14, 5,  6,  8,  6,  5,  12, 9,
                                     15, 5,  11, 6,  8,  13, 12, 5,  12, 13, 14, 11, 8,
                                     5,  6};

__device__ const int sp_ripemd[80] = {8,  9,  9,  11, 13, 15, 15, 5,  7,  7,  8,  11, 14,
                                      14, 12, 6,  9,  13, 15, 7,  12, 8,  9,  11, 7,  7,
                                      12, 7,  6,  15, 13, 11, 9,  7,  15, 11, 8,  6,  6,
                                      14, 12, 13, 5,  14, 13, 13, 7,  5,  15, 5,  8,  11,
                                      14, 14, 6,  14, 6,  9,  12, 9,  12, 5,  15, 8,  8,
                                      5,  12, 9,  12, 5,  14, 6,  8,  13, 6,  5,  15, 13,
                                      11, 11};

__device__ void ripemd160_block(const std::uint8_t block[64], std::uint32_t state[5]) {
  std::uint32_t x[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    x[i] = static_cast<std::uint32_t>(block[i * 4]) |
           (static_cast<std::uint32_t>(block[i * 4 + 1]) << 8) |
           (static_cast<std::uint32_t>(block[i * 4 + 2]) << 16) |
           (static_cast<std::uint32_t>(block[i * 4 + 3]) << 24);
  }

  std::uint32_t A1 = 0x67452301u;
  std::uint32_t B1 = 0xefcdab89u;
  std::uint32_t C1 = 0x98badcfeu;
  std::uint32_t D1 = 0x10325476u;
  std::uint32_t E1 = 0xc3d2e1f0u;

  std::uint32_t A2 = A1;
  std::uint32_t B2 = B1;
  std::uint32_t C2 = C1;
  std::uint32_t D2 = D1;
  std::uint32_t E2 = E1;

#pragma unroll
  for (int j = 0; j < 80; ++j) {
    std::uint32_t T = rol32(A1 + f_ripemd(j, B1, C1, D1) + x[r_ripemd[j]] + K_ripemd(j),
                             s_ripemd[j]) + E1;
    A1 = E1;
    E1 = D1;
    D1 = rol32(C1, 10);
    C1 = B1;
    B1 = T;

    std::uint32_t Tp = rol32(A2 + f_ripemd(79 - j, B2, C2, D2) + x[rp_ripemd[j]] + Kp_ripemd(j),
                              sp_ripemd[j]) + E2;
    A2 = E2;
    E2 = D2;
    D2 = rol32(C2, 10);
    C2 = B2;
    B2 = Tp;
  }

  std::uint32_t T = state[1] + C1 + D2;
  state[1] = state[2] + D1 + E2;
  state[2] = state[3] + E1 + A2;
  state[3] = state[4] + A1 + B2;
  state[4] = state[0] + B1 + C2;
  state[0] = T;
}

__global__ void hash160_kernel(const std::uint8_t* pub_keys, std::size_t count,
                               std::uint8_t* hashes) {
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  std::uint8_t sha_block[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) sha_block[i] = 0;

#pragma unroll
  for (int i = 0; i < 33; ++i) {
    sha_block[i] = pub_keys[idx * 33 + i];
  }
  sha_block[33] = 0x80;
  const std::uint32_t bit_len = 33 * 8;
  sha_block[62] = static_cast<std::uint8_t>((bit_len >> 8) & 0xFF);
  sha_block[63] = static_cast<std::uint8_t>(bit_len & 0xFF);

  std::uint32_t sha_state[8];
  sha256_block(sha_block, sha_state);

  std::uint8_t ripemd_block[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) ripemd_block[i] = 0;

#pragma unroll
  for (int i = 0; i < 8; ++i) {
    ripemd_block[i * 4] = static_cast<std::uint8_t>((sha_state[i] >> 24) & 0xFF);
    ripemd_block[i * 4 + 1] = static_cast<std::uint8_t>((sha_state[i] >> 16) & 0xFF);
    ripemd_block[i * 4 + 2] = static_cast<std::uint8_t>((sha_state[i] >> 8) & 0xFF);
    ripemd_block[i * 4 + 3] = static_cast<std::uint8_t>(sha_state[i] & 0xFF);
  }
  ripemd_block[32] = 0x80;
  ripemd_block[60] = 0x00;
  ripemd_block[61] = 0x00;
  ripemd_block[62] = 0x01;
  ripemd_block[63] = 0x00;

  std::uint32_t ripemd_state[5] = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u,
                                   0xc3d2e1f0u};
  ripemd160_block(ripemd_block, ripemd_state);

#pragma unroll
  for (int i = 0; i < 5; ++i) {
    hashes[idx * 20 + i * 4] = static_cast<std::uint8_t>(ripemd_state[i] & 0xFF);
    hashes[idx * 20 + i * 4 + 1] =
        static_cast<std::uint8_t>((ripemd_state[i] >> 8) & 0xFF);
    hashes[idx * 20 + i * 4 + 2] =
        static_cast<std::uint8_t>((ripemd_state[i] >> 16) & 0xFF);
    hashes[idx * 20 + i * 4 + 3] =
        static_cast<std::uint8_t>((ripemd_state[i] >> 24) & 0xFF);
  }
}

bool gpu_hash160_is_available() {
  std::call_once(init_flag, []() { cuda_available = initialize_cuda(); });
  return cuda_available;
}

bool gpu_hash160_hash(const std::uint8_t* pub_keys, std::size_t count,
                      std::uint8_t* hash_out) {
  if (count == 0) {
    return true;
  }
  if (!gpu_hash160_is_available()) {
    return false;
  }

  std::lock_guard<std::mutex> lock(gpu_mutex);

  if (!ensure_device_capacity(count)) {
    return false;
  }

  const std::size_t input_bytes = count * 33;
  const std::size_t output_bytes = count * 20;
  if (!check_cuda(cudaMemcpy(d_inputs, pub_keys, input_bytes, cudaMemcpyHostToDevice))) {
    return false;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  hash160_kernel<<<blocks, threads>>>(d_inputs, count, d_outputs);
  if (!check_cuda(cudaGetLastError())) {
    return false;
  }
  if (!check_cuda(cudaMemcpy(hash_out, d_outputs, output_bytes, cudaMemcpyDeviceToHost))) {
    return false;
  }
  return true;
}

#endif  // HAVE_CUDA
