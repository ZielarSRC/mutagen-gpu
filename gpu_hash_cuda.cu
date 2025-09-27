// CUDA HASH160: RIPEMD-160(SHA-256(pubkey[33])) — tuned for NVIDIA A10G (sm_86)
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace gpu_hash {
namespace detail {

static uint8_t* d_pubkeys = nullptr;
static uint8_t* d_hashes = nullptr;
static uint8_t* h_pk_pinned = nullptr;
static uint8_t* h_hash_pinned = nullptr;
static size_t capacity = 0;
static cudaStream_t stream = 0;
static bool ready = false;

static bool ensure_capacity(size_t n) {
  if (n == 0) return false;
  if (n <= capacity) return true;

  size_t want = 1;
  while (want < n) want <<= 1;

  if (d_pubkeys) {
    cudaFree(d_pubkeys);
    d_pubkeys = nullptr;
  }
  if (d_hashes) {
    cudaFree(d_hashes);
    d_hashes = nullptr;
  }
  if (h_pk_pinned) {
    cudaFreeHost(h_pk_pinned);
    h_pk_pinned = nullptr;
  }
  if (h_hash_pinned) {
    cudaFreeHost(h_hash_pinned);
    h_hash_pinned = nullptr;
  }

  if (cudaMalloc(&d_pubkeys, want * 33) != cudaSuccess) return false;
  if (cudaMalloc(&d_hashes, want * 20) != cudaSuccess) return false;
  if (cudaHostAlloc(&h_pk_pinned, want * 33, cudaHostAllocDefault) != cudaSuccess)
    return false;
  if (cudaHostAlloc(&h_hash_pinned, want * 20, cudaHostAllocDefault) != cudaSuccess)
    return false;

  capacity = want;
  return true;
}

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
  return (x >> n) | (x << (32 - n));
}
__device__ __forceinline__ uint32_t rotl32(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

__constant__ uint32_t K256[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U,
    0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U,
    0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU,
    0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

__device__ __forceinline__ uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (x & z) ^ (y & z);
}
__device__ __forceinline__ uint32_t BS0(uint32_t x) {
  return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}
__device__ __forceinline__ uint32_t BS1(uint32_t x) {
  return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}
__device__ __forceinline__ uint32_t SS0(uint32_t x) {
  return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}
__device__ __forceinline__ uint32_t SS1(uint32_t x) {
  return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

__device__ void sha256_one_block_33bytes(const uint8_t msg[33], uint8_t out32[32]) {
  uint32_t W[64];
#pragma unroll
  for (int i = 0; i < 16; ++i) W[i] = 0U;

  int b = 0;
  for (; b < 33; ++b) {
    int wi = b >> 2;
    int sh = 24 - 8 * (b & 3);
    W[wi] |= uint32_t(msg[b]) << sh;
  }
  {
    int wi = b >> 2;
    int sh = 24 - 8 * (b & 3);
    W[wi] |= 0x80U << sh;
  }
  W[14] = 0;
  W[15] = 33U * 8U;

  for (int t = 16; t < 64; ++t) W[t] = SS1(W[t - 2]) + W[t - 7] + SS0(W[t - 15]) + W[t - 16];

  uint32_t a = 0x6a09e667U, b0 = 0xbb67ae85U, c = 0x3c6ef372U, d = 0xa54ff53aU;
  uint32_t e = 0x510e527fU, f = 0x9b05688cU, g = 0x1f83d9abU, h = 0x5be0cd19U;

#pragma unroll
  for (int t = 0; t < 64; ++t) {
    uint32_t T1 = h + BS1(e) + Ch(e, f, g) + K256[t] + W[t];
    uint32_t T2 = BS0(a) + Maj(a, b0, c);
    h = g;
    g = f;
    f = e;
    e = d + T1;
    d = c;
    c = b0;
    b0 = a;
    a = T1 + T2;
  }

  a += 0x6a09e667U;
  b0 += 0xbb67ae85U;
  c += 0x3c6ef372U;
  d += 0xa54ff53aU;
  e += 0x510e527fU;
  f += 0x9b05688cU;
  g += 0x1f83d9abU;
  h += 0x5be0cd19U;

  uint32_t H[8] = {a, b0, c, d, e, f, g, h};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out32[4 * i + 0] = uint8_t((H[i] >> 24) & 0xff);
    out32[4 * i + 1] = uint8_t((H[i] >> 16) & 0xff);
    out32[4 * i + 2] = uint8_t((H[i] >> 8) & 0xff);
    out32[4 * i + 3] = uint8_t((H[i] >> 0) & 0xff);
  }
}

__device__ __forceinline__ uint32_t F1(uint32_t x, uint32_t y, uint32_t z) {
  return x ^ y ^ z;
}
__device__ __forceinline__ uint32_t F2(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) | (~x & z);
}
__device__ __forceinline__ uint32_t F3(uint32_t x, uint32_t y, uint32_t z) {
  return (x | ~y) ^ z;
}
__device__ __forceinline__ uint32_t F4(uint32_t x, uint32_t y, uint32_t z) {
  return (x & z) | (y & ~z);
}
__device__ __forceinline__ uint32_t F5(uint32_t x, uint32_t y, uint32_t z) {
  return x ^ (y | ~z);
}

__constant__ uint32_t KL[5] = {0x00000000U, 0x5a827999U, 0x6ed9eba1U, 0x8f1bbcdcU, 0xa953fd4eU};
__constant__ uint32_t KR[5] = {0x50a28be6U, 0x5c4dd124U, 0x6d703ef3U, 0x7a6d76e9U, 0x00000000U};
__constant__ uint8_t RL[80] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 4, 13, 1,
                               10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8, 3, 10, 14, 4, 9, 15, 8, 1,
                               2, 7, 0, 6, 13, 11, 5, 12, 1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15,
                               14, 5, 6, 2, 4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13};
__constant__ uint8_t RR[80] = {5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 6, 11, 3, 7,
                               0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2, 15, 5, 1, 3, 7, 14, 6, 9,
                               11, 8, 12, 2, 10, 0, 4, 13, 8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13,
                               9, 7, 10, 14, 12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11};
__constant__ uint8_t SL[80] = {11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8, 7, 6, 8, 13,
                               11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12, 11, 13, 6, 7, 14, 9, 13,
                               15, 14, 8, 13, 6, 5, 12, 7, 5, 11, 12, 14, 15, 14, 15, 9, 8, 9, 14,
                               5, 6, 8, 6, 5, 12, 9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8,
                               5, 6};
__constant__ uint8_t SR[80] = {8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6, 9, 13, 15, 7,
                               12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11, 9, 7, 15, 11, 8, 6, 6, 14,
                               12, 13, 5, 14, 13, 13, 7, 5, 15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12,
                               9, 12, 5, 15, 8, 8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11,
                               11};

__device__ void ripemd160_one_block_32bytes(const uint8_t m32[32], uint8_t out20[20]) {
  uint32_t X[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) X[i] = 0U;
  for (int i = 0; i < 32; ++i) {
    int wi = i >> 2;
    int sh = (i & 3) * 8;
    X[wi] |= uint32_t(m32[i]) << sh;
  }
  {
    int i = 32;
    int wi = i >> 2;
    int sh = (i & 3) * 8;
    X[wi] |= 0x80U << sh;
  }
  X[14] = 32U * 8U;
  X[15] = 0U;

  uint32_t al = 0x67452301U, bl = 0xefcdab89U, cl = 0x98badcfeU, dl = 0x10325476U,
           el = 0xc3d2e1f0U;
  uint32_t ar = al, br = bl, cr = cl, dr = dl, er = el;

#pragma unroll
  for (int j = 0; j < 80; ++j) {
    uint32_t tl, tr;
    if (j < 16) {
      tl = rotl32(al + F1(bl, cl, dl) + X[RL[j]] + KL[0], SL[j]) + el;
      al = el;
      el = dl;
      dl = rotl32(cl, 10);
      cl = bl;
      bl = tl;
      tr = rotl32(ar + F5(br, cr, dr) + X[RR[j]] + KR[0], SR[j]) + er;
      ar = er;
      er = dr;
      dr = rotl32(cr, 10);
      cr = br;
      br = tr;
    } else if (j < 32) {
      tl = rotl32(al + F2(bl, cl, dl) + X[RL[j]] + KL[1], SL[j]) + el;
      al = el;
      el = dl;
      dl = rotl32(cl, 10);
      cl = bl;
      bl = tl;
      tr = rotl32(ar + F4(br, cr, dr) + X[RR[j]] + KR[1], SR[j]) + er;
      ar = er;
      er = dr;
      dr = rotl32(cr, 10);
      cr = br;
      br = tr;
    } else if (j < 48) {
      tl = rotl32(al + F3(bl, cl, dl) + X[RL[j]] + KL[2], SL[j]) + el;
      al = el;
      el = dl;
      dl = rotl32(cl, 10);
      cl = bl;
      bl = tl;
      tr = rotl32(ar + F3(br, cr, dr) + X[RR[j]] + KR[2], SR[j]) + er;
      ar = er;
      er = dr;
      dr = rotl32(cr, 10);
      cr = br;
      br = tr;
    } else if (j < 64) {
      tl = rotl32(al + F4(bl, cl, dl) + X[RL[j]] + KL[3], SL[j]) + el;
      al = el;
      el = dl;
      dl = rotl32(cl, 10);
      cl = bl;
      bl = tl;
      tr = rotl32(ar + F2(br, cr, dr) + X[RR[j]] + KR[3], SR[j]) + er;
      ar = er;
      er = dr;
      dr = rotl32(cr, 10);
      cr = br;
      br = tr;
    } else {
      tl = rotl32(al + F5(bl, cl, dl) + X[RL[j]] + KL[4], SL[j]) + el;
      al = el;
      el = dl;
      dl = rotl32(cl, 10);
      cl = bl;
      bl = tl;
      tr = rotl32(ar + F1(br, cr, dr) + X[RR[j]] + KR[4], SR[j]) + er;
      ar = er;
      er = dr;
      dr = rotl32(cr, 10);
      cr = br;
      br = tr;
    }
  }

  uint32_t t = 0x10325476U + cl + dr;
  cl = 0x98badcfeU + dl + er;
  dl = 0xefcdab89U + el + ar;
  el = 0xc3d2e1f0U + al + br;
  al = 0x67452301U + bl + cr;
  uint32_t H[5] = {al, bl, cl, dl, el};
#pragma unroll
  for (int i = 0; i < 5; ++i) {
    out20[4 * i + 0] = uint8_t((H[i] >> 0) & 0xff);
    out20[4 * i + 1] = uint8_t((H[i] >> 8) & 0xff);
    out20[4 * i + 2] = uint8_t((H[i] >> 16) & 0xff);
    out20[4 * i + 3] = uint8_t((H[i] >> 24) & 0xff);
  }
}

__global__ void hash160_kernel(const uint8_t* __restrict__ pubkeys33,
                               uint8_t* __restrict__ out20, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint8_t* pk = pubkeys33 + size_t(i) * 33;
  uint8_t h256[32];
  sha256_one_block_33bytes(pk, h256);
  uint8_t h160[20];
  ripemd160_one_block_32bytes(h256, h160);
  uint8_t* dst = out20 + size_t(i) * 20;
#pragma unroll
  for (int b = 0; b < 20; ++b) dst[b] = h160[b];
}

bool cuda_compute_hash160_batch(int numKeys, const uint8_t (*pubKeys)[33],
                                uint8_t (*hashResults)[20]) {
  if (!ready || numKeys <= 0) return false;
  if (!ensure_capacity(size_t(numKeys))) return false;

  std::memcpy(h_pk_pinned, pubKeys, size_t(numKeys) * 33);

  if (cudaMemcpyAsync(d_pubkeys, h_pk_pinned, size_t(numKeys) * 33,
                      cudaMemcpyHostToDevice, stream) != cudaSuccess)
    return false;

  int threads = 256;
  int blocks = (numKeys + threads - 1) / threads;
  hash160_kernel<<<blocks, threads, 0, stream>>>(d_pubkeys, d_hashes, numKeys);
  if (cudaGetLastError() != cudaSuccess) return false;

  if (cudaMemcpyAsync(h_hash_pinned, d_hashes, size_t(numKeys) * 20,
                      cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    return false;

  if (cudaStreamSynchronize(stream) != cudaSuccess) return false;

  std::memcpy(hashResults, h_hash_pinned, size_t(numKeys) * 20);
  return true;
}

bool cuda_initialize() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
    std::fprintf(stderr, "[GPU] No CUDA device.\n");
    return false;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "[GPU] cudaSetDevice(0) failed.\n");
    return false;
  }
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
    std::fprintf(stderr, "[GPU] cudaStreamCreate failed.\n");
    return false;
  }
  ready = true;
  return true;
}

void cuda_shutdown() {
  if (d_pubkeys) {
    cudaFree(d_pubkeys);
    d_pubkeys = nullptr;
  }
  if (d_hashes) {
    cudaFree(d_hashes);
    d_hashes = nullptr;
  }
  if (h_pk_pinned) {
    cudaFreeHost(h_pk_pinned);
    h_pk_pinned = nullptr;
  }
  if (h_hash_pinned) {
    cudaFreeHost(h_hash_pinned);
    h_hash_pinned = nullptr;
  }
  if (stream) {
    cudaStreamDestroy(stream);
    stream = 0;
  }
  capacity = 0;
  ready = false;
}

}  // namespace detail
}  // namespace gpu_hash
