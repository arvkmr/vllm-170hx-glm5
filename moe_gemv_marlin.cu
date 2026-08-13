// Small-batch W4A16 MoE GEMV reading vLLM Marlin-packed buffers in place.
//
// Layout (derived + exact-verified, see glm52_moe_decode.py):
//   packed word w (of 128 per 16k x 64n tile-chunk), nibble i:
//     w = q*32 + pn*16 + pk*4 + j        i = i2*4 + i1*2 + i0
//     k = 8*i0 + 2*pk + i2               n = 16*j + 8*i1 + 2*q + pn
//   scales [K/G, N] half: within 64-col group, marlin col = 8*(n%8) + n//8
//   zeros packed [K/G, N/8] int32: logical n at word (2q+pn), nibble (i1*4+j)
//
// Schedule: one block per (token-block slot, 64-col chunk), 128 threads.
// Thread t owns word w=t of every k-tile: its 8 nibbles touch 4 k-values
// {2pk, 2pk+1, 2pk+8, 2pk+9} and 2 n-values {base_n, base_n+8}. The
// unshuffle is therefore free per-thread index arithmetic; the k-loop does
// one coalesced 512B word load per tile plus FMAs from an SMEM-staged x
// tile; the n-reduction happens ONCE at the end via SMEM atomics.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define TOK_MAX 1

__global__ void moe_gemv_marlin_kernel(
    const half* __restrict__ x,          // [M, K]
    const int* __restrict__ w_packed,    // [E, K/16, N*2] int32
    const half* __restrict__ scales,     // [E, K/G, N]
    const int* __restrict__ zp_packed,   // [E, K/G, N/8]
    float* __restrict__ y,               // [num_token_slots, N]
    const int* __restrict__ sorted_token_ids,  // [slots * tok_block]
    const int* __restrict__ expert_ids,        // [slots]
    const float* __restrict__ topk_w,          // [M * top_k]
    int K, int N, int G,
    int tok_block, int num_valid_tokens,
    long stride_we, long stride_se, long stride_ze,
    int mul_routed, int x_row_div)
{
  const int slot = blockIdx.x;
  const int chunk = blockIdx.y;
  const int zoff = blockIdx.z * TOK_MAX;   // token sub-slot
  const int e = expert_ids[slot];
  if (e < 0) return;

  const int t = threadIdx.x;          // word index within chunk, 0..127
  // static per-thread coordinates
  const int q  = t >> 5;
  const int pn = (t >> 4) & 1;
  const int pk = (t >> 2) & 3;
  const int j  = t & 3;
  const int base_n = 16 * j + 2 * q + pn;   // n for i1=0; +8 for i1=1
  const int k0 = 2 * pk;                    // k values: k0, k0+1, k0+8, k0+9

  // token setup
  int toks[TOK_MAX];
  int xrow[TOK_MAX];
  bool tmask[TOK_MAX];
  #pragma unroll
  for (int m = 0; m < TOK_MAX; m++) {
    toks[m] = sorted_token_ids[slot * tok_block + zoff + m];
    tmask[m] = toks[m] < num_valid_tokens;
    xrow[m] = tmask[m] ? toks[m] / x_row_div : 0;
  }

  bool any_valid = false;
  #pragma unroll
  for (int m = 0; m < TOK_MAX; m++) any_valid |= tmask[m];
  if (!any_valid) return;                   // empty sub-slot: free exit

  __shared__ float yn[TOK_MAX][64];         // final n-accumulators
  for (int idx = t; idx < TOK_MAX * 64; idx += blockDim.x)
    ((float*)yn)[idx] = 0.0f;
  __syncthreads();

  const int* wp = w_packed + e * stride_we + (long)chunk * 128;
  const half* sp = scales + e * stride_se + chunk * 64;
  const int* zp = zp_packed + e * stride_ze + chunk * 8;
  const int NW = N * 2;                     // words per k-tile row (N*16/8/4B)
  const int num_kt = K / 16;
  const int gt_tiles = G / 16;

  float acc0[TOK_MAX];   // partial for base_n
  float acc1[TOK_MAX];   // partial for base_n + 8
  #pragma unroll
  for (int m = 0; m < TOK_MAX; m++) { acc0[m] = 0.f; acc1[m] = 0.f; }

  float scale0 = 0.f, scale1 = 0.f, sz0 = 0.f, sz1 = 0.f;

  const int num_groups = num_kt / gt_tiles;
  for (int g = 0; g < num_groups; g++) {
    // scales: marlin col = 8*(n%8) + n/8
    const int n0 = base_n, n1 = base_n + 8;
    scale0 = __half2float(sp[(long)g * N + 8 * (n0 & 7) + (n0 >> 3)]);
    scale1 = __half2float(sp[(long)g * N + 8 * (n1 & 7) + (n1 >> 3)]);
    // zeros: word 2q+pn, nibble i1*4 + j
    const int zw = zp[(long)g * (N >> 3) + 2 * q + pn];
    sz0 = scale0 * (float)((zw >> (4 * j)) & 0xF);
    sz1 = scale1 * (float)((zw >> (4 * (4 + j))) & 0xF);

    // one full scale group (4 k-tiles) per iteration: independent chains
    unsigned wrds[4];
    half2 xp[TOK_MAX][4][2];
    #pragma unroll
    for (int tt = 0; tt < 4; tt++) {
      const int kt = g * 4 + tt;
      wrds[tt] = (unsigned)__ldg(&wp[(long)kt * NW + t]);
      #pragma unroll
      for (int m = 0; m < TOK_MAX; m++) {
        const half* xm = x + (long)xrow[m] * K + kt * 16 + k0;
        xp[m][tt][0] = __ldg((const half2*)(xm));
        xp[m][tt][1] = __ldg((const half2*)(xm + 8));
      }
    }
    #pragma unroll
    for (int tt = 0; tt < 4; tt++) {
      const unsigned wrd = wrds[tt];
      #pragma unroll
      for (int i1 = 0; i1 < 2; i1++) {
        const float sc = i1 ? scale1 : scale0;
        const float szv = i1 ? sz1 : sz0;
        float wv[4];
        #pragma unroll
        for (int i2 = 0; i2 < 2; i2++) {
          #pragma unroll
          for (int i0 = 0; i0 < 2; i0++) {
            const int i = i2 * 4 + i1 * 2 + i0;
            const float nv = (float)((wrd >> (4 * i)) & 0xF);
            wv[i0 * 2 + i2] = nv * sc - szv;
          }
        }
        #pragma unroll
        for (int m = 0; m < TOK_MAX; m++) {
          float a = __half2float(__low2half(xp[m][tt][0]))  * wv[0]
                  + __half2float(__high2half(xp[m][tt][0])) * wv[1]
                  + __half2float(__low2half(xp[m][tt][1]))  * wv[2]
                  + __half2float(__high2half(xp[m][tt][1])) * wv[3];
          if (i1) acc1[m] += a; else acc0[m] += a;
        }
      }
    }
  }

  // reduce by n: each thread contributes its 2 n-partials
  #pragma unroll
  for (int m = 0; m < TOK_MAX; m++) {
    atomicAdd(&yn[m][base_n], acc0[m]);
    atomicAdd(&yn[m][base_n + 8], acc1[m]);
  }
  __syncthreads();

  // write out
  for (int idx = t; idx < TOK_MAX * 64; idx += blockDim.x) {
    const int m = idx >> 6, n = idx & 63;
    if (!tmask[m]) continue;
    float v = yn[m][n];
    if (mul_routed) v *= topk_w[toks[m]];
    y[(long)toks[m] * N + chunk * 64 + n] = v;
  }
}

extern "C" void moe_gemv_marlin_launch(
    const void* x, const int* w_packed, const void* scales,
    const int* zp_packed, float* y, const int* sorted_token_ids,
    const int* expert_ids, const float* topk_w,
    int K, int N, int G, int tok_block, int num_valid_tokens,
    long stride_we, long stride_se, long stride_ze,
    int mul_routed, int x_row_div, int slots, void* stream)
{
  dim3 grid(slots, N / 64, tok_block / TOK_MAX);
  dim3 block(128);
  moe_gemv_marlin_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
      (const half*)x, w_packed, (const half*)scales, zp_packed, y,
      sorted_token_ids, expert_ids, topk_w, K, N, G, tok_block,
      num_valid_tokens, stride_we, stride_se, stride_ze,
      mul_routed, x_row_div);
}
