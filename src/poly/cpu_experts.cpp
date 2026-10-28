// src/poly/cpu_experts.cpp - see the header.
#include "cpu_experts.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "../src/ggml-common.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <mutex>
#include <numeric>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::poly {
namespace {

// Rows of gate and up, and of down, a task.  With fewer tasks than several a thread, the last round left threads
// idle at the end of every layer: 64 rows gave Tiel's 512-row experts 8 tasks each, and its down products 32 tasks.
constexpr int kRowsPerTask = 32;
// 32 and not 16: with 16 a task read 10 KB of each expert's down, too little for the hardware prefetcher to get
// going (GLM 5.3 Flash Q2_K, 4 experts a call, i7-13700K, DDR5-4000: down 31.6 GB/s; with 32 and kPrefetchRows 2,
// 45 GB/s)
constexpr int kRowsPerDownTask = 32;
// Each row's dot products first ask for the row this many rows on in the same task, which the hardware prefetcher
// does not see coming across the rows of several experts
constexpr int kPrefetchRows = 4;
inline int spins_before_sleep() {
    static const int spins = [] {
        const char* v = std::getenv("STRATA_CPU_SPIN");
        return (v && std::atoi(v) > 0) ? std::atoi(v) : 200000;   // ~3-5 ms on Alder Lake to bridge GPU attention without sleeping
    }();
    return spins;
}
constexpr int kMaxMultiTok = 8;

const ggml_type_traits_cpu* traits(int type) { return ggml_get_type_traits_cpu((ggml_type) type); }

void init_once() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_cpu_init(); });
}

thread_local bool in_task = false;   // this thread is running a task of a batch

inline void prefetch_row(const uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; i += 64) _mm_prefetch((const char*) p + i, _MM_HINT_T0);
}

// Bit-exact AVX2 implementation of ggml's quantize_row_q8_K_ref (which is scalar in ggml-cpu on x86).
void q8k_quant_avx2(const float* x, void* vy, int64_t k) {
    block_q8_K* y = (block_q8_K*) vy;
    const int64_t nb = k / QK_K;
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 magic = _mm256_set1_ps(12582912.f);
    const __m256i mant = _mm256_set1_epi32(0x007fffff), off = _mm256_set1_epi32(0x00400000);
    const __m256i c127 = _mm256_set1_epi32(127);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int64_t i = 0; i < nb; ++i, x += QK_K) {
        __m256 m = _mm256_setzero_ps();
        for (int j = 0; j < QK_K; j += 8) m = _mm256_max_ps(_mm256_and_ps(_mm256_loadu_ps(x + j), absm), m);
        __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        h = _mm_max_ps(h, _mm_movehl_ps(h, h));
        h = _mm_max_ss(h, _mm_movehdup_ps(h));
        const float amax = _mm_cvtss_f32(h);
        if (!amax) {
            y[i].d = 0;
            std::memset(y[i].qs, 0, QK_K);
            continue;
        }
        float mx = 0.f;
        const __m256 va = _mm256_set1_ps(amax);
        for (int j = 0; j < QK_K; j += 8) {
            const int msk =
                _mm256_movemask_ps(_mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x + j), absm), va, _CMP_EQ_OQ));
            if (msk) {
                int b = 0;
                while (!((msk >> b) & 1)) ++b;
                mx = x[j + b];
                break;
            }
        }
        const float iscale = -127.f / mx;
        const __m256 vis = _mm256_set1_ps(iscale);
        for (int j = 0; j < QK_K; j += 32) {
            __m256i v[4];
            for (int q = 0; q < 4; ++q) {
                __m256 p = _mm256_mul_ps(vis, _mm256_loadu_ps(x + j + 8 * q));
                const __m256 t = _mm256_add_ps(p, magic);
                v[q] = _mm256_min_epi32(_mm256_sub_epi32(_mm256_and_si256(_mm256_castps_si256(t), mant), off), c127);
                v[q] = _mm256_srai_epi32(_mm256_slli_epi32(v[q], 24), 24);
            }
            for (int q = 0; q < 2; ++q) {
                const __m256i s = _mm256_add_epi32(v[2 * q], v[2 * q + 1]);
                __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
                s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
                y[i].bsums[j / 16 + q] = (int16_t) _mm_cvtsi128_si32(s4);
            }
            const __m256i p16a = _mm256_packs_epi32(v[0], v[1]), p16b = _mm256_packs_epi32(v[2], v[3]);
            const __m256i p8 = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(p16a, p16b), perm);
            _mm256_storeu_si256((__m256i*) (y[i].qs + j), p8);
        }
        y[i].d = 1.f / iscale;
    }
}

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}

inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}

inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

alignas(32) const uint8_t k_shuffle_k4[256] = {
     0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
     4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
     8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
    12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15};

inline __m256i get_scale_shuffle_k4(int i) { return _mm256_loadu_si256((const __m256i*) k_shuffle_k4 + i); }

alignas(16) const uint8_t k_shuffle_q6[128] = {
     0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
     4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
     6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
     8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
    10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,
    12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,
    14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15};

inline __m128i get_scale_shuffle_q6(int i) { return _mm_loadu_si128((const __m128i*) k_shuffle_q6 + i); }

// Fused single-token gate + up Q4_K x Q8_K dot product: loads and reduces y (block_q8_K) once per block while
// executing the exact same AVX2 instruction chain as ggml_vec_dot_q4_K_q8_K for both gate and up.
void q4k_gate_up_1(const block_q4_K* gx, const block_q4_K* ux, int nb, const block_q8_K* y, float& g_out, float& u_out) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 g_acc = _mm256_setzero_ps(), u_acc = _mm256_setzero_ps();
    __m128 g_acc_m = _mm_setzero_ps(), u_acc_m = _mm_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float yd = y[i].d;
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

        uint32_t g_utmp[4], u_utmp[4];
        std::memcpy(g_utmp, gx[i].scales, 12);
        g_utmp[3] = ((g_utmp[2] >> 4) & kmask2) | (((g_utmp[1] >> 6) & kmask3) << 4);
        const uint32_t g_uaux = g_utmp[1] & kmask1;
        g_utmp[1] = (g_utmp[2] & kmask2) | (((g_utmp[0] >> 6) & kmask3) << 4);
        g_utmp[2] = g_uaux;
        g_utmp[0] &= kmask1;
        const __m256i g_ms = _mm256_cvtepu8_epi16(_mm_set_epi32((int) g_utmp[3], (int) g_utmp[2], (int) g_utmp[1], (int) g_utmp[0]));
        const __m128i g_mins = _mm256_extracti128_si256(g_ms, 1);
        const __m128i g_sc128 = _mm256_extracti128_si256(g_ms, 0);
        const __m256i g_scales = _mm256_insertf128_si256(_mm256_castsi128_si256(g_sc128), g_sc128, 1);
        g_acc_m = _mm_fmadd_ps(_mm_set1_ps(-yd * half_at(&gx[i], 1)), _mm_cvtepi32_ps(_mm_madd_epi16(g_mins, q8s)), g_acc_m);

        std::memcpy(u_utmp, ux[i].scales, 12);
        u_utmp[3] = ((u_utmp[2] >> 4) & kmask2) | (((u_utmp[1] >> 6) & kmask3) << 4);
        const uint32_t u_uaux = u_utmp[1] & kmask1;
        u_utmp[1] = (u_utmp[2] & kmask2) | (((u_utmp[0] >> 6) & kmask3) << 4);
        u_utmp[2] = u_uaux;
        u_utmp[0] &= kmask1;
        const __m256i u_ms = _mm256_cvtepu8_epi16(_mm_set_epi32((int) u_utmp[3], (int) u_utmp[2], (int) u_utmp[1], (int) u_utmp[0]));
        const __m128i u_mins = _mm256_extracti128_si256(u_ms, 1);
        const __m128i u_sc128 = _mm256_extracti128_si256(u_ms, 0);
        const __m256i u_scales = _mm256_insertf128_si256(_mm256_castsi128_si256(u_sc128), u_sc128, 1);
        u_acc_m = _mm_fmadd_ps(_mm_set1_ps(-yd * half_at(&ux[i], 1)), _mm_cvtepi32_ps(_mm_madd_epi16(u_mins, q8s)), u_acc_m);

        const uint8_t* gq4 = gx[i].qs;
        const uint8_t* uq4 = ux[i].qs;
        const int8_t* q8 = y[i].qs;
        __m256i g_sumi = _mm256_setzero_si256(), u_sumi = _mm256_setzero_si256();
        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i shuf_l = get_scale_shuffle_k4(2 * j + 0);
            const __m256i shuf_h = get_scale_shuffle_k4(2 * j + 1);
            const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
            const __m256i q8h = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            q8 += 64;

            const __m256i gbits = _mm256_loadu_si256((const __m256i*) gq4);
            gq4 += 32;
            __m256i gp16l = _mm256_maddubs_epi16(_mm256_and_si256(gbits, m4), q8l);
            gp16l = _mm256_madd_epi16(_mm256_shuffle_epi8(g_scales, shuf_l), gp16l);
            __m256i gp16h = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(gbits, 4), m4), q8h);
            gp16h = _mm256_madd_epi16(_mm256_shuffle_epi8(g_scales, shuf_h), gp16h);
            g_sumi = _mm256_add_epi32(g_sumi, _mm256_add_epi32(gp16l, gp16h));

            const __m256i ubits = _mm256_loadu_si256((const __m256i*) uq4);
            uq4 += 32;
            __m256i up16l = _mm256_maddubs_epi16(_mm256_and_si256(ubits, m4), q8l);
            up16l = _mm256_madd_epi16(_mm256_shuffle_epi8(u_scales, shuf_l), up16l);
            __m256i up16h = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(ubits, 4), m4), q8h);
            up16h = _mm256_madd_epi16(_mm256_shuffle_epi8(u_scales, shuf_h), up16h);
            u_sumi = _mm256_add_epi32(u_sumi, _mm256_add_epi32(up16l, up16h));
        }
        g_acc = _mm256_fmadd_ps(_mm256_set1_ps(yd * half_at(&gx[i], 0)), _mm256_cvtepi32_ps(g_sumi), g_acc);
        u_acc = _mm256_fmadd_ps(_mm256_set1_ps(yd * half_at(&ux[i], 0)), _mm256_cvtepi32_ps(u_sumi), u_acc);
    }
    g_acc_m = _mm_add_ps(g_acc_m, _mm_movehl_ps(g_acc_m, g_acc_m));
    g_acc_m = _mm_add_ss(g_acc_m, _mm_movehdup_ps(g_acc_m));
    g_out = hsum_float_8(g_acc) + _mm_cvtss_f32(g_acc_m);

    u_acc_m = _mm_add_ps(u_acc_m, _mm_movehl_ps(u_acc_m, u_acc_m));
    u_acc_m = _mm_add_ss(u_acc_m, _mm_movehdup_ps(u_acc_m));
    u_out = hsum_float_8(u_acc) + _mm_cvtss_f32(u_acc_m);
}

// Multi-token Q4_K x Q8_K dot product: unpacks each Q4_K block once across `nt` (<= kMaxMultiTok) tokens,
// bit-exact with ggml_vec_dot_q4_K_q8_K.
void q4k_dot_multi(const block_q4_K* x, int nb, const uint8_t* const* act, int nt, float* res) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc[kMaxMultiTok];
    __m128 acc_m[kMaxMultiTok];
    for (int t = 0; t < nt; ++t) {
        acc[t] = _mm256_setzero_ps();
        acc_m[t] = _mm_setzero_ps();
    }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales =
            _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        __m256i scale_l[4], scale_h[4], q4l[4], q4h[4];
        const uint8_t* q4 = x[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_l[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_h[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            q4l[j] = _mm256_and_si256(q4bits, m4);
            q4h[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            acc_m[t] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[t]);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16l = _mm256_maddubs_epi16(q4l[j], q8l);
                p16l = _mm256_madd_epi16(scale_l[j], p16l);
                const __m256i q8h = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16h = _mm256_maddubs_epi16(q4h[j], q8h);
                p16h = _mm256_madd_epi16(scale_h[j], p16h);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        __m128 m = acc_m[t];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        res[t] = hsum_float_8(acc[t]) + _mm_cvtss_f32(m);
    }
}

// Multi-token Q6_K x Q8_K dot product: unpacks each Q6_K block once across `nt` (<= kMaxMultiTok) tokens,
// bit-exact with ggml_vec_dot_q6_K_q8_K.
void q6k_dot_multi(const block_q6_K* x, int nb, const uint8_t* const* act, int nt, float* res) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i m192 = _mm256_set1_epi8(-64);
    const __m256i m15 = _mm256_set1_epi8(15);
    __m256 acc[kMaxMultiTok];
    for (int t = 0; t < nt; ++t) acc[t] = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float xd = h2f(x[i].d);
        const uint8_t* q4 = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const __m128i scales = _mm_loadu_si128((const __m128i*) x[i].scales);
        const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
        __m256i q4_0[2], q4_1[2], q4_2[2], q4_3[2];
        __m256i sc_0[2], sc_1[2], sc_2[2], sc_3[2];
        int is = 0;
        for (int j = 0; j < QK_K / 128; ++j) {
            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*) q4);
            q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*) qh);
            qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m12), 2);
            const __m256i q4h_2 = _mm256_and_si256(q4bitsH, m48);
            const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, m192), 2);

            q4_0[j] = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
            q4_1[j] = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
            q4_2[j] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
            q4_3[j] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

            sc_0[j] = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(is + 0)));
            sc_1[j] = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(is + 1)));
            sc_2[j] = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(is + 2)));
            sc_3[j] = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(is + 3)));
            is += 4;
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 128; ++j) {
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8_3 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;

                __m256i p16_0 = _mm256_madd_epi16(sc_0[j], _mm256_maddubs_epi16(q4_0[j], q8_0));
                __m256i p16_1 = _mm256_madd_epi16(sc_1[j], _mm256_maddubs_epi16(q4_1[j], q8_1));
                __m256i p16_2 = _mm256_madd_epi16(sc_2[j], _mm256_maddubs_epi16(q4_2[j], q8_2));
                __m256i p16_3 = _mm256_madd_epi16(sc_3[j], _mm256_maddubs_epi16(q4_3[j], q8_3));

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
            }
            sumi = _mm256_sub_epi32(sumi, q8sclsub);
            acc[t] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]);
}

// Fused 2-row single-token gate + up Q4_K x Q8_K dot product: loads y (block_q8_K) once per block for 4 dot products
// (gate row 0, up row 0, gate row 1, up row 1), bit-exact with ggml_vec_dot_q4_K_q8_K.
void q4k_gate_up_2rows(const block_q4_K* gx0, const block_q4_K* ux0, const block_q4_K* gx1, const block_q4_K* ux1,
                       int nb, const block_q8_K* y, float& g0_out, float& u0_out, float& g1_out, float& u1_out) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const block_q4_K* xs[4] = {gx0, ux0, gx1, ux1};
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    __m128 acc_m[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
    for (int i = 0; i < nb; ++i) {
        const float yd = y[i].d;
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        __m256i scales[4], sumi[4];
        const uint8_t* q4[4];
        for (int k = 0; k < 4; ++k) {
            const block_q4_K& b = xs[k][i];
            uint32_t utmp[4];
            std::memcpy(utmp, b.scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
            const __m256i ms = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]));
            const __m128i mins = _mm256_extracti128_si256(ms, 1);
            const __m128i sc128 = _mm256_extracti128_si256(ms, 0);
            scales[k] = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
            acc_m[k] = _mm_fmadd_ps(_mm_set1_ps(-yd * half_at(&b, 1)), _mm_cvtepi32_ps(_mm_madd_epi16(mins, q8s)), acc_m[k]);
            q4[k] = b.qs;
            sumi[k] = _mm256_setzero_si256();
        }
        const int8_t* q8 = y[i].qs;
        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i shuf_l = get_scale_shuffle_k4(2 * j + 0);
            const __m256i shuf_h = get_scale_shuffle_k4(2 * j + 1);
            const __m256i q8l = _mm256_loadu_si256((const __m256i*) q8);
            const __m256i q8h = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            q8 += 64;
            for (int k = 0; k < 4; ++k) {
                const __m256i bits = _mm256_loadu_si256((const __m256i*) q4[k]);
                q4[k] += 32;
                __m256i p16l = _mm256_maddubs_epi16(_mm256_and_si256(bits, m4), q8l);
                p16l = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[k], shuf_l), p16l);
                __m256i p16h = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(bits, 4), m4), q8h);
                p16h = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[k], shuf_h), p16h);
                sumi[k] = _mm256_add_epi32(sumi[k], _mm256_add_epi32(p16l, p16h));
            }
        }
        for (int k = 0; k < 4; ++k) {
            acc[k] = _mm256_fmadd_ps(_mm256_set1_ps(yd * half_at(&xs[k][i], 0)), _mm256_cvtepi32_ps(sumi[k]), acc[k]);
        }
    }
    float out[4];
    for (int k = 0; k < 4; ++k) {
        __m128 m = acc_m[k];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        out[k] = hsum_float_8(acc[k]) + _mm_cvtss_f32(m);
    }
    g0_out = out[0];
    u0_out = out[1];
    g1_out = out[2];
    u1_out = out[3];
}

// Fused 4-row single-token Q4_K x Q8_K dot product for down-projection: loads y once across 4 contiguous rows of down.
void q4k_down_4rows(const block_q4_K* x0, const block_q4_K* x1, const block_q4_K* x2, const block_q4_K* x3,
                    int nb, const block_q8_K* y, float* res) {
    float g0, u0, g1, u1;
    q4k_gate_up_2rows(x0, x1, x2, x3, nb, y, g0, u0, g1, u1);
    res[0] = g0;
    res[1] = u0;
    res[2] = g1;
    res[3] = u1;
}

// Fused 4-row single-token Q6_K x Q8_K dot product for down-projection: loads y once across 4 contiguous rows of down,
// bit-exact with ggml_vec_dot_q6_K_q8_K.
void q6k_down_4rows(const block_q6_K* x0, const block_q6_K* x1, const block_q6_K* x2, const block_q6_K* x3,
                    int nb, const block_q8_K* y, float* res) {
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i m12 = _mm256_set1_epi8(12);
    const __m256i m48 = _mm256_set1_epi8(48);
    const __m256i m192 = _mm256_set1_epi8(-64);
    const __m256i m15 = _mm256_set1_epi8(15);
    const block_q6_K* xs[4] = {x0, x1, x2, x3};
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int i = 0; i < nb; ++i) {
        const float yd = y[i].d;
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        __m256i q8_0[2], q8_1[2], q8_2[2], q8_3[2];
        const int8_t* q8 = y[i].qs;
        for (int j = 0; j < QK_K / 128; ++j) {
            q8_0[j] = _mm256_loadu_si256((const __m256i*) q8);
            q8 += 32;
            q8_1[j] = _mm256_loadu_si256((const __m256i*) q8);
            q8 += 32;
            q8_2[j] = _mm256_loadu_si256((const __m256i*) q8);
            q8 += 32;
            q8_3[j] = _mm256_loadu_si256((const __m256i*) q8);
            q8 += 32;
        }
        for (int k = 0; k < 4; ++k) {
            const block_q6_K& b = xs[k][i];
            const float d = yd * h2f(b.d);
            const uint8_t* q4 = b.ql;
            const uint8_t* qh = b.qh;
            const __m128i scales = _mm_loadu_si128((const __m128i*) b.scales);
            const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
            const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
            __m256i sumi = _mm256_setzero_si256();
            int is = 0;
            for (int j = 0; j < QK_K / 128; ++j) {
                const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*) q4);
                q4 += 32;
                const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*) q4);
                q4 += 32;
                const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*) qh);
                qh += 32;

                const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m3), 4);
                const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m12), 2);
                const __m256i q4h_2 = _mm256_and_si256(q4bitsH, m48);
                const __m256i q4h_3 = _mm256_srli_epi16(_mm256_and_si256(q4bitsH, m192), 2);

                const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
                const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
                const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m15), q4h_2);
                const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m15), q4h_3);

                const __m256i sc_0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(is + 0)));
                const __m256i sc_1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(is + 1)));
                const __m256i sc_2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(is + 2)));
                const __m256i sc_3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle_q6(is + 3)));
                is += 4;

                __m256i p16_0 = _mm256_madd_epi16(sc_0, _mm256_maddubs_epi16(q4_0, q8_0[j]));
                __m256i p16_1 = _mm256_madd_epi16(sc_1, _mm256_maddubs_epi16(q4_1, q8_1[j]));
                __m256i p16_2 = _mm256_madd_epi16(sc_2, _mm256_maddubs_epi16(q4_2, q8_2[j]));
                __m256i p16_3 = _mm256_madd_epi16(sc_3, _mm256_maddubs_epi16(q4_3, q8_3[j]));

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
            }
            sumi = _mm256_sub_epi32(sumi, q8sclsub);
            acc[k] = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc[k]);
        }
    }
    for (int k = 0; k < 4; ++k) res[k] = hsum_float_8(acc[k]);
}

// Fused 4-row single-token Q5_K x Q8_K dot product (also used for 2-row gate+up), bit-exact with ggml_vec_dot_q5_K_q8_K.
void q5k_down_4rows(const block_q5_K* x0, const block_q5_K* x1, const block_q5_K* x2, const block_q5_K* x3,
                    int nb, const block_q8_K* y, float* res) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone = _mm256_set1_epi8(1);
    const block_q5_K* xs[4] = {x0, x1, x2, x3};
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    float summs[4] = {0.f, 0.f, 0.f, 0.f};
    for (int i = 0; i < nb; ++i) {
        const float yd = y[i].d;
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        __m256i scales[4], hbits[4], hmask[4], sumi[4];
        const uint8_t* q5[4];
        for (int k = 0; k < 4; ++k) {
            const block_q5_K& b = xs[k][i];
            const float dmin = -yd * half_at(&b, 1);
            uint32_t utmp[4];
            std::memcpy(utmp, b.scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
            const __m256i ms = _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]));
            const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(ms, 1), q8s);
            const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs[k] += dmin * _mm_extract_epi32(hsum, 0);
            const __m128i sc128 = _mm256_extracti128_si256(ms, 0);
            scales[k] = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
            hbits[k] = _mm256_loadu_si256((const __m256i*) b.qh);
            hmask[k] = mone;
            q5[k] = b.qs;
            sumi[k] = _mm256_setzero_si256();
        }
        const int8_t* q8 = y[i].qs;
        int bit = 0;
        for (int j = 0; j < QK_K / 64; ++j) {
            const __m256i shuf_0 = get_scale_shuffle_k4(2 * j + 0);
            const __m256i shuf_1 = get_scale_shuffle_k4(2 * j + 1);
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) q8);
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) (q8 + 32));
            q8 += 64;
            const int b0 = bit++, b1 = bit++;
            for (int k = 0; k < 4; ++k) {
                const __m256i q5bits = _mm256_loadu_si256((const __m256i*) q5[k]);
                q5[k] += 32;
                const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
                const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits[k], hmask[k]), b0), 4);
                const __m256i q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
                hmask[k] = _mm256_slli_epi16(hmask[k], 1);

                const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
                const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits[k], hmask[k]), b1), 4);
                const __m256i q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
                hmask[k] = _mm256_slli_epi16(hmask[k], 1);

                __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
                __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);
                p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[k], shuf_0), p16_0);
                p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[k], shuf_1), p16_1);
                sumi[k] = _mm256_add_epi32(sumi[k], _mm256_add_epi32(p16_0, p16_1));
            }
        }
        for (int k = 0; k < 4; ++k) {
            acc[k] = _mm256_fmadd_ps(_mm256_set1_ps(yd * half_at(&xs[k][i], 0)), _mm256_cvtepi32_ps(sumi[k]), acc[k]);
        }
    }
    for (int k = 0; k < 4; ++k) res[k] = hsum_float_8(acc[k]) + summs[k];
}

// Multi-token Q5_K x Q8_K dot product: unpacks each Q5_K block once across `nt` (<= kMaxMultiTok) tokens,
// bit-exact with ggml_vec_dot_q5_K_q8_K.
void q5k_dot_multi(const block_q5_K* x, int nb, const uint8_t* const* act, int nt, float* res) {
    static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone = _mm256_set1_epi8(1);
    __m256 acc[kMaxMultiTok];
    float summs[kMaxMultiTok];
    for (int t = 0; t < nt; ++t) {
        acc[t] = _mm256_setzero_ps();
        summs[t] = 0.f;
    }
    for (int i = 0; i < nb; ++i) {
        const float xd = half_at(&x[i], 0), xdmin = half_at(&x[i], 1);
        uint32_t utmp[4];
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;
        const __m256i mins_and_scales =
            _mm256_cvtepu8_epi16(_mm_set_epi32((int) utmp[3], (int) utmp[2], (int) utmp[1], (int) utmp[0]));
        const __m128i mins = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        const __m256i hbits = _mm256_loadu_si256((const __m256i*) x[i].qh);
        __m256i hmask = mone;
        __m256i scale_0[4], scale_1[4], q5_0[4], q5_1[4];
        const uint8_t* q5 = x[i].qs;
        int bit = 0;
        for (int j = 0; j < QK_K / 64; ++j) {
            scale_0[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            scale_1[j] = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q5bits = _mm256_loadu_si256((const __m256i*) q5);
            q5 += 32;
            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5_0[j] = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            q5_1[j] = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);
        }
        for (int t = 0; t < nt; ++t) {
            const block_q8_K* y = (const block_q8_K*) act[t] + i;
            const float d = y->d * xd;
            const float dmin = -y->d * xdmin;
            const int8_t* q8 = y->qs;
            const __m256i q8sums = _mm256_loadu_si256((const __m256i*) y->bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod = _mm_madd_epi16(mins, q8s);
            const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs[t] += dmin * _mm_extract_epi32(hsum, 0);
            __m256i sumi = _mm256_setzero_si256();
            for (int j = 0; j < QK_K / 64; ++j) {
                const __m256i q8_0 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8);
                q8 += 32;
                __m256i p16_0 = _mm256_madd_epi16(scale_0[j], _mm256_maddubs_epi16(q5_0[j], q8_0));
                __m256i p16_1 = _mm256_madd_epi16(scale_1[j], _mm256_maddubs_epi16(q5_1[j], q8_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            }
            acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc[t]);
        }
    }
    for (int t = 0; t < nt; ++t) res[t] = hsum_float_8(acc[t]) + summs[t];
}

}  // namespace

Workers::Workers(int threads) {
#if defined(_WIN32)
    // Pin the 6-core compute pool to physical P-cores 0..5 (logical processors 0, 2, 4, 6, 8, 10 on i5-12400).
    // Auxiliary pools (e.g. 4-thread stagers_) are left unpinned so they don't re-pin the caller thread.
    if (threads == 6) SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << 0);
#endif
    for (int i = 1; i < threads; ++i) {
        th_.emplace_back([this, i, threads] {
#if defined(_WIN32)
            if (threads == 6) {
                DWORD_PTR mask = (DWORD_PTR) 1 << ((i % 6) * 2);
                SetThreadAffinityMask(GetCurrentThread(), mask);
            }
#endif
            loop();
        });
    }
}

Workers::~Workers() {
    quit_.store(true);
    posted_.fetch_add(1);
    posted_.notify_all();
    for (auto& t : th_) t.join();
}

// Whether it ran a task.  next is read before it is counted up, so it stops near n however often a worker comes
// back to a batch whose tasks are taken.
bool Workers::drain(Batch& b) {
    bool ran = false;
    while (b.next.load(std::memory_order_relaxed) < b.n) {
        const int i = b.next.fetch_add(1, std::memory_order_relaxed);
        if (i >= b.n) break;
        in_task = true;
        (*b.fn)(i);
        in_task = false;
        b.left.fetch_sub(1, std::memory_order_release);
        ran = true;
    }
    return ran;
}

// A worker with nothing to do only reads posted_: a write by each of them, every turn of the loop, kept run() from
// ever seeing users_ at zero for long.  When a batch has been posted since the worker last looked, it counts itself
// in users_ and reads cur_ again before it uses it, so run() can tell when nobody holds its Batch any more.
//
// A batch stays posted after its last task is taken, until run() takes it down, and run() cannot while it has no
// CPU.  That happens with more threads than the system runs: a laptop that parks cores under load makes that of
// any count.  So a worker that found no task in what is posted leaves it alone: it pauses, then sleeps, until
// posted_ moves.  Going round the finished batch instead, the workers kept run() off the CPU for the rest of
// their time slices, every batch.
void Workers::loop() {
    int idle = 0;
    uint32_t seen = posted_.load() - 1;   // the count at which this worker last found nothing to do
    while (!quit_.load(std::memory_order_relaxed)) {
        const uint32_t now = posted_.load(std::memory_order_acquire);
        if (now != seen) {
            // cur_ is set before posted_ counts: the batch of `now` is there, or a later one, or it is done
            bool ran = false;
            if (cur_.load(std::memory_order_acquire)) {
                users_.fetch_add(1);
                Batch* b = cur_.load();
                if (b) ran = drain(*b);
                users_.fetch_sub(1);
            }
            if (ran) {
                idle = 0;
                continue;
            }
            seen = now;
        }
        if (++idle < spins_before_sleep()) {
            _mm_pause();
            continue;
        }
        // run() counts posted_ up before it reads sleepers_, and wait() returns at once when posted_ has moved
        sleepers_.fetch_add(1);
        if (!quit_.load()) posted_.wait(seen);
        sleepers_.fetch_sub(1);
        idle = 0;
    }
}

void Workers::run(int n, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    // One task is nothing to share out.  And a task that shares out work of its own does it alone: there is one
    // batch at a time, and a worker that waited below for the others to let go of theirs would wait for itself.
    if (n == 1 || in_task) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    const std::lock_guard<std::mutex> one(one_);
    Batch b;
    b.fn = &fn;
    b.n = n;
    b.left.store(n);
    cur_.store(&b);
    posted_.fetch_add(1);
    if (sleepers_.load() > 0) posted_.notify_all();
    drain(b);
    cur_.store(nullptr);   // every task is taken: a worker that comes now has nothing to find
    int spin1 = 0;
    while (b.left.load(std::memory_order_acquire) != 0) {
        if (++spin1 > 2000) std::this_thread::yield();
        else _mm_pause();
    }
    int spin2 = 0;
    while (users_.load() != 0) {
        if (++spin2 > 2000) std::this_thread::yield();
        else _mm_pause();
    }
}

bool cpu_expert_types_ok(int gu_type, int d_type, int n_embd, int n_ff, std::string& err) {
    init_once();
    for (const int type : {gu_type, d_type}) {
        const ggml_type_traits_cpu* t = traits(type);
        const ggml_type_traits_cpu* a = t && t->vec_dot ? traits(t->vec_dot_type) : nullptr;
        if (a == nullptr || a->from_float == nullptr) {
            err = "ggml-cpu has no dot product for tensor type " + std::to_string(type);
            return false;
        }
        const int n = type == gu_type ? n_embd : n_ff;
        if (n % ggml_blck_size((ggml_type) type) || n % ggml_blck_size(t->vec_dot_type)) {
            err = "a row of " + std::to_string(n) + " values is not whole blocks of tensor type " + std::to_string(type);
            return false;
        }
    }
    return true;
}

void cpu_experts(Workers& workers, const CpuExpert* experts, int n, int rows, int gu_type, int u_type, int d_type,
                 int n_embd, int n_ff, const float* x, float limit, bool gelu, float* out,
                 const std::function<void()>* meanwhile) {
    init_once();
    const ggml_type_traits_cpu* tg = traits(gu_type);
    const ggml_type_traits_cpu* tu = traits(u_type);
    const ggml_type_traits_cpu* td = traits(d_type);
    const size_t gu_row = ggml_row_size((ggml_type) gu_type, n_embd);
    const size_t u_row = ggml_row_size((ggml_type) u_type, n_embd);
    const size_t d_row = ggml_row_size((ggml_type) d_type, n_ff);
    const size_t a_bytes = ggml_row_size(tg->vec_dot_type, n_embd);
    const size_t au_bytes = ggml_row_size(tu->vec_dot_type, n_embd);
    const size_t h_bytes = ggml_row_size(td->vec_dot_type, n_ff);
    const bool own_act = tu->vec_dot_type != tg->vec_dot_type;   // up's dot product takes another quantization

    // The workers get these through the pointers below: a thread_local named in their code would be their own.
    static thread_local std::vector<uint8_t> act, act_u, hq;
    static thread_local std::vector<float> h;
    static thread_local std::vector<int> by, run;
    act.resize((size_t) rows * a_bytes);
    if (own_act) act_u.resize((size_t) rows * au_bytes);
    hq.resize((size_t) n * h_bytes);
    h.resize((size_t) n * n_ff);
    uint8_t* actp = act.data();
    const ggml_from_float_t act_from =
        tg->vec_dot_type == GGML_TYPE_Q8_K ? &q8k_quant_avx2 : traits(tg->vec_dot_type)->from_float;
    workers.run(rows, [&](int t) { act_from(x + (size_t) t * n_embd, actp + (size_t) t * a_bytes, n_embd); });
    uint8_t* actup = actp;
    if (own_act) {
        actup = act_u.data();
        const auto au_from = traits(tu->vec_dot_type)->from_float;
        workers.run(rows, [&](int t) { au_from(x + (size_t) t * n_embd, actup + (size_t) t * au_bytes, n_embd); });
    }
    const size_t au_step = own_act ? au_bytes : a_bytes;

    // The uses of one expert next to each other (by), and where each expert's run of them starts (run).
    by.resize((size_t) n);
    std::iota(by.begin(), by.end(), 0);
    std::stable_sort(by.begin(), by.end(), [&](int a, int b) { return experts[a].gate < experts[b].gate; });
    run.clear();
    for (int i = 0; i < n; ++i)
        if (i == 0 || experts[by[(size_t) i]].gate != experts[by[(size_t) i - 1]].gate) run.push_back(i);
    const int n_run = (int) run.size();
    run.push_back(n);

    float* hp = h.data();
    const int* byp = by.data();
    const int* runp = run.data();
    const int gu_tasks = (n_ff + kRowsPerTask - 1) / kRowsPerTask;
    const int first = meanwhile ? 1 : 0;   // task 0, so it is taken at once
    const int nb_gu = n_embd / QK_K;
    workers.run(first + n_run * gu_tasks, [&](int t) {
        if (t < first) return (*meanwhile)();
        t -= first;
        const int i0 = runp[t / gu_tasks], i1 = runp[t / gu_tasks + 1];
        const CpuExpert& e = experts[byp[i0]];
        const int r0 = (t % gu_tasks) * kRowsPerTask, r1 = std::min(n_ff, r0 + kRowsPerTask);
        if (gu_type == GGML_TYPE_Q4_K || gu_type == GGML_TYPE_Q5_K || gu_type == GGML_TYPE_Q6_K) {
            if (i1 - i0 == 1) {
                const int j = byp[i0];
                const block_q8_K* a = (const block_q8_K*) (actp + (size_t) experts[j].row * a_bytes);
                float* hj = hp + (size_t) j * n_ff;
                auto act_fn = [&](float g, float u) -> float {
                    if (gelu) return 0.5f * g * (1.f + std::tanh(0.7978845608f * g * (1.f + 0.044715f * g * g))) * u;
                    g = std::min(g, limit);
                    u = std::clamp(u, -limit, limit);
                    return g / (1.f + std::exp(-g)) * u;
                };
                int r = r0;
                for (; r + 1 < r1; r += 2) {
                    const uint8_t* gate0 = e.gate + (size_t) r * gu_row;
                    const uint8_t* up0 = e.up + (size_t) r * gu_row;
                    const uint8_t* gate1 = e.gate + (size_t) (r + 1) * gu_row;
                    const uint8_t* up1 = e.up + (size_t) (r + 1) * gu_row;
                    if (r + 2 < r1) {
                        prefetch_row(gate0 + 2 * gu_row, 2 * gu_row);
                        prefetch_row(up0 + 2 * gu_row, 2 * gu_row);
                    }
                    float gu4[4] = {0.f, 0.f, 0.f, 0.f};
                    if (gu_type == GGML_TYPE_Q4_K) {
                        q4k_gate_up_2rows((const block_q4_K*) gate0, (const block_q4_K*) up0,
                                          (const block_q4_K*) gate1, (const block_q4_K*) up1, nb_gu, a,
                                          gu4[0], gu4[1], gu4[2], gu4[3]);
                    } else if (gu_type == GGML_TYPE_Q5_K) {
                        q5k_down_4rows((const block_q5_K*) gate0, (const block_q5_K*) up0,
                                       (const block_q5_K*) gate1, (const block_q5_K*) up1, nb_gu, a, gu4);
                    } else {
                        q6k_down_4rows((const block_q6_K*) gate0, (const block_q6_K*) up0,
                                       (const block_q6_K*) gate1, (const block_q6_K*) up1, nb_gu, a, gu4);
                    }
                    hj[r] = act_fn(gu4[0], gu4[1]);
                    hj[r + 1] = act_fn(gu4[2], gu4[3]);
                }
                for (; r < r1; ++r) {
                    const uint8_t* gate = e.gate + (size_t) r * gu_row;
                    const uint8_t* up = e.up + (size_t) r * gu_row;
                    float g = 0.f, u = 0.f;
                    if (gu_type == GGML_TYPE_Q4_K) {
                        q4k_gate_up_1((const block_q4_K*) gate, (const block_q4_K*) up, nb_gu, a, g, u);
                    } else {
                        tg->vec_dot(n_embd, &g, 0, gate, 0, a, 0, 1);
                        tg->vec_dot(n_embd, &u, 0, up, 0, a, 0, 1);
                    }
                    hj[r] = act_fn(g, u);
                }
                return;
            }
            for (int base = i0; base < i1; base += kMaxMultiTok) {
                const int nt = std::min(kMaxMultiTok, i1 - base);
                const uint8_t* a_ptrs[kMaxMultiTok];
                for (int k = 0; k < nt; ++k) a_ptrs[k] = actp + (size_t) experts[byp[base + k]].row * a_bytes;
                float g_arr[kMaxMultiTok], u_arr[kMaxMultiTok];
                for (int r = r0; r < r1; ++r) {
                    const uint8_t* gate = e.gate + (size_t) r * gu_row;
                    const uint8_t* up = e.up + (size_t) r * gu_row;
                    if (r + kPrefetchRows < r1) {
                        prefetch_row(gate + (size_t) kPrefetchRows * gu_row, gu_row);
                        prefetch_row(up + (size_t) kPrefetchRows * gu_row, gu_row);
                    }
                    if (gu_type == GGML_TYPE_Q4_K) {
                        q4k_dot_multi((const block_q4_K*) gate, nb_gu, a_ptrs, nt, g_arr);
                        q4k_dot_multi((const block_q4_K*) up, nb_gu, a_ptrs, nt, u_arr);
                    } else if (gu_type == GGML_TYPE_Q5_K) {
                        q5k_dot_multi((const block_q5_K*) gate, nb_gu, a_ptrs, nt, g_arr);
                        q5k_dot_multi((const block_q5_K*) up, nb_gu, a_ptrs, nt, u_arr);
                    } else {
                        q6k_dot_multi((const block_q6_K*) gate, nb_gu, a_ptrs, nt, g_arr);
                        q6k_dot_multi((const block_q6_K*) up, nb_gu, a_ptrs, nt, u_arr);
                    }
                    for (int k = 0; k < nt; ++k) {
                        const int j = byp[base + k];
                        float g = g_arr[k], u = u_arr[k];
                        if (gelu) {
                            hp[(size_t) j * n_ff + r] =
                                0.5f * g * (1.f + std::tanh(0.7978845608f * g * (1.f + 0.044715f * g * g))) * u;
                            continue;
                        }
                        g = std::min(g, limit);
                        u = std::clamp(u, -limit, limit);
                        hp[(size_t) j * n_ff + r] = g / (1.f + std::exp(-g)) * u;
                    }
                }
            }
            return;
        }
        for (int r = r0; r < r1; ++r) {
            const uint8_t* gate = e.gate + (size_t) r * gu_row;
            const uint8_t* up = e.up + (size_t) r * u_row;
            if (r + kPrefetchRows < r1) {
                prefetch_row(gate + (size_t) kPrefetchRows * gu_row, gu_row);
                prefetch_row(up + (size_t) kPrefetchRows * u_row, u_row);
            }
            for (int i = i0; i < i1; ++i) {
                const int j = byp[i];
                const uint8_t* a = actp + (size_t) experts[j].row * a_bytes;
                const uint8_t* au = actup + (size_t) experts[j].row * au_step;
                float g = 0.f, u = 0.f;
                tg->vec_dot(n_embd, &g, 0, gate, 0, a, 0, 1);
                tu->vec_dot(n_embd, &u, 0, up, 0, au, 0, 1);
                if (gelu) {
                    hp[(size_t) j * n_ff + r] =
                        0.5f * g * (1.f + std::tanh(0.7978845608f * g * (1.f + 0.044715f * g * g))) * u;
                    continue;
                }
                g = std::min(g, limit);
                u = std::clamp(u, -limit, limit);
                hp[(size_t) j * n_ff + r] = g / (1.f + std::exp(-g)) * u;
            }
        }
    });
    uint8_t* hqp = hq.data();
    const ggml_from_float_t hq_from =
        td->vec_dot_type == GGML_TYPE_Q8_K ? &q8k_quant_avx2 : traits(td->vec_dot_type)->from_float;
    workers.run(n, [&](int j) { hq_from(hp + (size_t) j * n_ff, hqp + (size_t) j * h_bytes, n_ff); });

    const int d_tasks = (n_embd + kRowsPerDownTask - 1) / kRowsPerDownTask;
    const int nb_d = n_ff / QK_K;
    workers.run(d_tasks, [&](int t) {
        static thread_local std::vector<float> one;   // an output value of every use across [r0, r1)
        const int r0 = t * kRowsPerDownTask, r1 = std::min(n_embd, r0 + kRowsPerDownTask);
        const int nr = r1 - r0;
        one.resize((size_t) nr * n);
        float* s = one.data();
        for (int u = 0; u < n_run; ++u) {
            const uint8_t* down_base = experts[byp[runp[u]]].down;
            const int i0 = runp[u], i1 = runp[u + 1];
            if (i1 - i0 == 1 && (d_type == GGML_TYPE_Q4_K || d_type == GGML_TYPE_Q5_K || d_type == GGML_TYPE_Q6_K)) {
                const int j = byp[i0];
                const block_q8_K* y = (const block_q8_K*) (hqp + (size_t) j * h_bytes);
                int r = r0;
                for (; r + 3 < r1; r += 4) {
                    const uint8_t* d0 = down_base + (size_t) (r + 0) * d_row;
                    const uint8_t* d1 = down_base + (size_t) (r + 1) * d_row;
                    const uint8_t* d2 = down_base + (size_t) (r + 2) * d_row;
                    const uint8_t* d3 = down_base + (size_t) (r + 3) * d_row;
                    if (r + 4 < r1) prefetch_row(d0 + 4 * d_row, 4 * d_row);
                    float res4[4];
                    if (d_type == GGML_TYPE_Q4_K) {
                        q4k_down_4rows((const block_q4_K*) d0, (const block_q4_K*) d1,
                                       (const block_q4_K*) d2, (const block_q4_K*) d3, nb_d, y, res4);
                    } else if (d_type == GGML_TYPE_Q5_K) {
                        q5k_down_4rows((const block_q5_K*) d0, (const block_q5_K*) d1,
                                       (const block_q5_K*) d2, (const block_q5_K*) d3, nb_d, y, res4);
                    } else {
                        q6k_down_4rows((const block_q6_K*) d0, (const block_q6_K*) d1,
                                       (const block_q6_K*) d2, (const block_q6_K*) d3, nb_d, y, res4);
                    }
                    s[(size_t) (r - r0 + 0) * n + j] = res4[0];
                    s[(size_t) (r - r0 + 1) * n + j] = res4[1];
                    s[(size_t) (r - r0 + 2) * n + j] = res4[2];
                    s[(size_t) (r - r0 + 3) * n + j] = res4[3];
                }
                for (; r < r1; ++r) {
                    td->vec_dot(n_ff, s + (size_t) (r - r0) * n + j, 0, down_base + (size_t) r * d_row, 0, y, 0, 1);
                }
                continue;
            }
            if (i1 - i0 > 1 && (d_type == GGML_TYPE_Q4_K || d_type == GGML_TYPE_Q5_K || d_type == GGML_TYPE_Q6_K)) {
                for (int base = i0; base < i1; base += kMaxMultiTok) {
                    const int nt = std::min(kMaxMultiTok, i1 - base);
                    const uint8_t* h_ptrs[kMaxMultiTok];
                    for (int k = 0; k < nt; ++k) h_ptrs[k] = hqp + (size_t) byp[base + k] * h_bytes;
                    float d_arr[kMaxMultiTok];
                    for (int r = r0; r < r1; ++r) {
                        const uint8_t* down = down_base + (size_t) r * d_row;
                        if (r + kPrefetchRows < r1) prefetch_row(down + (size_t) kPrefetchRows * d_row, d_row);
                        if (d_type == GGML_TYPE_Q4_K)
                            q4k_dot_multi((const block_q4_K*) down, nb_d, h_ptrs, nt, d_arr);
                        else if (d_type == GGML_TYPE_Q5_K)
                            q5k_dot_multi((const block_q5_K*) down, nb_d, h_ptrs, nt, d_arr);
                        else
                            q6k_dot_multi((const block_q6_K*) down, nb_d, h_ptrs, nt, d_arr);
                        float* sr = s + (size_t) (r - r0) * n;
                        for (int k = 0; k < nt; ++k) sr[byp[base + k]] = d_arr[k];
                    }
                }
                continue;
            }
            for (int r = r0; r < r1; ++r) {
                const uint8_t* down = down_base + (size_t) r * d_row;
                if (r + kPrefetchRows < r1) prefetch_row(down + (size_t) kPrefetchRows * d_row, d_row);
                float* sr = s + (size_t) (r - r0) * n;
                for (int i = i0; i < i1; ++i)
                    td->vec_dot(n_ff, sr + byp[i], 0, down, 0, hqp + (size_t) byp[i] * h_bytes, 0, 1);
            }
        }
        for (int r = r0; r < r1; ++r) {
            const float* sr = s + (size_t) (r - r0) * n;
            // a token's sum in the order its experts were given: the order it has in a call of its own
            int j = 0;
            for (int row = 0; row < rows; ++row) {
                float acc = 0.f;
                for (; j < n && experts[j].row == row; ++j) acc += experts[j].weight * sr[j];
                out[(size_t) row * n_embd + r] = acc;
            }
        }
    });
}

}  // namespace strata::poly
