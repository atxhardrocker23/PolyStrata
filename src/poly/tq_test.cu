// src/poly/tq_test.cu - the TurboQuant cache (--kv turbo, fattn.hpp's Kvtq) on Qwen3.6's shape (heads of 256 values,
// 16 heads, 2 KV heads):
//   1. turbo_rotate() and its inverse give the vectors back;
//   2. fdec_tq() (a few tokens) and fattn_tq() (a prompt batch) against the arithmetic in doubles on the cache the
//      store wrote, dequantized here (the rotated domain both read);
//   3. the whole path (store, rotated queries, attention, output turned back) against the exact attention over the
//      unquantized keys and values: the quantization's own error, printed, and bounded loosely.
//
// usage: poly_tq_test   (exit 0: every check within its bound)
#include "fattn.hpp"
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

constexpr int kDim = 256, kKv = 2, kHeads = 16, kWidth = kKv * kDim;
constexpr size_t kScore = size_t(16) << 20;
const int8_t kI4[16] = {-127, -96, -75, -58, -44, -31, -18, -6, 6, 18, 31, 44, 58, 75, 96, 127};
const int8_t kI3[8] = {-127, -79, -45, -14, 14, 45, 79, 127};
const float kK4 = 0.241529f / 127.f, kK3 = 0.190207f / 127.f;

double uniform(std::mt19937& rng) { return (rng() + 0.5) / 4294967296.0; }
float draw(std::mt19937& rng) {
    const double u = uniform(rng), v = uniform(rng);
    return (float) (std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v));
}

template <typename T> T* dev(size_t n) {
    T* d = nullptr;
    if (cudaMalloc(&d, n * sizeof(T)) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        std::exit(2);
    }
    return d;
}
template <typename T> T* upload(const std::vector<T>& v) {
    T* d = dev<T>(v.size());
    cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> v(n);
    cudaDeviceSynchronize();
    cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return v;
}

// causal attention in doubles of tokens [pos, pos + n): out[(t * kHeads + h) * kDim + d]
std::vector<double> attend_ref(const std::vector<float>& q, const std::vector<float>& K, const std::vector<float>& V,
                               long long pos, int n) {
    std::vector<double> out((size_t) n * kHeads * kDim, 0.0);
    const double scale = 1.0 / std::sqrt((double) kDim);
    std::vector<double> s;
    for (int t = 0; t < n; ++t)
        for (int h = 0; h < kHeads; ++h) {
            const int g = h / (kHeads / kKv);
            const long long rows = pos + t + 1;
            s.assign((size_t) rows, 0.0);
            double top = -1e300;
            const float* qr = &q[((size_t) t * kHeads + h) * kDim];
            for (long long r = 0; r < rows; ++r) {
                const float* kr = &K[((size_t) r * kKv + g) * kDim];
                double x = 0;
                for (int d = 0; d < kDim; ++d) x += (double) qr[d] * kr[d];
                s[(size_t) r] = x * scale;
                top = std::max(top, s[(size_t) r]);
            }
            double sum = 0;
            for (double& x : s) sum += x = std::exp(x - top);
            double* o = &out[((size_t) t * kHeads + h) * kDim];
            for (long long r = 0; r < rows; ++r) {
                const float* vr = &V[((size_t) r * kKv + g) * kDim];
                const double p = s[(size_t) r] / sum;
                for (int d = 0; d < kDim; ++d) o[d] += p * vr[d];
            }
        }
    return out;
}

// the largest difference against the doubles, as a share of the largest reference value (at least 1)
double worst(const std::vector<float>& got, const std::vector<double>& want) {
    double big = 1, bad = 0;
    for (double x : want) big = std::max(big, std::fabs(x));
    for (size_t i = 0; i < want.size(); ++i) bad = std::max(bad, std::fabs(got[i] - want[i]));
    return bad / big;
}

// relative L2 error per output row, the mean over the rows
double rel_l2(const std::vector<float>& got, const std::vector<double>& want) {
    double acc = 0;
    const size_t rows = want.size() / kDim;
    for (size_t r = 0; r < rows; ++r) {
        double e = 0, w = 0;
        for (int d = 0; d < kDim; ++d) {
            const double x = got[r * kDim + d] - want[r * kDim + d];
            e += x * x;
            w += want[r * kDim + d] * want[r * kDim + d];
        }
        acc += std::sqrt(e / std::max(w, 1e-30));
    }
    return acc / (double) rows;
}

}  // namespace

int main() {
    std::mt19937 rng(7);
    const int rows = 6000;   // cache rows; three splits of the decode kernel
    int bad = 0;

    // keys with a few large channels, as real ones have; values plain
    std::vector<float> K((size_t) rows * kWidth), V((size_t) rows * kWidth);
    for (size_t i = 0; i < K.size(); ++i) K[i] = draw(rng) * ((i % kDim) % 37 == 0 ? 6.f : 1.f);
    for (float& x : V) x = draw(rng);

    float* dK = upload(K);
    float* dV = upload(V);
    uint8_t* kq = dev<uint8_t>((size_t) rows * kKv * kTurboKeyBytes);
    uint8_t* vq = dev<uint8_t>((size_t) rows * kKv * kTurboValBytes);
    float* kn = dev<float>((size_t) rows * kKv * 2);
    float* vn = dev<float>((size_t) rows * kKv * 2);
    // in pieces, as the engine stores a prompt batch at a time, and the last rows from a position on the card
    for (int at = 0; at < rows - 8; at += 1024) {
        const int n = std::min(1024, rows - 8 - at);
        turbo_store(dK + (size_t) at * kWidth, dV + (size_t) at * kWidth, kq, kn, vq, vn, kKv, at, n, nullptr, 0);
    }
    long long* dpos = upload(std::vector<long long>{rows - 8});
    turbo_store(dK + (size_t) (rows - 8) * kWidth, dV + (size_t) (rows - 8) * kWidth, kq, kn, vq, vn, kKv, 0, 8, dpos,
                0);

    // the cache dequantized here: what the readers must see
    const auto hkq = down(kq, (size_t) rows * kKv * kTurboKeyBytes), hvq = down(vq, (size_t) rows * kKv * kTurboValBytes);
    const auto hkn = down(kn, (size_t) rows * kKv * 2), hvn = down(vn, (size_t) rows * kKv * 2);
    std::vector<float> Kd((size_t) rows * kWidth), Vd((size_t) rows * kWidth);
    for (int r = 0; r < rows; ++r)
        for (int g = 0; g < kKv; ++g) {
            const size_t row = (size_t) r * kKv + g;
            for (int d = 0; d < kDim; ++d) {
                const uint8_t b = hkq[row * kTurboKeyBytes + d / 2];
                Kd[row * kDim + d] = kI4[(d & 1) ? b >> 4 : b & 15] * kK4 * hkn[row * 2 + d / 128];
                const uint8_t* vr = &hvq[row * kTurboValBytes];
                const int idx = ((vr[d / 4] >> (2 * (d % 4))) & 3) | (((vr[64 + d / 8] >> (d % 8)) & 1) << 2);
                Vd[row * kDim + d] = kI3[idx] * kK3 * hvn[row * 2 + d / 128];
            }
        }
    // the dequantized halves keep their length (the corrected norm)
    {
        double worst_len = 0;
        std::vector<float> Kr((size_t) kDim);
        float* dx = dev<float>(kWidth);
        for (int r = 0; r < 64; ++r) {
            cudaMemcpy(dx, &K[(size_t) r * kWidth], kWidth * sizeof(float), cudaMemcpyHostToDevice);
            for (int g = 0; g < kKv; ++g)
                for (int b = 0; b < 2; ++b) {
                    double a = 0, z = 0;
                    for (int d = 128 * b; d < 128 * b + 128; ++d) {
                        a += (double) K[((size_t) r * kKv + g) * kDim + d] * K[((size_t) r * kKv + g) * kDim + d];
                        z += (double) Kd[((size_t) r * kKv + g) * kDim + d] * Kd[((size_t) r * kKv + g) * kDim + d];
                    }
                    worst_len = std::max(worst_len, std::fabs(std::sqrt(z) / std::sqrt(a) - 1));
                }
        }
        cudaFree(dx);
        std::printf("dequantized halves' length vs the original: worst %.2e\n", worst_len);
        if (worst_len > 1e-3) ++bad;
    }

    // 1. the rotation and its inverse
    {
        std::vector<float> x((size_t) 64 * kDim);
        for (float& v : x) v = draw(rng);
        float* dx = upload(x);
        turbo_rotate(dx, 64, false, 0);
        const auto y = down(dx, x.size());
        double ny = 0, nx = 0;
        for (size_t i = 0; i < x.size(); ++i) nx += (double) x[i] * x[i], ny += (double) y[i] * y[i];
        turbo_rotate(dx, 64, true, 0);
        const auto z = down(dx, x.size());
        double e = 0;
        for (size_t i = 0; i < x.size(); ++i) e = std::max(e, (double) std::fabs(z[i] - x[i]));
        std::printf("rotation: length kept to %.2e, inverse back to %.2e\n", std::fabs(ny / nx - 1), e);
        if (e > 1e-4 || std::fabs(ny / nx - 1) > 1e-5) ++bad;
        cudaFree(dx);
    }

    float* score = dev<float>(kScore);
    const float scale = 1.f / std::sqrt((float) kDim);
    struct Case {
        long long pos;
        int n;
        bool batch;
    };
    // decode: one token, four, at the start of a split and across; batches: from 0 and from the middle
    const Case cases[] = {{rows - 1, 1, false}, {rows - 4, 4, false}, {1023, 1, false}, {3000, 3, false},
                          {0, 1024, true}, {rows - 1024, 1024, true}, {2500, 700, true}};
    for (const Case& c : cases) {
        if (c.batch && !fattn_fits(kDim, kHeads, kKv, c.n, kScore)) {
            std::printf("fattn_tq: no kernel on this card, skipped\n");
            continue;
        }
        if (!c.batch && !fdec_fits(kDim, kHeads, kKv, rows, kScore)) {
            std::printf("fdec_tq: no kernel on this card, skipped\n");
            continue;
        }
        std::vector<float> q((size_t) c.n * kHeads * kDim);
        for (float& x : q) x = draw(rng);
        float* dq = upload(q);
        float* out = dev<float>(q.size());
        turbo_rotate(dq, c.n * kHeads, false, 0);
        const auto qr = down(dq, q.size());
        const Kvtq kt{kq, kn}, vt{vq, vn};
        const bool ok = c.batch ? fattn_tq(dq, kt, vt, c.pos, c.n, out, kDim, kHeads, kKv, scale, score, kScore, 0)
                                : fdec_tq(dq, kt, vt, c.pos, c.n, nullptr, 0, out, kDim, kHeads, kKv, scale, score,
                                          kScore, 0);
        if (!ok) {
            std::printf("%s pos %lld n %d: the kernel did not take it\n", c.batch ? "fattn_tq" : "fdec_tq", c.pos, c.n);
            ++bad;
            continue;
        }
        const auto rot = down(out, q.size());
        // 2. the readers: against the doubles on the dequantized cache, in the rotated domain
        const double e_read = worst(rot, attend_ref(qr, Kd, Vd, c.pos, c.n));
        // 3. the whole path, back from the rotation, against the exact attention
        turbo_rotate(out, c.n * kHeads, true, 0);
        const auto got = down(out, q.size());
        const auto exact = attend_ref(q, K, V, c.pos, c.n);
        const double e_all = rel_l2(got, exact);
        const double room = c.batch ? 2e-2 : 7.5e-3;   // fattn_test.cu's bounds (halves in the kernel's sums)
        std::printf("%-8s pos %5lld n %4d: readers vs doubles %.2e (bound %.1e) | turbo vs exact attention, mean "
                    "relative L2 %.3f\n",
                    c.batch ? "fattn_tq" : "fdec_tq", c.pos, c.n, e_read, room, e_all);
        if (e_read > room || e_all > 0.25) ++bad;
        // a decode token from the position on the card comes out the same to the bit
        if (!c.batch) {
            long long* at = upload(std::vector<long long>{c.pos});
            float* out2 = dev<float>(q.size());
            fdec_tq(dq, kt, vt, 0, c.n, at, rows, out2, kDim, kHeads, kKv, scale, score, kScore, 0);
            const auto rot2 = down(out2, q.size());
            const bool same = std::equal(rot.begin(), rot.end(), rot2.begin());
            if (!same) {
                std::printf("         from the position on the card: not the same bits\n");
                ++bad;
            }
            cudaFree(at);
            cudaFree(out2);
        }
        cudaFree(dq);
        cudaFree(out);
    }
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::printf("CUDA error: %s\n", cudaGetErrorString(e));
        ++bad;
    }
    std::printf(bad ? "TQ TEST FAILED (%d)\n" : "TQ TEST OK\n", bad);
    return bad ? 1 : 0;
}
