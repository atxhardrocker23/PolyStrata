#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace poly_scratch {

// Sizes and offsets are float counts. Sixty-four floats give 256-byte alignment.
template <size_t A, size_t B>
struct Plan {
    std::array<size_t, A> a{};
    std::array<size_t, B> b{};
    size_t total = 0;
};

inline size_t plus(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::overflow_error("scratch arena overflow");
    return a + b;
}

inline size_t aligned(size_t value) {
    return plus(value, 63) & ~size_t(63);
}

template <size_t N>
size_t offsets(const std::array<size_t, N>& sizes, std::array<size_t, N>& at) {
    size_t size = 0;
    for (size_t i = 0; i < N; ++i) {
        at[i] = size;
        size = plus(size, aligned(sizes[i]));
    }
    return size;
}

template <size_t A, size_t B>
Plan<A, B> plan(const std::array<size_t, A>& a, const std::array<size_t, B>& b) {
    Plan<A, B> result;
    const size_t a_size = offsets(a, result.a);
    const size_t b_size = offsets(b, result.b);
    result.total = std::max(a_size, b_size);
    return result;
}

} // namespace poly_scratch
