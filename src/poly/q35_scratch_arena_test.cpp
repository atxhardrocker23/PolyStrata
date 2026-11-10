// Host-only scratch layout checks; no GPU context or model is needed.
#include "q35_scratch_arena.hpp"

#include <array>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>

using poly_scratch::plan;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <size_t N>
size_t check_family(const std::array<size_t, N>& sizes, const std::array<size_t, N>& offsets, size_t limit) {
    size_t expected_end = 0;
    for (size_t i = 0; i < N; ++i) {
        require(offsets[i] % 64 == 0, "buffer offset is not aligned");
        require(offsets[i] == expected_end, "unexpected buffer offset");
        require(offsets[i] <= limit && sizes[i] <= limit - offsets[i], "buffer exceeds arena");
        for (size_t j = 0; j < i; ++j) {
            if (sizes[i] != 0 && sizes[j] != 0)
                require(offsets[i] >= offsets[j] + sizes[j] || offsets[j] >= offsets[i] + sizes[i],
                        "buffers within a family overlap");
        }
        expected_end += ((sizes[i] + 63) / 64) * 64;
    }
    return expected_end;
}

template <typename Function>
void expect_overflow(Function operation) {
    bool threw = false;
    try {
        operation();
    } catch (const std::overflow_error&) {
        threw = true;
    }
    require(threw, "expected overflow rejection");
}

void check_tiel_and_random_layouts() {
    constexpr size_t batch = 2048;
    constexpr size_t heads = batch * 16 * 256, kv = batch * 2 * 256;
    std::array<size_t, 6> recurrent = {
        batch * 8192, batch * 8192, batch * 4096, batch * 32, batch * 32, batch * 4096};
    std::array<size_t, 7> attention = {
        2 * heads, heads, kv, kv, kv, heads, heads + heads / 2 + 1 + 3 * batch * 16};

    auto layout = plan(recurrent, attention);
    const size_t recurrent_size = check_family(recurrent, layout.a, layout.total);
    const size_t attention_size = check_family(attention, layout.b, layout.total);
    require(recurrent_size == 50462720 && attention_size == 49381440, "unexpected Tiel family sizes");
    require(layout.total == 50462720, "unexpected Tiel arena size");
    require((recurrent_size + attention_size - layout.total) * sizeof(float) == 197525760,
            "unexpected Tiel memory saving");

    std::mt19937_64 random(42);
    for (int trial = 0; trial < 10000; ++trial) {
        for (auto& size : recurrent) size = random() % 100000;
        for (auto& size : attention) size = random() % 100000;
        layout = plan(recurrent, attention);
        const size_t a_size = check_family(recurrent, layout.a, layout.total);
        const size_t b_size = check_family(attention, layout.b, layout.total);
        require(layout.total == (a_size > b_size ? a_size : b_size), "arena must fit the larger family");
    }
}

void check_empty_and_overflow() {
    std::array<size_t, 6> recurrent{};
    const std::array<size_t, 7> attention{};
    const auto empty = plan(recurrent, attention);
    require(empty.total == 0, "empty families need no arena");
    check_family(recurrent, empty.a, empty.total);
    check_family(attention, empty.b, empty.total);

    recurrent[0] = std::numeric_limits<size_t>::max();
    expect_overflow([&] { (void) plan(recurrent, attention); });

    recurrent.fill(std::numeric_limits<size_t>::max() / 2 - 127);
    expect_overflow([&] { (void) plan(recurrent, attention); });
}

void check_unused_combined_size() {
    // Neither family overflows; their sum does. Only their maximum is allocated.
    const size_t large = std::numeric_limits<size_t>::max() / 2 + 1;
    const std::array<size_t, 1> family = {large};
    try {
        const auto layout = plan(family, family);
        require(layout.total == large, "unexpected shared arena size");
        require(layout.a[0] == 0 && layout.b[0] == 0, "families must share the arena origin");
    } catch (const std::overflow_error&) {
        throw std::runtime_error("planner checked the unused combined family size");
    }
}
} // namespace

int main() {
    try {
        check_tiel_and_random_layouts();
        check_empty_and_overflow();
        check_unused_combined_size();
        std::puts("PASS 10005 arena fixtures: geometry, alignment, bounds, overlap, empty and overflow");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
