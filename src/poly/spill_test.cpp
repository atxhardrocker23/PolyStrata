// src/poly/spill_test.cpp - Verify the idle gate that delays spill writes until the engine is idle.
//
//   g++ -std=c++20 src/poly/spill_test.cpp src/poly/spill.cpp -o spill_test && ./spill_test
#include "spill.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

using strata::poly::SpillGate;

namespace {

int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

// idle_seconds == 0: wait_for_idle returns immediately even when busy() is set.
void idle_s_zero() {
    SpillGate gate;
    gate.busy();
    auto start = std::chrono::steady_clock::now();
    check(gate.wait_for_idle(), "idle_s==0 returns immediately even when busy");
    auto elapsed = std::chrono::steady_clock::now() - start;
    check(elapsed < std::chrono::milliseconds(50), "idle_s==0 returns at once even when busy");
}

// busy() -> wait_for_idle blocks until idle() is called, then returns.
// Uses the condition seam (returns false initially, true after idle()) to keep it fast.
void busy_then_idle() {
    SpillGate gate;
    gate.idle_seconds(5);
    gate.busy();

    bool ready = false;
    bool done = false;
    std::thread t([&] {
        check(gate.wait_for_idle(nullptr, [&] { return ready; }), "wait_for_idle returns when the condition holds");
        done = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    check(!done, "wait_for_idle blocks while busy");

    gate.idle();
    ready = true;
    t.join();
    check(done, "wait_for_idle returns after idle()");
}

// idle_seconds > 0 with a probe: wait stays false until the probe's condition holds.
void probe_with_idle_s_positive() {
    SpillGate gate;
    gate.idle_seconds(3);
    gate.busy();

    bool probe_ready = false;
    bool done = false;
    std::thread t([&] {
        check(gate.wait_for_idle([&] { return probe_ready; }), "wait_for_idle with probe returns when probe is true");
        done = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    check(!done, "wait_for_idle with probe stays false");

    probe_ready = true;
    gate.idle();
    t.join();
    check(done, "wait_for_idle with probe returns when probe is true");
}

// The bug case: busy() after idle() must block even though idle_since_ is old.
// OR semantics would return immediately; AND semantics block until idle() clears busy_.
void busy_blocks_even_after_long_idle() {
    SpillGate gate;
    gate.idle_seconds(5);
    gate.idle();   // idle_since_ is set
    gate.busy();   // busy_ = true, idle_since_ is old

    bool ready = false;
    bool done = false;
    std::thread t([&] {
        check(gate.wait_for_idle(nullptr, [&] { return ready; }), "wait_for_idle returns when the condition holds");
        done = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    check(!done, "wait_for_idle blocks even when idle_since_ is old");

    ready = true;
    gate.idle();
    t.join();
    check(done, "wait_for_idle returns when ready is set");
}

}  // namespace

int main() {
    idle_s_zero();
    busy_then_idle();
    probe_with_idle_s_positive();
    busy_blocks_even_after_long_idle();
    std::printf("spill_test: %d checks passed\n", checks);
    return 0;
}
