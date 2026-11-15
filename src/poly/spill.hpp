// src/poly/spill.hpp - memory for what the conversation cache keeps, which leaves the process between requests.
//
// A saved state or a parked conversation's rows are written to heap memory, as before.  At the first release(), after
// the request, a thread writes them to a temporary file and frees that memory; the system writes the file's pages to
// the disk and counts them as available, and a read maps the file and finds them in the file cache, or on the disk if
// another program needed the memory.  Later releases take the pages that read brought back out of the process.
// Written to a file's mapping directly, every request read its prompt 14 ms slower on the laptop (Qwen3.6's 63 MB
// state); private memory released in place stayed in use (Windows compresses it), and memory offered with
// OfferVirtualMemory may be thrown away.  Without a temporary file (a full disk, a temporary directory in RAM) it
// stays in heap memory, as before.  Spill writes wait until the engine has been idle for idle_seconds (default 0,
// meaning no wait), so the engine's decode is not starved for memory bandwidth during a conversation switch.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>

namespace strata::poly {

/// Process-wide gate: the engine calls busy() at the start of a request and idle() when it returns.
/// Spill writes call wait_for_idle() before touching the file: if idle_seconds is 0 (default) they proceed
/// immediately; otherwise they wait until the engine has been idle for that many seconds.
class SpillGate {
public:
    /// Set the idle threshold in seconds. s >= 0; 0 means no wait. Later calls apply to writes not yet running.
    void idle_seconds(int s);
    /// Mark the engine as busy: spill writes will wait until idle() is called and the idle window passes.
    void busy();
    /// Mark the engine as idle: the idle window starts now.
    void idle();
    /// Wait until the engine has been idle for idle_seconds.  With idle_seconds==0 (the default) this returns
    /// immediately regardless of busy state.  Returns true when the wait is over.  Sleeps in ~250 ms slices so a
    /// process teardown never blocks on it.  Two test seams: probe replaces the entire busy/idle logic (returns
    /// when probe() is true); condition replaces only the busy/elapsed check (with the idle_seconds==0 fast-path
    /// still applied first).  Both default to nullptr so production code is unchanged.
    bool wait_for_idle(const std::function<bool()>& probe = nullptr,
                       const std::function<bool()>& condition = nullptr) const;
    /// Process-wide singleton: the writer is a singleton, so the gate is too.
    static SpillGate& instance();

private:
    std::atomic<int> idle_seconds_{0};
    std::atomic<bool> busy_{false};
    std::atomic<long long> idle_since_{0};  // steady-clock milliseconds since epoch, updated by idle()
};

class Spill : public std::enable_shared_from_this<Spill> {
public:
    explicit Spill(size_t bytes);
    ~Spill();
    Spill(const Spill&) = delete;
    Spill& operator=(const Spill&) = delete;

    /// Null when there was no memory for it.  The bytes are written before the first release() and only read after
    /// it; a pointer is good until the next release().
    uint8_t* data() const;
    size_t bytes() const { return n_; }

    /// Between requests, on the thread that uses data().
    void release() const;

private:
    void write() const;   // on the writer's thread
    void map() const;
    void drop() const;

    const size_t n_;
    mutable std::mutex m_;
    mutable uint8_t* heap_ = nullptr;
    mutable uint8_t* view_ = nullptr;   // the file's mapping, once the heap is freed and data() asked for it
    mutable bool queued_ = false;       // handed to the writer: no other write follows, nor a second hand-over
    mutable bool filed_ = false;        // the file holds it
    mutable bool used_ = false;         // data() was asked since the last release()
    mutable bool cold_ = false;         // the mapping's pages are out of the process
    mutable intptr_t file_ = -1;        // the file: a handle on Windows, a descriptor elsewhere
    mutable void* mapping_ = nullptr;   // Windows: the file mapping's handle
};

}  // namespace strata::poly
