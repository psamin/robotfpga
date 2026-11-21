// Host-only stand-in for Vitis HLS <hls_stream.h>: hls::stream<T, DEPTH> as a bounded, blocking, thread-safe FIFO
// of exactly DEPTH entries (the HLS default 2 when DEPTH is 0). With -DSMX_DFSIM and this directory first on the
// include path, softmax_core.h runs its DATAFLOW processes as threads on these FIFOs: a deadlock of the hardware
// FIFO network (a full or empty FIFO that never moves) shows as a blocked read or write. After SMX_DFSIM_TIMEOUT
// seconds (default 30) of no progress on one FIFO the run prints the FIFO and exits with code 7.
#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>

#ifndef SMX_DFSIM_TIMEOUT
#define SMX_DFSIM_TIMEOUT 30
#endif

namespace hls {

template <typename T, int DEPTH = 0>
class stream {
  std::deque<T> q_;
  std::mutex m_;
  std::condition_variable cv_;
  std::string name_;
  size_t cap_;

  [[noreturn]] void dead(const char* what) {
    printf("DEADLOCK: %s blocked on %s (depth %zu, %zu entries)\n", what, name_.c_str(), cap_, q_.size());
    fflush(stdout);
    _Exit(7);
  }

 public:
  size_t max_occupancy = 0;
  explicit stream(const char* name = "anon") : name_(name), cap_(DEPTH > 0 ? DEPTH : 2) {}
  stream(const stream&) = delete;
  stream& operator=(const stream&) = delete;

  void write(const T& v) {
    std::unique_lock<std::mutex> l(m_);
    if (!cv_.wait_for(l, std::chrono::seconds(SMX_DFSIM_TIMEOUT), [&] { return q_.size() < cap_; }))
      dead("write");
    q_.push_back(v);
    if (q_.size() > max_occupancy) max_occupancy = q_.size();
    cv_.notify_all();
  }
  T read() {
    std::unique_lock<std::mutex> l(m_);
    if (!cv_.wait_for(l, std::chrono::seconds(SMX_DFSIM_TIMEOUT), [&] { return !q_.empty(); })) dead("read");
    T v = q_.front();
    q_.pop_front();
    cv_.notify_all();
    return v;
  }
  void read(T& v) { v = read(); }
  bool read_nb(T& v) {
    std::lock_guard<std::mutex> l(m_);
    if (q_.empty()) return false;
    v = q_.front();
    q_.pop_front();
    cv_.notify_all();
    return true;
  }
  bool write_nb(const T& v) {
    std::lock_guard<std::mutex> l(m_);
    if (q_.size() >= cap_) return false;
    q_.push_back(v);
    cv_.notify_all();
    return true;
  }
  bool empty() {
    std::lock_guard<std::mutex> l(m_);
    return q_.empty();
  }
  bool full() {
    std::lock_guard<std::mutex> l(m_);
    return q_.size() >= cap_;
  }
};

}  // namespace hls
