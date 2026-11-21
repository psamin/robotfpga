// Bounded, blocking hls::stream for the ROPE dataflow deadlock model (CONTRACT.md 7: "threads + FIFOs bounded at
// the pragma depths"). Built only into tb_rope_dfsim: -Idfsim comes before the Vitis include directory, so this
// header replaces <hls_stream.h>, and rope_mode.h (ROPE_DFSIM) runs every DATAFLOW process as a thread.
// The depth of a stream comes from its name (rope_dfsim_depth, defined by the testbench from the ROPE_D_* macros
// that the STREAM pragmas use). A blocked read or write that makes no progress for 20 s is reported as a deadlock
// and ends the process with exit code 7.
#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>

int rope_dfsim_depth(const std::string& name);

namespace hls {
template <typename T>
class stream {
  std::deque<T> q_;
  std::mutex m_;
  std::condition_variable cv_;
  std::string name_;
  size_t cap_;

 public:
  size_t max_occupancy = 0;
  explicit stream(const char* n = "anon") : name_(n), cap_((size_t)rope_dfsim_depth(n)) {}
  void write(const T& v) {
    std::unique_lock<std::mutex> l(m_);
    if (!cv_.wait_for(l, std::chrono::seconds(20), [&] { return q_.size() < cap_; })) {
      printf("DEADLOCK: write blocked on full %s (depth %zu)\n", name_.c_str(), cap_);
      fflush(stdout);
      _Exit(7);
    }
    q_.push_back(v);
    if (q_.size() > max_occupancy) max_occupancy = q_.size();
    cv_.notify_all();
  }
  T read() {
    std::unique_lock<std::mutex> l(m_);
    if (!cv_.wait_for(l, std::chrono::seconds(20), [&] { return !q_.empty(); })) {
      printf("DEADLOCK: read blocked on empty %s (depth %zu)\n", name_.c_str(), cap_);
      fflush(stdout);
      _Exit(7);
    }
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
  bool empty() {
    std::lock_guard<std::mutex> l(m_);
    return q_.empty();
  }
  bool full() {
    std::lock_guard<std::mutex> l(m_);
    return q_.size() >= cap_;
  }
  size_t size() {
    std::lock_guard<std::mutex> l(m_);
    return q_.size();
  }
};
}  // namespace hls
