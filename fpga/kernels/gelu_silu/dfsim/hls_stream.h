// Bounded, blocking hls::stream for the threaded DATAFLOW deadlock model of gelu_silu.h (host only).
// Build tb_gelu_silu.cpp with -DGS_DFSIM -I<this dir> ahead of the Vitis include directory: gs_df then runs every
// process in its own std::thread, and every stream holds at most the depth its STREAM pragma gives (the testbench
// maps the stream's name to the same GS_D_* macro through gs_dfsim_depth). A read on an empty or a write on a full
// stream that waits longer than GS_DFSIM_TIMEOUT seconds prints DEADLOCK with the stream's name and exits with 7.
#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>

#ifndef GS_DFSIM_TIMEOUT
#define GS_DFSIM_TIMEOUT 20
#endif

int gs_dfsim_depth(const std::string& name);  // defined by the testbench

namespace hls {

template <typename T>
class stream {
  std::deque<T> q_;
  mutable std::mutex m_;
  std::condition_variable cv_;
  std::string name_;
  size_t cap_;
  size_t peak_ = 0;

  [[noreturn]] void deadlock(const char* what) {
    fprintf(stdout, "DEADLOCK: %s blocked on %s (depth %zu, holds %zu)\n", what, name_.c_str(), cap_, q_.size());
    fflush(stdout);
    _Exit(7);
  }

 public:
  explicit stream(const char* name = "anon") : name_(name), cap_((size_t)gs_dfsim_depth(name)) {}
  stream(const stream&) = delete;
  stream& operator=(const stream&) = delete;

  void write(const T& v) {
    std::unique_lock<std::mutex> l(m_);
    if (!cv_.wait_for(l, std::chrono::seconds(GS_DFSIM_TIMEOUT), [&] { return q_.size() < cap_; })) deadlock("write");
    q_.push_back(v);
    if (q_.size() > peak_) peak_ = q_.size();
    cv_.notify_all();
  }
  T read() {
    std::unique_lock<std::mutex> l(m_);
    if (!cv_.wait_for(l, std::chrono::seconds(GS_DFSIM_TIMEOUT), [&] { return !q_.empty(); })) deadlock("read");
    T v = q_.front();
    q_.pop_front();
    cv_.notify_all();
    return v;
  }
  void read(T& v) { v = read(); }
  bool empty() const {
    std::lock_guard<std::mutex> l(m_);
    return q_.empty();
  }
  size_t size() const {
    std::lock_guard<std::mutex> l(m_);
    return q_.size();
  }
  size_t peak() const { return peak_; }
  const std::string& name() const { return name_; }
};

}  // namespace hls
