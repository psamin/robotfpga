// Host-only bounded hls::stream for tb_attn's threaded DATAFLOW deadlock model (CONTRACT.md 7) of the units the
// attention path runs on: the row unit (quantize/quant_mode.h, -DQZ_DFSIM: streams named "qz_*", depth from the
// QZ_D_* macros through qz_dfsim_depth()) and the softmax unit (softmax/softmax_core.h, -DSMX_DFSIM: depth as the
// stream's template argument). It merges the two units' own dfsim headers (quantize/dfsim, softmax/tb_dfsim), whose
// interfaces differ, so one tb_attn build can thread both units. A read of an empty or a write to a full stream that
// waits longer than ATTN_DFSIM_TIMEOUT seconds prints DEADLOCK with the stream's name and exits with code 7.
//   g++ ... -DQZ_DFSIM -DSMX_DFSIM -Idfsim -I<Vitis_HLS>/include tb_attn.cpp      (this directory first)
#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <string>

#ifndef ATTN_DFSIM_TIMEOUT
#define ATTN_DFSIM_TIMEOUT 30
#endif

int qz_dfsim_depth(const std::string& name);  // defined by tb_attn.cpp (0: not a row-unit stream)

namespace hls {

template <typename T, int DEPTH = 0>
class stream {
  std::deque<T> q_;
  mutable std::mutex m_;
  std::condition_variable cv_;
  std::string name_;
  size_t cap_;

  [[noreturn]] void deadlock(const char* what) {
    fprintf(stdout, "DEADLOCK: %s blocked on %s (depth %zu, holds %zu)\n", what, name_.c_str(), cap_, q_.size());
    fflush(stdout);
    _Exit(7);
  }
  static size_t depth_of(const char* name) {
    if (DEPTH > 0) return (size_t)DEPTH;
    const int d = qz_dfsim_depth(name);
    return d > 0 ? (size_t)d : 2u;  // the HLS default depth
  }

 public:
  explicit stream(const char* name = "anon") : name_(name), cap_(depth_of(name)) {}
  stream(const stream&) = delete;
  stream& operator=(const stream&) = delete;

  void write(const T& v) {
    std::unique_lock<std::mutex> l(m_);
    if (!cv_.wait_for(l, std::chrono::seconds(ATTN_DFSIM_TIMEOUT), [&] { return q_.size() < cap_; })) deadlock("write");
    q_.push_back(v);
    cv_.notify_all();
  }
  T read() {
    std::unique_lock<std::mutex> l(m_);
    if (!cv_.wait_for(l, std::chrono::seconds(ATTN_DFSIM_TIMEOUT), [&] { return !q_.empty(); })) deadlock("read");
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
  bool empty() const {
    std::lock_guard<std::mutex> l(m_);
    return q_.empty();
  }
  bool full() const {
    std::lock_guard<std::mutex> l(m_);
    return q_.size() >= cap_;
  }
  size_t size() const {
    std::lock_guard<std::mutex> l(m_);
    return q_.size();
  }
};

}  // namespace hls
