// Host-only (g++) DATAFLOW model with bounded FIFOs, for tb_nr.cpp's threaded build (-DNR_DFSIM -Idfsim ahead of
// the Vitis include directory). It replaces <hls_stream.h>: every hls::stream holds at most the depth its STREAM
// pragma gives (the testbench maps each stream's name to the same macro, nr_dfsim_depth), and the region runs every
// process in its own thread (NR_RUN / NR_JOIN in norm_residual.h). The NORM front end and the row unit's backbone
// processes (quantize/quant_mode.h) both run on it, unchanged. A region that would hang the IP hangs here too: the
// watchdog sees no FIFO traffic for a while with threads still running, prints which streams are full or empty and
// who waits, and exits with code 3. Vitis HLS never sees this file.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

int nr_dfsim_depth(const std::string& name);  // defined by the testbench: the pragma depth of each named stream

namespace dfsim {

struct base {
  std::string name;
  size_t depth, peak = 0;
  int waiting_rd = 0, waiting_wr = 0;
  virtual size_t count() = 0;
  virtual ~base() {}
};

inline std::mutex& reg_mu() {
  static std::mutex m;
  return m;
}
inline std::vector<base*>& registry() {
  static std::vector<base*> r;
  return r;
}
// highest occupancy seen per stream name over every region run so far (tb_nr prints it against the depths)
struct peak_rec {
  std::string name;
  size_t depth, peak;
};
inline std::vector<peak_rec>& peaks() {
  static std::vector<peak_rec> p;
  return p;
}
inline std::atomic<long long>& progress() {
  static std::atomic<long long> p(0);
  return p;
}

inline std::vector<std::thread>& threads() {
  static thread_local std::vector<std::thread> t;  // per region (a nested region runs in its own thread)
  return t;
}
inline std::atomic<int>& live() {
  static std::atomic<int> n(0);
  return n;
}
inline void run(std::function<void()> f) {
  live()++;
  threads().push_back(std::thread([f] {
    f();
    live()--;
  }));
}

// Join every process of the region. The watchdog (one, for the outermost region) reports a deadlock when no
// stream moved for `ms` milliseconds while processes are still running (every one blocked on a full or empty
// stream).
inline void join_all(int ms = 3000) {
  static std::atomic<int> depth(0);
  const bool outer = depth++ == 0;
  std::atomic<bool> done(false);
  std::thread watch;
  if (outer)
    watch = std::thread([&] {
      long long last = progress().load();
      auto t0 = std::chrono::steady_clock::now();
      while (!done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const long long p = progress().load();
        if (p != last) {
          last = p;
          t0 = std::chrono::steady_clock::now();
        } else if (live().load() > 0 && std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::steady_clock::now() - t0)
                                                .count() > ms) {
          std::lock_guard<std::mutex> g(reg_mu());
          printf("DEADLOCK: no FIFO progress for %d ms, %d process(es) blocked; streams:\n", ms, live().load());
          for (base* b : registry())
            printf("  %-6s depth %4zu holds %4zu (peak %4zu)  readers waiting %d  writers waiting %d\n",
                   b->name.c_str(), b->depth, b->count(), b->peak, b->waiting_rd, b->waiting_wr);
          fflush(stdout);
          std::_Exit(3);
        }
      }
    });
  for (auto& t : threads()) t.join();
  threads().clear();
  depth--;
  if (outer) {
    done = true;
    watch.join();
  }
}

}  // namespace dfsim

namespace hls {

template <typename T>
class stream : public dfsim::base {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<T> q;

 public:
  explicit stream(const char* n = "anon") {
    name = n;
    const int d = nr_dfsim_depth(name);
    depth = d < 1 ? 1 : (size_t)d;
    std::lock_guard<std::mutex> g(dfsim::reg_mu());
    dfsim::registry().push_back(this);
  }
  stream(const stream&) = delete;
  stream& operator=(const stream&) = delete;
  ~stream() {
    std::lock_guard<std::mutex> g(dfsim::reg_mu());
    bool found = false;
    for (auto& p : dfsim::peaks())
      if (p.name == name) {
        p.peak = p.peak > peak ? p.peak : peak;
        found = true;
      }
    if (!found) dfsim::peaks().push_back(dfsim::peak_rec{name, depth, peak});
    auto& r = dfsim::registry();
    for (size_t i = 0; i < r.size(); i++)
      if (r[i] == this) {
        r.erase(r.begin() + i);
        break;
      }
  }
  void write(const T& v) {
    std::unique_lock<std::mutex> l(mu);
    waiting_wr++;
    cv.wait(l, [&] { return q.size() < depth; });
    waiting_wr--;
    q.push_back(v);
    if (q.size() > peak) peak = q.size();
    dfsim::progress()++;
    cv.notify_all();
  }
  T read() {
    std::unique_lock<std::mutex> l(mu);
    waiting_rd++;
    cv.wait(l, [&] { return !q.empty(); });
    waiting_rd--;
    T v = q.front();
    q.pop_front();
    dfsim::progress()++;
    cv.notify_all();
    return v;
  }
  void read(T& v) { v = read(); }
  bool read_nb(T& v) {  // never blocks (the backbone's sr polls with it, as the hardware does)
    std::lock_guard<std::mutex> l(mu);
    if (q.empty()) return false;
    v = q.front();
    q.pop_front();
    dfsim::progress()++;
    cv.notify_all();
    return true;
  }
  bool write_nb(const T& v) {  // never blocks
    std::lock_guard<std::mutex> l(mu);
    if (q.size() >= depth) return false;
    q.push_back(v);
    if (q.size() > peak) peak = q.size();
    dfsim::progress()++;
    cv.notify_all();
    return true;
  }
  bool empty() {
    std::lock_guard<std::mutex> l(mu);
    return q.empty();
  }
  bool full() {
    std::lock_guard<std::mutex> l(mu);
    return q.size() >= depth;
  }
  size_t count() override {
    std::lock_guard<std::mutex> l(mu);
    return q.size();
  }
};

}  // namespace hls
