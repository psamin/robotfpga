// Host-only (g++) DATAFLOW model with bounded FIFOs: every process of a dataflow region runs in its own thread
// and every stream blocks at the depth its declaration gives, as in hardware. A region that would hang the IP
// hangs here too; the watchdog reports it (which streams are full / empty, who waits) and exits with code 3.
// Built only with -DNR_DFSIM (tb_nr.cpp's threaded build); Vitis HLS never sees this file.
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

template <class T>
class stream : public base {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<T> q;

 public:
  stream(size_t d, const char* n) {
    depth = d < 1 ? 1 : d;
    name = n;
    std::lock_guard<std::mutex> g(reg_mu());
    registry().push_back(this);
  }
  ~stream() {
    std::lock_guard<std::mutex> g(reg_mu());
    bool found = false;
    for (auto& p : peaks())
      if (p.name == name) {
        p.peak = p.peak > peak ? p.peak : peak;
        found = true;
      }
    if (!found) peaks().push_back(peak_rec{name, depth, peak});
    auto& r = registry();
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
    progress()++;
    cv.notify_all();
  }
  T read() {
    std::unique_lock<std::mutex> l(mu);
    waiting_rd++;
    cv.wait(l, [&] { return !q.empty(); });
    waiting_rd--;
    T v = q.front();
    q.pop_front();
    progress()++;
    cv.notify_all();
    return v;
  }
  bool empty() {
    std::lock_guard<std::mutex> l(mu);
    return q.empty();
  }
  size_t count() override {
    std::lock_guard<std::mutex> l(mu);
    return q.size();
  }
};

inline std::vector<std::thread>& threads() {
  static thread_local std::vector<std::thread> t;
  return t;
}
inline void run(std::function<void()> f) { threads().push_back(std::thread(f)); }

// Join every process of the region; a region with no FIFO traffic for `ms` milliseconds while threads are still
// running is a deadlock (every process blocked on a full or empty stream).
inline void join_all(int ms = 3000) {
  std::atomic<int> done(0);
  const int n = (int)threads().size();
  std::thread watch([&] {
    long long last = progress().load();
    auto t0 = std::chrono::steady_clock::now();
    while (done.load() < n) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      const long long p = progress().load();
      if (p != last) {
        last = p;
        t0 = std::chrono::steady_clock::now();
      } else if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count() >
                 ms) {
        std::lock_guard<std::mutex> g(reg_mu());
        printf("DEADLOCK: no FIFO progress for %d ms; streams:\n", ms);
        for (base* b : registry())
          printf("  %-6s depth %4zu holds %4zu (peak %4zu)  readers waiting %d  writers waiting %d\n",
                 b->name.c_str(), b->depth, b->count(), b->peak, b->waiting_rd, b->waiting_wr);
        fflush(stdout);
        std::_Exit(3);
      }
    }
  });
  for (auto& t : threads()) {
    t.join();
    done++;
  }
  watch.join();
  threads().clear();
}

}  // namespace dfsim
