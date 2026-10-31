// Unit tests for the monitor without Python: a synthetic bin, a target pulled straight out,
// and a neighbour that comes forward with it. Plus the per-frame latency.
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "monitor.h"

namespace {

PmConfig config() {
  PmConfig c{};
  c.move_m = 0.012;
  c.dilate_px = 2;
  c.warn_px = 6;
  c.abort_px = 14;
  c.abort_ticks = 2;
  c.pressure_slow = 0.30;
  c.pressure_abort = 0.16;
  c.inside_y = -0.02;
  c.extrude_steps = 0;
  c.extrude = 0.0;
  c.sag_m = 0.0;
  c.sag_steps = 0;
  c.follow = 0.0;
  c.width = 64;
  c.height = 48;
  c.f = 32.0 / std::tan(46.0 * M_PI / 180.0 / 2.0);
  c.cx = 32.0;
  c.cy = 24.0;
  c.cam[0] = 0.0;
  c.cam[1] = -0.42;
  c.cam[2] = 0.12;
  return c;
}

}  // namespace

int main() {
  const PmConfig c = config();
  const int W = c.width, H = c.height, n = W * H;
  std::vector<float> ref(n, 0.70f);  // back wall
  std::vector<uint8_t> mask(n, 0);
  for (int v = 20; v < 30; ++v)
    for (int u = 26; u < 38; ++u) {
      ref[v * W + u] = 0.50f;  // the target's front face
      mask[v * W + u] = 1;
    }
  for (int v = 12; v < 20; ++v)
    for (int u = 26; u < 38; ++u) ref[v * W + u] = 0.52f;  // an item resting on it
  const double tool0[3] = {0.0, 0.08, 0.12};
  void* m = pm_create(&c, ref.data(), mask.data(), tool0);

  // 1. nothing moves: no signal, continue
  int32_t sig = -1;
  assert(pm_step(m, ref.data(), tool0, 0.8, &sig) == PM_CONTINUE && sig == 0);

  // 2. the target comes 5 cm out: its own pixels are predicted, still no signal
  std::vector<float> f2 = ref;
  for (int v = 20; v < 30; ++v)
    for (int u = 26; u < 38; ++u) f2[v * W + u] = 0.45f;
  const double tool1[3] = {0.0, 0.03, 0.12};
  assert(pm_step(m, f2.data(), tool1, 0.8, &sig) == PM_CONTINUE);
  assert(sig == 0);

  // 3. the item on top comes 3 cm forward with it: slow, then abort on the second frame
  std::vector<float> f3 = f2;
  for (int v = 12; v < 17; ++v)
    for (int u = 26; u < 38; ++u) f3[v * W + u] = 0.49f;
  int32_t a1 = pm_step(m, f3.data(), tool1, 0.8, &sig);
  assert(sig >= 14);
  assert(a1 == PM_SLOW);
  int32_t a2 = pm_step(m, f3.data(), tool1, 0.8, &sig);
  assert(a2 == PM_ABORT);
  pm_destroy(m);

  // 4. losing vacuum inside the bin aborts; outside it does not
  m = pm_create(&c, ref.data(), mask.data(), tool0);
  assert(pm_step(m, ref.data(), tool0, 0.10, &sig) == PM_ABORT);
  const double outside[3] = {0.0, -0.10, 0.12};
  assert(pm_step(m, ref.data(), outside, 0.10, &sig) == PM_SLOW);

  // latency
  const int reps = 2000;
  auto t0 = std::chrono::steady_clock::now();
  int32_t acc = 0;
  for (int i = 0; i < reps; ++i) acc += pm_signal(m, f3.data(), tool1);
  auto t1 = std::chrono::steady_clock::now();
  const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
  pm_destroy(m);
  std::printf("monitor tests passed; %.1f us per frame (64x48), checksum %d\n", us, acc);
  return 0;
}
