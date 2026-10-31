// Extraction monitor, real-time implementation. See monitor.h.
// Build with -ffp-contract=off (no fused multiply-add) so results match numpy bit for bit.
#include "monitor.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

struct Monitor {
  PmConfig cfg;
  std::vector<float> ref;
  std::vector<uint8_t> stat;     // dilated target mask at the seal
  std::vector<double> px, py, pz;  // target points (world) at the seal
  double tool0[3];
  // scratch, sized once
  std::vector<uint8_t> pred, tmp;
  int32_t over = 0;
  bool slowed = false;
};

void dilate(std::vector<uint8_t>& m, std::vector<uint8_t>& tmp, int H, int W, int r) {
  for (int it = 0; it < r; ++it) {
    tmp = m;
    for (int v = 0; v < H; ++v) {
      for (int u = 0; u < W; ++u) {
        const int i = v * W + u;
        uint8_t x = tmp[i];
        if (v > 0) x |= m[i - W];
        if (v < H - 1) x |= m[i + W];
        if (u > 0) x |= m[i - 1];
        if (u < W - 1) x |= m[i + 1];
        tmp[i] = x;
      }
    }
    m.swap(tmp);
  }
}

}  // namespace

extern "C" {

void* pm_create(const PmConfig* cfg, const float* ref, const uint8_t* mask, const double* tool0) {
  auto* m = new Monitor();
  m->cfg = *cfg;
  const int H = cfg->height, W = cfg->width, n = H * W;
  m->ref.assign(ref, ref + n);
  m->stat.assign(n, 0);
  std::vector<double> fx, fy, fz;
  for (int v = 0; v < H; ++v) {
    for (int u = 0; u < W; ++u) {
      const int i = v * W + u;
      if (mask[i]) m->stat[i] = 1;
      if (mask[i] && ref[i] > 0.0f) {
        const double d = static_cast<double>(ref[i]);
        fx.push_back((static_cast<double>(u) + 0.5 - cfg->cx) / cfg->f * d + cfg->cam[0]);
        fz.push_back(-(static_cast<double>(v) + 0.5 - cfg->cy) / cfg->f * d + cfg->cam[2]);
        fy.push_back(d + cfg->cam[1]);
      }
    }
  }
  // the visible face and copies pushed back along +y (the item's hidden length)
  // and lowered by up to sag_m (the held item hangs on the compliant cup)
  const int ne = cfg->extrude_steps, ng = cfg->sag_steps;
  for (int g = 0; g <= (ng > 0 ? ng : 0); ++g) {
    const double oz = ng > 0 ? (cfg->sag_m * static_cast<double>(g)) / static_cast<double>(ng) : 0.0;
    for (int k = 0; k <= (ne > 0 ? ne : 0); ++k) {
      const double oy = ne > 0 ? (cfg->extrude * static_cast<double>(k)) / static_cast<double>(ne) : 0.0;
      for (size_t j = 0; j < fx.size(); ++j) {
        m->px.push_back(fx[j]);
        m->py.push_back(fy[j] + oy);
        m->pz.push_back(fz[j] - oz);
      }
    }
  }
  m->tmp.assign(n, 0);
  dilate(m->stat, m->tmp, H, W, cfg->dilate_px);
  m->pred.assign(n, 0);
  std::memcpy(m->tool0, tool0, sizeof(m->tool0));
  return m;
}

int32_t pm_signal(void* handle, const float* frame, const double* tool) {
  auto* m = static_cast<Monitor*>(handle);
  const PmConfig& c = m->cfg;
  const int H = c.height, W = c.width, n = H * W;
  const double dx = tool[0] - m->tool0[0], dy = tool[1] - m->tool0[1], dz = tool[2] - m->tool0[2];
  std::fill(m->pred.begin(), m->pred.end(), 0);
  for (size_t k = 0; k < m->px.size(); ++k) {
    const double rx = (m->px[k] + dx) - c.cam[0];
    const double ry = (m->py[k] + dy) - c.cam[1];
    const double rz = (m->pz[k] + dz) - c.cam[2];
    if (!(ry > 1e-6)) continue;
    const double fu = std::floor(rx / ry * c.f + c.cx);
    const double fv = std::floor(-rz / ry * c.f + c.cy);
    if (fu < 0 || fu >= W || fv < 0 || fv >= H) continue;
    m->pred[static_cast<int>(fv) * W + static_cast<int>(fu)] = 1;
  }
  dilate(m->pred, m->tmp, H, W, c.dilate_px);
  const double pulled = m->tool0[1] - tool[1];
  const double follow = c.follow * pulled;
  const double need = c.move_m > follow ? c.move_m : follow;  // max(move_m, follow * pulled)
  int32_t count = 0;
  for (int i = 0; i < n; ++i) {
    if (m->stat[i] || m->pred[i]) continue;
    if (!(m->ref[i] > 0.0f) || !(frame[i] > 0.0f)) continue;
    const double diff = static_cast<double>(m->ref[i]) - static_cast<double>(frame[i]);
    if (diff > need) ++count;
  }
  return count;
}

int32_t pm_step(void* handle, const float* frame, const double* tool, double pressure, int32_t* signal) {
  auto* m = static_cast<Monitor*>(handle);
  const PmConfig& c = m->cfg;
  const int32_t s = pm_signal(handle, frame, tool);
  if (signal) *signal = s;
  m->over = (s >= c.abort_px) ? m->over + 1 : 0;
  if (m->over >= c.abort_ticks) return PM_ABORT;
  if (pressure < c.pressure_abort && tool[1] > c.inside_y) return PM_ABORT;
  if (s >= c.warn_px || pressure < c.pressure_slow) m->slowed = true;
  return m->slowed ? PM_SLOW : PM_CONTINUE;
}

void pm_destroy(void* handle) { delete static_cast<Monitor*>(handle); }

}  // extern "C"
