// Extraction monitor, real-time implementation.
//
// Same arithmetic, in the same order, as src/podpick/control/monitor.py (the reference).
// tests/test_native.py checks both return the same signal and action on every frame.
// No dependencies beyond the C++17 standard library; no allocation in pm_step.
#pragma once
#include <cstdint>

extern "C" {

struct PmConfig {
  double move_m;
  int32_t dilate_px;
  int32_t warn_px;
  int32_t abort_px;
  int32_t abort_ticks;
  double pressure_slow;
  double pressure_abort;
  double inside_y;
  int32_t extrude_steps;
  double extrude;  // catalogue length of the item behind its visible face
  double sag_m;    // the held item may hang this much lower than the tool says
  int32_t sag_steps;
  double follow;  // a neighbour must come forward by this share of the tool's travel to count
  // camera
  int32_t width;
  int32_t height;
  double f;
  double cx;
  double cy;
  double cam[3];
};

enum PmAction { PM_CONTINUE = 0, PM_SLOW = 1, PM_ABORT = 2 };

// ref: H*W depth after the seal; mask: H*W target mask (0/1); tool0: tool tip at the seal.
void* pm_create(const PmConfig* cfg, const float* ref, const uint8_t* mask, const double* tool0);
// One frame. Returns a PmAction and writes the number of moved neighbour pixels to *signal.
int32_t pm_step(void* m, const float* frame, const double* tool, double pressure, int32_t* signal);
// Signal only (no state change), for tests.
int32_t pm_signal(void* m, const float* frame, const double* tool);
void pm_destroy(void* m);
}
