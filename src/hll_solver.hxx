#pragma once

#include "mhd_flux.hxx"
#include "mhd_state.hxx"

namespace mhd {

/*
 * HLL wave-speed estimates.
 */
struct HllWaveSpeeds {
  double left;
  double right;
};

/*
 * Estimate the slowest left-going and fastest right-going
 * MHD wave speeds at an interface.
 */
HllWaveSpeeds estimate_hll_wave_speeds(
    const ConservativeState& state_left,
    const ConservativeState& state_right,
    Direction direction);

/*
 * Calculate the HLL numerical flux between two cells.
 */
ConservativeState hll_flux(
    const ConservativeState& state_left,
    const ConservativeState& state_right,
    Direction direction);

} // namespace mhd
