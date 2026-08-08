#pragma once

#include "mhd_state.hxx"

namespace mhd {

enum class Direction {
  x,
  y
};

/*
 * Calculate the physical ideal-MHD flux in either the
 * x-direction or y-direction.
 */
ConservativeState physical_flux(
    const ConservativeState& conservative,
    Direction direction);

/*
 * Calculate the fast magnetosonic wave speed.
 *
 * This is the fastest physical wave in ideal MHD and will
 * later be used by the HLL solver and CFL time-step condition.
 */
double fast_magnetosonic_speed(
    const PrimitiveState& primitive,
    Direction direction);

/*
 * Largest signal speed in the selected direction:
 *
 * |v_normal| + c_fast
 */
double maximum_signal_speed(
    const ConservativeState& conservative,
    Direction direction);

} // namespace mhd
