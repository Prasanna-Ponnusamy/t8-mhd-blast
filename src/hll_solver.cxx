#include "hll_solver.hxx"

#include <algorithm>
#include <stdexcept>

namespace mhd {

HllWaveSpeeds estimate_hll_wave_speeds(
    const ConservativeState& state_left,
    const ConservativeState& state_right,
    const Direction direction)
{
  if (!is_physical(state_left)) {
    throw std::runtime_error(
        "HLL solver received a nonphysical left state.");
  }

  if (!is_physical(state_right)) {
    throw std::runtime_error(
        "HLL solver received a nonphysical right state.");
  }

  const PrimitiveState primitive_left =
      conservative_to_primitive(state_left);

  const PrimitiveState primitive_right =
      conservative_to_primitive(state_right);

  const double velocity_left =
      direction == Direction::x
      ? primitive_left.vx
      : primitive_left.vy;

  const double velocity_right =
      direction == Direction::x
      ? primitive_right.vx
      : primitive_right.vy;

  const double fast_left =
      fast_magnetosonic_speed(
          primitive_left,
          direction);

  const double fast_right =
      fast_magnetosonic_speed(
          primitive_right,
          direction);

  /*
   * Davis wave-speed estimates:
   *
   * S_L = min(v_L - cf_L, v_R - cf_R)
   * S_R = max(v_L + cf_L, v_R + cf_R)
   */
  HllWaveSpeeds speeds{};

  speeds.left = std::min(
      velocity_left - fast_left,
      velocity_right - fast_right);

  speeds.right = std::max(
      velocity_left + fast_left,
      velocity_right + fast_right);

  return speeds;
}

ConservativeState hll_flux(
    const ConservativeState& state_left,
    const ConservativeState& state_right,
    const Direction direction)
{
  const ConservativeState flux_left =
      physical_flux(state_left, direction);

  const ConservativeState flux_right =
      physical_flux(state_right, direction);

  const HllWaveSpeeds speeds =
      estimate_hll_wave_speeds(
          state_left,
          state_right,
          direction);

  /*
   * If all waves move toward the right, the interface sees
   * only the left-state flux.
   */
  if (speeds.left >= 0.0) {
    return flux_left;
  }

  /*
   * If all waves move toward the left, the interface sees
   * only the right-state flux.
   */
  if (speeds.right <= 0.0) {
    return flux_right;
  }

  /*
   * Otherwise, waves travel in both directions and the
   * interface lies inside the approximate HLL wave fan.
   *
   * F_HLL =
   *   (S_R F_L - S_L F_R
   *    + S_L S_R (U_R - U_L))
   *   / (S_R - S_L)
   */
  const double denominator =
      speeds.right - speeds.left;

  if (denominator <= 0.0) {
    throw std::runtime_error(
        "Invalid HLL wave-speed denominator.");
  }

  ConservativeState numerical_flux{};

  for (std::size_t variable = 0;
       variable < number_of_variables;
       ++variable) {
    numerical_flux[variable] =
        (
          speeds.right * flux_left[variable]
          - speeds.left * flux_right[variable]
          + speeds.left * speeds.right
            * (state_right[variable]
               - state_left[variable])
        )
        / denominator;
  }

  return numerical_flux;
}

} // namespace mhd
