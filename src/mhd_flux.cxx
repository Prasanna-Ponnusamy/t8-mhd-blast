#include "mhd_flux.hxx"

#include <algorithm>
#include <cmath>

namespace mhd {

ConservativeState physical_flux(
    const ConservativeState& conservative,
    const Direction direction)
{
  const PrimitiveState primitive =
      conservative_to_primitive(conservative);

  const double rho = primitive.rho;

  const double vx = primitive.vx;
  const double vy = primitive.vy;
  const double vz = primitive.vz;

  const double pressure = primitive.pressure;

  const double bx = primitive.bx;
  const double by = primitive.by;
  const double bz = primitive.bz;

  const double magnetic_field_squared =
      bx * bx + by * by + bz * bz;

  /*
   * Total pressure:
   *
   * p_total = p_gas + |B|^2 / 2
   */
  const double total_pressure =
      pressure + 0.5 * magnetic_field_squared;

  /*
   * Scalar product v dot B.
   */
  const double velocity_dot_magnetic =
      vx * bx + vy * by + vz * bz;

  ConservativeState flux{};

  if (direction == Direction::x) {
    /*
     * Flux through a face whose normal points in x.
     */
    flux[density] = rho * vx;

    flux[momentum_x] =
        rho * vx * vx + total_pressure - bx * bx;

    flux[momentum_y] =
        rho * vx * vy - bx * by;

    flux[momentum_z] =
        rho * vx * vz - bx * bz;

    flux[total_energy] =
        (conservative[total_energy] + total_pressure) * vx
        - velocity_dot_magnetic * bx;

    /*
     * Ideal-MHD induction equation.
     *
     * The normal magnetic component Bx has zero physical
     * flux in the x-direction.
     */
    flux[magnetic_x] = 0.0;
    flux[magnetic_y] = vx * by - vy * bx;
    flux[magnetic_z] = vx * bz - vz * bx;

    /*
     * GLM terms will be added in a later stage.
     */
    flux[glm_psi] = 0.0;
  }
  else {
    /*
     * Flux through a face whose normal points in y.
     */
    flux[density] = rho * vy;

    flux[momentum_x] =
        rho * vy * vx - by * bx;

    flux[momentum_y] =
        rho * vy * vy + total_pressure - by * by;

    flux[momentum_z] =
        rho * vy * vz - by * bz;

    flux[total_energy] =
        (conservative[total_energy] + total_pressure) * vy
        - velocity_dot_magnetic * by;

    flux[magnetic_x] = vy * bx - vx * by;
    flux[magnetic_y] = 0.0;
    flux[magnetic_z] = vy * bz - vz * by;

    flux[glm_psi] = 0.0;
  }

  return flux;
}

double fast_magnetosonic_speed(
    const PrimitiveState& primitive,
    const Direction direction)
{
  /*
   * Sound speed squared:
   *
   * a^2 = gamma * p / rho
   */
  const double sound_speed_squared =
      gamma_gas * primitive.pressure / primitive.rho;

  /*
   * Total Alfvén speed squared:
   *
   * v_A^2 = |B|^2 / rho
   */
  const double magnetic_field_squared =
      primitive.bx * primitive.bx
    + primitive.by * primitive.by
    + primitive.bz * primitive.bz;

  const double alfven_speed_squared =
      magnetic_field_squared / primitive.rho;

  /*
   * Alfvén speed based on the magnetic-field component normal
   * to the cell face.
   */
  const double normal_magnetic_field =
      direction == Direction::x
      ? primitive.bx
      : primitive.by;

  const double normal_alfven_speed_squared =
      normal_magnetic_field * normal_magnetic_field
      / primitive.rho;

  /*
   * Fast magnetosonic speed:
   *
   * cf^2 = 0.5 [a^2 + vA^2
   *        + sqrt((a^2 + vA^2)^2 - 4 a^2 vAn^2)]
   */
  const double sum =
      sound_speed_squared + alfven_speed_squared;

  /*
   * Round-off errors can make the discriminant slightly
   * negative, so clamp it to zero.
   */
  const double discriminant = std::max(
      0.0,
      sum * sum
      - 4.0 * sound_speed_squared
      * normal_alfven_speed_squared);

  const double fast_speed_squared =
      0.5 * (sum + std::sqrt(discriminant));

  return std::sqrt(std::max(0.0, fast_speed_squared));
}

double maximum_signal_speed(
    const ConservativeState& conservative,
    const Direction direction)
{
  const PrimitiveState primitive =
      conservative_to_primitive(conservative);

  const double normal_velocity =
      direction == Direction::x
      ? primitive.vx
      : primitive.vy;

  return std::abs(normal_velocity)
       + fast_magnetosonic_speed(primitive, direction);
}

} // namespace mhd
