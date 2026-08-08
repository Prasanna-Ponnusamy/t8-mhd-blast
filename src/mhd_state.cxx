#include "mhd_state.hxx"

#include <cmath>

namespace mhd {

double kinetic_energy_density(
    const PrimitiveState& primitive)
{
  const double velocity_squared =
      primitive.vx * primitive.vx
    + primitive.vy * primitive.vy
    + primitive.vz * primitive.vz;

  return 0.5 * primitive.rho * velocity_squared;
}

double magnetic_energy_density(
    const PrimitiveState& primitive)
{
  const double magnetic_field_squared =
      primitive.bx * primitive.bx
    + primitive.by * primitive.by
    + primitive.bz * primitive.bz;

  return 0.5 * magnetic_field_squared;
}

ConservativeState primitive_to_conservative(
    const PrimitiveState& primitive)
{
  ConservativeState conservative{};

  conservative[density] = primitive.rho;

  conservative[momentum_x] = primitive.rho * primitive.vx;
  conservative[momentum_y] = primitive.rho * primitive.vy;
  conservative[momentum_z] = primitive.rho * primitive.vz;

  const double internal_energy =
      primitive.pressure / (gamma_gas - 1.0);

  conservative[total_energy] =
      internal_energy
    + kinetic_energy_density(primitive)
    + magnetic_energy_density(primitive);

  conservative[magnetic_x] = primitive.bx;
  conservative[magnetic_y] = primitive.by;
  conservative[magnetic_z] = primitive.bz;

  conservative[glm_psi] = primitive.psi;

  return conservative;
}

double gas_pressure(
    const ConservativeState& conservative)
{
  const double rho = conservative[density];

  if (rho <= 0.0) {
    return -1.0;
  }

  const double momentum_squared =
      conservative[momentum_x] * conservative[momentum_x]
    + conservative[momentum_y] * conservative[momentum_y]
    + conservative[momentum_z] * conservative[momentum_z];

  const double magnetic_field_squared =
      conservative[magnetic_x] * conservative[magnetic_x]
    + conservative[magnetic_y] * conservative[magnetic_y]
    + conservative[magnetic_z] * conservative[magnetic_z];

  const double kinetic_energy =
      0.5 * momentum_squared / rho;

  const double magnetic_energy =
      0.5 * magnetic_field_squared;

  return (gamma_gas - 1.0)
       * (conservative[total_energy]
          - kinetic_energy
          - magnetic_energy);
}

PrimitiveState conservative_to_primitive(
    const ConservativeState& conservative)
{
  PrimitiveState primitive{};

  primitive.rho = conservative[density];

  primitive.vx =
      conservative[momentum_x] / primitive.rho;
  primitive.vy =
      conservative[momentum_y] / primitive.rho;
  primitive.vz =
      conservative[momentum_z] / primitive.rho;

  primitive.pressure = gas_pressure(conservative);

  primitive.bx = conservative[magnetic_x];
  primitive.by = conservative[magnetic_y];
  primitive.bz = conservative[magnetic_z];

  primitive.psi = conservative[glm_psi];

  return primitive;
}

bool is_physical(
    const ConservativeState& conservative)
{
  constexpr double density_floor = 1.0e-12;
  constexpr double pressure_floor = 1.0e-12;

  if (!std::isfinite(conservative[density]) ||
      !std::isfinite(conservative[total_energy])) {
    return false;
  }

  return conservative[density] > density_floor
      && gas_pressure(conservative) > pressure_floor;
}

} // namespace mhd
