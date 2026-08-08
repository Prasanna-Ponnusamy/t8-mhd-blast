#pragma once

#include <array>
#include <cstddef>

namespace mhd {

constexpr double gamma_gas = 5.0 / 3.0;
constexpr std::size_t number_of_variables = 9;

enum Variable : std::size_t {
  density = 0,
  momentum_x,
  momentum_y,
  momentum_z,
  total_energy,
  magnetic_x,
  magnetic_y,
  magnetic_z,
  glm_psi
};

using ConservativeState = std::array<double, number_of_variables>;

struct PrimitiveState {
  double rho;

  double vx;
  double vy;
  double vz;

  double pressure;

  double bx;
  double by;
  double bz;

  double psi;
};

ConservativeState primitive_to_conservative(
    const PrimitiveState& primitive);

PrimitiveState conservative_to_primitive(
    const ConservativeState& conservative);

double kinetic_energy_density(
    const PrimitiveState& primitive);

double magnetic_energy_density(
    const PrimitiveState& primitive);

double gas_pressure(
    const ConservativeState& conservative);

bool is_physical(
    const ConservativeState& conservative);

} // namespace mhd
