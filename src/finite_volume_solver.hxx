#pragma once

#include "uniform_grid.hxx"

namespace mhd {

class FiniteVolumeSolver {
public:
  explicit FiniteVolumeSolver(double cfl_number);

  double calculate_time_step(
      const UniformGrid& grid) const;

  void advance(
      UniformGrid& grid,
      double time_step) const;

private:
  double cfl_number_;
};

} // namespace mhd
