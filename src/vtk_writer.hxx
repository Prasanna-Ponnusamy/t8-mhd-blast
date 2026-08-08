#pragma once

#include "uniform_grid.hxx"

#include <string>

namespace mhd {

void write_vtu(
    const UniformGrid& grid,
    const std::string& filename,
    double simulation_time);

} // namespace mhd
