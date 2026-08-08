#pragma once

#include "mhd_state.hxx"

#include <t8.h>
#include <t8_forest/t8_forest_general.h>

#include <vector>

namespace mhd {

double perform_one_adaptive_euler_step(
    t8_forest_t forest,
    sc_MPI_Comm communicator,
    std::vector<ConservativeState>& states,
    double cfl_number,
    double maximum_time_step,
    bool print_diagnostics);

} // namespace mhd
