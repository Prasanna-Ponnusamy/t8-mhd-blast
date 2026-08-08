#pragma once

#include "mhd_state.hxx"

#include <t8.h>
#include <t8_forest/t8_forest_general.h>

#include <vector>

namespace mhd {

void run_adaptive_mhd_simulation(
    t8_forest_t forest,
    sc_MPI_Comm communicator,
    std::vector<ConservativeState>& states,
    double final_time,
    double cfl_number,
    int output_every_steps);

} // namespace mhd
