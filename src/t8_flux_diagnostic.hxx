#pragma once

#include "mhd_state.hxx"

#include <t8.h>
#include <t8_forest/t8_forest_general.h>

#include <vector>

namespace mhd {

void verify_adaptive_hll_fluxes(
    t8_forest_t forest,
    sc_MPI_Comm communicator,
    const std::vector<ConservativeState>& states);

} // namespace mhd
