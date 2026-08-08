#pragma once

#include <t8.h>
#include <t8_forest/t8_forest_general.h>

namespace mhd {

void initialize_and_write_adaptive_data(
    t8_forest_t forest,
    sc_MPI_Comm communicator,
    const char* output_prefix);

} // namespace mhd
