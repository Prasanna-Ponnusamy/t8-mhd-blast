#pragma once

#include <t8.h>
#include <t8_forest/t8_forest_general.h>

namespace mhd {

void verify_adaptive_face_neighbors(
    t8_forest_t forest,
    sc_MPI_Comm communicator);

} // namespace mhd
