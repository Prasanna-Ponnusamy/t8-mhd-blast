#pragma once

#include "mhd_state.hxx"

#include <t8.h>
#include <t8_forest/t8_forest_general.h>

#include <vector>

namespace mhd {

struct MagneticDivergenceMetrics {
  double l1;
  double l2;
  double maximum;

  double normalized_l1;
  double normalized_maximum;
};

MagneticDivergenceMetrics compute_magnetic_divergence(
    t8_forest_t forest,
    sc_MPI_Comm communicator,
    const std::vector<ConservativeState>& states,
    std::vector<double>& divergence_values);

} // namespace mhd
