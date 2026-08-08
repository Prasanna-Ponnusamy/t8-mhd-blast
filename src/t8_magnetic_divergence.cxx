#include "t8_magnetic_divergence.hxx"

#include <t8_forest/t8_forest_geometrical.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mhd {

namespace {

double magnetic_normal_component(
    const ConservativeState& state,
    const double normal[3])
{
  return state[magnetic_x] * normal[0]
       + state[magnetic_y] * normal[1]
       + state[magnetic_z] * normal[2];
}

} // namespace

MagneticDivergenceMetrics compute_magnetic_divergence(
    t8_forest_t forest,
    const sc_MPI_Comm communicator,
    const std::vector<ConservativeState>& states,
    std::vector<double>& divergence_values)
{
  T8_ASSERT(t8_forest_is_committed(forest));

  const t8_locidx_t local_element_count =
      t8_forest_get_local_num_leaf_elements(forest);

  const t8_locidx_t ghost_element_count =
      t8_forest_get_num_ghosts(forest);

  const t8_locidx_t total_element_count =
      local_element_count + ghost_element_count;

  if (states.size() !=
      static_cast<std::size_t>(total_element_count)) {
    throw std::runtime_error(
        "Divergence calculation received an invalid state array.");
  }

  divergence_values.assign(
      static_cast<std::size_t>(local_element_count),
      0.0);

  double local_volume = 0.0;
  double local_l1_integral = 0.0;
  double local_l2_integral = 0.0;
  double local_maximum = 0.0;

  double local_normalized_integral = 0.0;
  double local_normalized_maximum = 0.0;

  const t8_locidx_t local_tree_count =
      t8_forest_get_num_local_trees(forest);

  t8_locidx_t current_index = 0;

  for (t8_locidx_t tree_index = 0;
       tree_index < local_tree_count;
       ++tree_index) {
    const t8_locidx_t elements_in_tree =
        t8_forest_get_tree_num_leaf_elements(
            forest,
            tree_index);

    for (t8_locidx_t element_index = 0;
         element_index < elements_in_tree;
         ++element_index, ++current_index) {
      const t8_element_t* element =
          t8_forest_get_leaf_element_in_tree(
              forest,
              tree_index,
              element_index);

      const ConservativeState& current_state =
          states[static_cast<std::size_t>(
              current_index)];

      const double cell_volume =
          t8_forest_element_volume(
              forest,
              tree_index,
              element);

      double magnetic_flux_integral = 0.0;

      for (int face = 0;
           face < 4;
           ++face) {
        double outward_normal[3] = {
            0.0, 0.0, 0.0
        };

        t8_forest_element_face_normal(
            forest,
            tree_index,
            element,
            face,
            outward_normal);

        const double complete_face_area =
            t8_forest_element_face_area(
                forest,
                tree_index,
                element,
                face);

        const double current_normal_field =
            magnetic_normal_component(
                current_state,
                outward_normal);

        const t8_element_t** neighbor_leaves = nullptr;
        int* dual_faces = nullptr;
        t8_locidx_t* neighbor_indices = nullptr;

        int number_of_neighbors = 0;

        t8_eclass_t neighbor_eclass =
            T8_ECLASS_INVALID;

        t8_forest_leaf_face_neighbors(
            forest,
            tree_index,
            element,
            &neighbor_leaves,
            face,
            &dual_faces,
            &number_of_neighbors,
            &neighbor_indices,
            &neighbor_eclass);

        (void) neighbor_eclass;

        /*
         * Transmissive boundary: the face value equals the
         * interior magnetic field.
         */
        if (number_of_neighbors == 0) {
          magnetic_flux_integral +=
              complete_face_area
              * current_normal_field;

          continue;
        }

        /*
         * A coarse face may contain two fine subfaces.
         */
        const double subface_area =
            complete_face_area
            / static_cast<double>(
                number_of_neighbors);

        for (int neighbor_number = 0;
             neighbor_number < number_of_neighbors;
             ++neighbor_number) {
          const t8_locidx_t neighbor_index =
              neighbor_indices[neighbor_number];

          if (neighbor_index < 0 ||
              neighbor_index >= total_element_count) {
            throw std::runtime_error(
                "Invalid neighbour index during div(B) calculation.");
          }

          const ConservativeState& neighbor_state =
              states[static_cast<std::size_t>(
                  neighbor_index)];

          const double neighbor_normal_field =
              magnetic_normal_component(
                  neighbor_state,
                  outward_normal);

          /*
           * Arithmetic face-centred magnetic field.
           */
          const double face_normal_field =
              0.5
              * (current_normal_field
                 + neighbor_normal_field);

          magnetic_flux_integral +=
              subface_area
              * face_normal_field;
        }

        T8_FREE(neighbor_leaves);
        T8_FREE(neighbor_indices);
        T8_FREE(dual_faces);
      }

      const double divergence =
          magnetic_flux_integral
          / cell_volume;

      divergence_values[
          static_cast<std::size_t>(current_index)] =
          divergence;

      const double absolute_divergence =
          std::abs(divergence);

      const double magnetic_magnitude =
          std::sqrt(
              current_state[magnetic_x]
                * current_state[magnetic_x]
              + current_state[magnetic_y]
                * current_state[magnetic_y]
              + current_state[magnetic_z]
                * current_state[magnetic_z]);

      /*
       * Characteristic adaptive cell length.
       */
      const double cell_length =
          std::sqrt(cell_volume);

      const double normalized_divergence =
          cell_length
          * absolute_divergence
          / std::max(
              magnetic_magnitude,
              1.0e-14);

      local_volume += cell_volume;

      local_l1_integral +=
          cell_volume
          * absolute_divergence;

      local_l2_integral +=
          cell_volume
          * divergence
          * divergence;

      local_normalized_integral +=
          cell_volume
          * normalized_divergence;

      local_maximum = std::max(
          local_maximum,
          absolute_divergence);

      local_normalized_maximum = std::max(
          local_normalized_maximum,
          normalized_divergence);
    }
  }

  if (current_index != local_element_count) {
    throw std::runtime_error(
        "Divergence calculation missed local elements.");
  }

  double global_volume = 0.0;
  double global_l1_integral = 0.0;
  double global_l2_integral = 0.0;
  double global_maximum = 0.0;

  double global_normalized_integral = 0.0;
  double global_normalized_maximum = 0.0;

  sc_MPI_Allreduce(
      &local_volume,
      &global_volume,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_l1_integral,
      &global_l1_integral,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_l2_integral,
      &global_l2_integral,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_maximum,
      &global_maximum,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_MAX,
      communicator);

  sc_MPI_Allreduce(
      &local_normalized_integral,
      &global_normalized_integral,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_normalized_maximum,
      &global_normalized_maximum,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_MAX,
      communicator);

  MagneticDivergenceMetrics metrics{};

  metrics.l1 =
      global_l1_integral
      / global_volume;

  metrics.l2 =
      std::sqrt(
          global_l2_integral
          / global_volume);

  metrics.maximum =
      global_maximum;

  metrics.normalized_l1 =
      global_normalized_integral
      / global_volume;

  metrics.normalized_maximum =
      global_normalized_maximum;

  return metrics;
}

} // namespace mhd
