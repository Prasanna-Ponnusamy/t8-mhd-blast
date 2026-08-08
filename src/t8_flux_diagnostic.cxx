#include "t8_flux_diagnostic.hxx"

#include "hll_solver.hxx"
#include "mhd_flux.hxx"

#include <t8_schemes/t8_default/t8_default.hxx>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mhd {

namespace {

Direction face_direction(const int face)
{
  /*
   * t8code quadrilateral face numbering:
   *
   * face 0: -x
   * face 1: +x
   * face 2: -y
   * face 3: +y
   */
  if (face == 0 || face == 1) {
    return Direction::x;
  }

  if (face == 2 || face == 3) {
    return Direction::y;
  }

  throw std::runtime_error(
      "Invalid quadrilateral face number.");
}

bool face_has_positive_normal(const int face)
{
  return face == 1 || face == 3;
}

bool flux_is_finite(
    const ConservativeState& flux)
{
  for (const double value : flux) {
    if (!std::isfinite(value)) {
      return false;
    }
  }

  return true;
}

double maximum_absolute_component(
    const ConservativeState& flux)
{
  double maximum_value = 0.0;

  for (const double value : flux) {
    maximum_value = std::max(
        maximum_value,
        std::abs(value));
  }

  return maximum_value;
}

} // namespace

void verify_adaptive_hll_fluxes(
    t8_forest_t forest,
    const sc_MPI_Comm communicator,
    const std::vector<ConservativeState>& states)
{
  T8_ASSERT(t8_forest_is_committed(forest));
int mpi_size = 1;

sc_MPI_Comm_size(
    communicator,
    &mpi_size);
  const t8_locidx_t local_element_count =
      t8_forest_get_local_num_leaf_elements(forest);

  const t8_locidx_t ghost_element_count =
      t8_forest_get_num_ghosts(forest);

  const t8_locidx_t total_element_count =
      local_element_count + ghost_element_count;

  if (states.size() !=
      static_cast<std::size_t>(total_element_count)) {
    throw std::runtime_error(
        "MHD state array has the wrong size.");
  }

  const t8_locidx_t local_tree_count =
      t8_forest_get_num_local_trees(forest);

  const t8_scheme* scheme =
      t8_forest_get_scheme(forest);

  int local_flux_relations = 0;
  int local_boundary_fluxes = 0;
  int local_ghost_fluxes = 0;
  int local_coarse_fine_fluxes = 0;
  int local_nonzero_energy_fluxes = 0;
  int local_invalid_fluxes = 0;

  double local_maximum_flux = 0.0;

  constexpr int number_of_faces = 4;

  /*
   * current_index follows the same local leaf ordering used
   * when the MHD state vector was initialized.
   */
  t8_locidx_t current_index = 0;

  for (t8_locidx_t tree_index = 0;
       tree_index < local_tree_count;
       ++tree_index) {
    const t8_eclass_t tree_class =
        t8_forest_get_tree_class(
            forest,
            tree_index);

    if (tree_class != T8_ECLASS_QUAD) {
      throw std::runtime_error(
          "HLL diagnostic requires quadrilateral cells.");
    }

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

      const int current_level =
          scheme->element_get_level(
              tree_class,
              element);

      for (int face = 0;
           face < number_of_faces;
           ++face) {
        const Direction direction =
            face_direction(face);

        const bool positive_normal =
            face_has_positive_normal(face);

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

        /*
         * Transmissive boundary: use the interior state on
         * both sides of the boundary face.
         */
        if (number_of_neighbors == 0) {
          const ConservativeState boundary_flux =
              hll_flux(
                  current_state,
                  current_state,
                  direction);

          ++local_boundary_fluxes;
          ++local_flux_relations;

          if (!flux_is_finite(boundary_flux)) {
            ++local_invalid_fluxes;
          }

          local_maximum_flux = std::max(
              local_maximum_flux,
              maximum_absolute_component(
                  boundary_flux));

          if (std::abs(
                  boundary_flux[total_energy])
              > 1.0e-12) {
            ++local_nonzero_energy_fluxes;
          }

          continue;
        }

        for (int neighbor_number = 0;
             neighbor_number < number_of_neighbors;
             ++neighbor_number) {
          const t8_locidx_t neighbor_index =
              neighbor_indices[neighbor_number];

          if (neighbor_index < 0 ||
              neighbor_index >= total_element_count) {
            ++local_invalid_fluxes;
            continue;
          }

          const ConservativeState& neighbor_state =
              states[static_cast<std::size_t>(
                  neighbor_index)];

          if (neighbor_index >= local_element_count) {
            ++local_ghost_fluxes;
          }

          const int neighbor_level =
              scheme->element_get_level(
                  neighbor_eclass,
                  neighbor_leaves[neighbor_number]);

          if (neighbor_level != current_level) {
            ++local_coarse_fine_fluxes;
          }

          /*
           * hll_flux assumes its normal points from the left
           * state toward the right state.
           *
           * For -x and -y faces, the neighbour is the left
           * state. For +x and +y faces, the current cell is
           * the left state.
           */
          const ConservativeState numerical_flux =
              positive_normal
              ? hll_flux(
                    current_state,
                    neighbor_state,
                    direction)
              : hll_flux(
                    neighbor_state,
                    current_state,
                    direction);

          ++local_flux_relations;

          if (!flux_is_finite(numerical_flux)) {
            ++local_invalid_fluxes;
          }

          local_maximum_flux = std::max(
              local_maximum_flux,
              maximum_absolute_component(
                  numerical_flux));

          if (std::abs(
                  numerical_flux[total_energy])
              > 1.0e-12) {
            ++local_nonzero_energy_fluxes;
          }
        }

        T8_FREE(neighbor_leaves);
        T8_FREE(neighbor_indices);
        T8_FREE(dual_faces);
      }
    }
  }

  if (current_index != local_element_count) {
    throw std::runtime_error(
        "HLL diagnostic leaf count is inconsistent.");
  }

  int global_flux_relations = 0;
  int global_boundary_fluxes = 0;
  int global_ghost_fluxes = 0;
  int global_coarse_fine_fluxes = 0;
  int global_nonzero_energy_fluxes = 0;
  int global_invalid_fluxes = 0;

  double global_maximum_flux = 0.0;

  sc_MPI_Allreduce(
      &local_flux_relations,
      &global_flux_relations,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_boundary_fluxes,
      &global_boundary_fluxes,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_ghost_fluxes,
      &global_ghost_fluxes,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_coarse_fine_fluxes,
      &global_coarse_fine_fluxes,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_nonzero_energy_fluxes,
      &global_nonzero_energy_fluxes,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_invalid_fluxes,
      &global_invalid_fluxes,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_maximum_flux,
      &global_maximum_flux,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_MAX,
      communicator);

  t8_global_productionf(
      "\n"
      "Adaptive HLL flux diagnostic\n"
      "----------------------------\n"
      "Total directed fluxes:       %d\n"
      "Boundary fluxes:             %d\n"
      "MPI ghost fluxes:            %d\n"
      "Coarse-fine fluxes:          %d\n"
      "Nonzero energy fluxes:       %d\n"
      "Invalid fluxes:              %d\n"
      "Maximum absolute component:  %.8e\n",
      global_flux_relations,
      global_boundary_fluxes,
      global_ghost_fluxes,
      global_coarse_fine_fluxes,
      global_nonzero_energy_fluxes,
      global_invalid_fluxes,
      global_maximum_flux);

  if (global_invalid_fluxes != 0) {
    throw std::runtime_error(
        "Invalid adaptive HLL flux detected.");
  }

 if (mpi_size > 1 &&
    global_ghost_fluxes == 0) {
  throw std::runtime_error(
      "No HLL fluxes were calculated across MPI interfaces.");
}

  if (global_coarse_fine_fluxes == 0) {
    throw std::runtime_error(
        "No HLL fluxes were calculated across coarse-fine faces.");
  }

  if (global_nonzero_energy_fluxes == 0) {
    throw std::runtime_error(
        "The blast boundary produced no energy flux.");
  }

  if (!(global_maximum_flux > 0.0) ||
      !std::isfinite(global_maximum_flux)) {
    throw std::runtime_error(
        "Invalid maximum HLL flux magnitude.");
  }

  t8_global_productionf(
      "Adaptive HLL flux diagnostic passed.\n");
}

} // namespace mhd
