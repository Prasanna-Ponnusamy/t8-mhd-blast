#include "t8_euler_step.hxx"

#include "hll_solver.hxx"
#include "mhd_flux.hxx"

#include <t8_forest/t8_forest_geometrical.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mhd {

namespace {

struct FaceOrientation {
  Direction direction;
  double outward_sign;
};

FaceOrientation get_face_orientation(
    t8_forest_t forest,
    const t8_locidx_t tree_index,
    const t8_element_t* element,
    const int face)
{
  double normal[3] = {0.0, 0.0, 0.0};

  t8_forest_element_face_normal(
      forest,
      tree_index,
      element,
      face,
      normal);

  if (std::abs(normal[0]) >
      std::abs(normal[1])) {
    return {
        Direction::x,
        normal[0] >= 0.0 ? 1.0 : -1.0
    };
  }

  return {
      Direction::y,
      normal[1] >= 0.0 ? 1.0 : -1.0
  };
}

ConservativeState coordinate_oriented_flux(
    const ConservativeState& current_state,
    const ConservativeState& neighbor_state,
    const FaceOrientation& orientation)
{
  /*
   * hll_flux always assumes a normal in the positive
   * coordinate direction.
   */
  if (orientation.outward_sign > 0.0) {
    return hll_flux(
        current_state,
        neighbor_state,
        orientation.direction);
  }

  return hll_flux(
      neighbor_state,
      current_state,
      orientation.direction);
}

void exchange_ghost_states(
    t8_forest_t forest,
    std::vector<ConservativeState>& states)
{
  sc_array_t* state_array =
      sc_array_new_data(
          states.data(),
          sizeof(ConservativeState),
          states.size());

  t8_forest_ghost_exchange_data(
      forest,
      state_array);

  sc_array_destroy(state_array);
}

} // namespace

double perform_one_adaptive_euler_step(
    t8_forest_t forest,
    const sc_MPI_Comm communicator,
    std::vector<ConservativeState>& states,
    const double cfl_number,
    const double maximum_time_step,
    const bool print_diagnostics)
{
  T8_ASSERT(t8_forest_is_committed(forest));

  if (cfl_number <= 0.0 ||
      cfl_number > 1.0) {
    throw std::invalid_argument(
        "CFL number must lie in (0,1].");
  }

  const t8_locidx_t local_element_count =
      t8_forest_get_local_num_leaf_elements(forest);

  const t8_locidx_t ghost_element_count =
      t8_forest_get_num_ghosts(forest);

  const t8_locidx_t total_element_count =
      local_element_count + ghost_element_count;

  if (states.size() !=
      static_cast<std::size_t>(total_element_count)) {
    throw std::runtime_error(
        "Euler update received an incorrectly sized state array.");
  }

  /*
   * Ensure ghost states correspond to the current local
   * states before calculating the CFL condition or fluxes.
   */
  exchange_ghost_states(
      forest,
      states);

  const t8_locidx_t local_tree_count =
      t8_forest_get_num_local_trees(forest);

  /*
   * First pass: calculate a local stable time-step estimate.
   */
  double local_time_step =
      std::numeric_limits<double>::max();

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

      const ConservativeState& state =
          states[static_cast<std::size_t>(
              current_index)];

      const double cell_volume =
          t8_forest_element_volume(
              forest,
              tree_index,
              element);

      double spectral_sum = 0.0;

      for (int face = 0;
           face < 4;
           ++face) {
        const double face_area =
            t8_forest_element_face_area(
                forest,
                tree_index,
                element,
                face);

        const FaceOrientation orientation =
            get_face_orientation(
                forest,
                tree_index,
                element,
                face);

        const double signal_speed =
            maximum_signal_speed(
                state,
                orientation.direction);

        spectral_sum +=
            face_area * signal_speed;
      }

      if (!(spectral_sum > 0.0) ||
          !std::isfinite(spectral_sum)) {
        throw std::runtime_error(
            "Invalid adaptive CFL spectral sum.");
      }

      const double cell_time_step =
          cfl_number
          * cell_volume
          / spectral_sum;

      local_time_step = std::min(
          local_time_step,
          cell_time_step);
    }
  }

  if (current_index != local_element_count) {
    throw std::runtime_error(
        "CFL iteration did not visit every local element.");
  }

  double global_time_step = 0.0;

  sc_MPI_Allreduce(
      &local_time_step,
      &global_time_step,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_MIN,
      communicator);
      global_time_step = std::min(
    global_time_step,
    maximum_time_step);

  if (!(global_time_step > 0.0) ||
      !std::isfinite(global_time_step)) {
    throw std::runtime_error(
        "Invalid global adaptive time step.");
  }

  /*
   * Compute volume-weighted global totals before the update.
   */
  ConservativeState local_total_before{};
  ConservativeState global_total_before{};

  current_index = 0;

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

      const double cell_volume =
          t8_forest_element_volume(
              forest,
              tree_index,
              element);

      const ConservativeState& state =
          states[static_cast<std::size_t>(
              current_index)];

      for (std::size_t variable = 0;
           variable < number_of_variables;
           ++variable) {
        local_total_before[variable] +=
            cell_volume * state[variable];
      }
    }
  }

  sc_MPI_Allreduce(
      local_total_before.data(),
      global_total_before.data(),
      static_cast<int>(number_of_variables),
      sc_MPI_DOUBLE,
      sc_MPI_SUM,
      communicator);

  /*
   * Second pass: calculate the finite-volume residual for
   * every local leaf.
   */
  std::vector<ConservativeState> updated_states(
      static_cast<std::size_t>(local_element_count));

  current_index = 0;

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

      ConservativeState flux_integral{};

      for (int face = 0;
           face < 4;
           ++face) {
        const FaceOrientation orientation =
            get_face_orientation(
                forest,
                tree_index,
                element,
                face);

        const double complete_face_area =
            t8_forest_element_face_area(
                forest,
                tree_index,
                element,
                face);

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
         * Transmissive physical boundary.
         */
        if (number_of_neighbors == 0) {
          const ConservativeState flux =
              hll_flux(
                  current_state,
                  current_state,
                  orientation.direction);

          for (std::size_t variable = 0;
               variable < number_of_variables;
               ++variable) {
            flux_integral[variable] +=
                orientation.outward_sign
                * complete_face_area
                * flux[variable];
          }

          continue;
        }

        /*
         * If a coarse face touches two fine cells, each
         * neighbour occupies half of the coarse face.
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
                "Invalid neighbour index during Euler update.");
          }

          const ConservativeState& neighbor_state =
              states[static_cast<std::size_t>(
                  neighbor_index)];

          const ConservativeState flux =
              coordinate_oriented_flux(
                  current_state,
                  neighbor_state,
                  orientation);

          for (std::size_t variable = 0;
               variable < number_of_variables;
               ++variable) {
            flux_integral[variable] +=
                orientation.outward_sign
                * subface_area
                * flux[variable];
          }
        }

        T8_FREE(neighbor_leaves);
        T8_FREE(neighbor_indices);
        T8_FREE(dual_faces);
      }

      ConservativeState updated =
          current_state;

      for (std::size_t variable = 0;
           variable < number_of_variables;
           ++variable) {
        updated[variable] -=
            global_time_step
            / cell_volume
            * flux_integral[variable];
      }

      if (!is_physical(updated)) {
        throw std::runtime_error(
            "Euler update produced a nonphysical state.");
      }

      updated_states[
          static_cast<std::size_t>(current_index)] =
          updated;
    }
  }

  /*
   * Commit local updated states. Ghost entries are then
   * refreshed from their owning MPI processes.
   */
  for (t8_locidx_t element_index = 0;
       element_index < local_element_count;
       ++element_index) {
    states[static_cast<std::size_t>(
        element_index)] =
        updated_states[static_cast<std::size_t>(
            element_index)];
  }

  exchange_ghost_states(
      forest,
      states);

  /*
   * Calculate post-update ranges and global conserved totals.
   */
  ConservativeState local_total_after{};
  ConservativeState global_total_after{};

  double local_minimum_density = 1.0e100;
  double local_maximum_density = 0.0;
  double local_minimum_pressure = 1.0e100;
  double local_maximum_pressure = 0.0;

  int local_invalid_states = 0;

  current_index = 0;

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

      const double cell_volume =
          t8_forest_element_volume(
              forest,
              tree_index,
              element);

      const ConservativeState& state =
          states[static_cast<std::size_t>(
              current_index)];

      if (!is_physical(state)) {
        ++local_invalid_states;
        continue;
      }

      const PrimitiveState primitive =
          conservative_to_primitive(state);

      local_minimum_density = std::min(
          local_minimum_density,
          primitive.rho);

      local_maximum_density = std::max(
          local_maximum_density,
          primitive.rho);

      local_minimum_pressure = std::min(
          local_minimum_pressure,
          primitive.pressure);

      local_maximum_pressure = std::max(
          local_maximum_pressure,
          primitive.pressure);

      for (std::size_t variable = 0;
           variable < number_of_variables;
           ++variable) {
        local_total_after[variable] +=
            cell_volume * state[variable];
      }
    }
  }

  sc_MPI_Allreduce(
      local_total_after.data(),
      global_total_after.data(),
      static_cast<int>(number_of_variables),
      sc_MPI_DOUBLE,
      sc_MPI_SUM,
      communicator);

  double global_minimum_density = 0.0;
  double global_maximum_density = 0.0;
  double global_minimum_pressure = 0.0;
  double global_maximum_pressure = 0.0;

  int global_invalid_states = 0;

  sc_MPI_Allreduce(
      &local_minimum_density,
      &global_minimum_density,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_MIN,
      communicator);

  sc_MPI_Allreduce(
      &local_maximum_density,
      &global_maximum_density,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_MAX,
      communicator);

  sc_MPI_Allreduce(
      &local_minimum_pressure,
      &global_minimum_pressure,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_MIN,
      communicator);

  sc_MPI_Allreduce(
      &local_maximum_pressure,
      &global_maximum_pressure,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_MAX,
      communicator);

  sc_MPI_Allreduce(
      &local_invalid_states,
      &global_invalid_states,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  const double mass_error =
      std::abs(
          global_total_after[density]
          - global_total_before[density]);

  const double energy_error =
      std::abs(
          global_total_after[total_energy]
          - global_total_before[total_energy]);

  const double relative_mass_error =
      mass_error
      / std::max(
          std::abs(global_total_before[density]),
          1.0e-30);

  const double relative_energy_error =
      energy_error
      / std::max(
          std::abs(global_total_before[total_energy]),
          1.0e-30);

  if (print_diagnostics) {
  t8_global_productionf(
      "\n"
      "Adaptive Euler update\n"
      "---------------------\n"
      "CFL number:                 %.6f\n"
      "Time step:                  %.16e\n"
      "Density range:              [%.8e, %.8e]\n"
      "Pressure range:             [%.8e, %.8e]\n"
      "Relative mass error:        %.8e\n"
      "Relative energy error:      %.8e\n"
      "Invalid local states:       %d\n",
      cfl_number,
      global_time_step,
      global_minimum_density,
      global_maximum_density,
      global_minimum_pressure,
      global_maximum_pressure,
      relative_mass_error,
      relative_energy_error,
      global_invalid_states);
}

  if (global_invalid_states != 0) {
    throw std::runtime_error(
        "Invalid state detected after Euler update.");
  }

  constexpr double conservation_tolerance =
      1.0e-11;

  if (relative_mass_error >
          conservation_tolerance ||
      relative_energy_error >
          conservation_tolerance) {
    throw std::runtime_error(
        "Adaptive Euler update failed conservation check.");
  }

  if (print_diagnostics) {
  t8_global_productionf(
      "Adaptive Euler update passed.\n");
}
  return global_time_step;
}

} // namespace mhd
