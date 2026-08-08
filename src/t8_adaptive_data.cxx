#include "t8_adaptive_data.hxx"
#include "t8_flux_diagnostic.hxx"
#include "mhd_state.hxx"
#include "t8_euler_step.hxx"
#include <t8_forest/t8_forest_geometrical.h>
#include <t8_forest/t8_forest_io.h>
#include "t8_time_loop.hxx"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <type_traits>

namespace mhd {

void initialize_and_write_adaptive_data(
    t8_forest_t forest,
    const sc_MPI_Comm communicator,
    const char* output_prefix)
{
  T8_ASSERT(t8_forest_is_committed(forest));

  const t8_locidx_t local_element_count =
      t8_forest_get_local_num_leaf_elements(forest);
const t8_locidx_t ghost_element_count =
    t8_forest_get_num_ghosts(forest);

const t8_locidx_t total_element_count =
    local_element_count + ghost_element_count;
  /*
   * One conservative MHD state per local adaptive leaf.
   * The ordering is the local t8code space-filling-curve
   * ordering.
   */
 /*
 * Local states occupy indices:
 *
 *   0, ..., local_element_count - 1
 *
 * Ghost states occupy:
 *
 *   local_element_count, ..., total_element_count - 1
 */
std::vector<ConservativeState> states(
    static_cast<std::size_t>(total_element_count));
  std::vector<double> density(
      static_cast<std::size_t>(local_element_count));

  std::vector<double> pressure(
      static_cast<std::size_t>(local_element_count));

  std::vector<double> total_energy_values(
      static_cast<std::size_t>(local_element_count));

  std::vector<double> velocity(
      3 * static_cast<std::size_t>(local_element_count));

  std::vector<double> magnetic_field(
      3 * static_cast<std::size_t>(local_element_count));

  constexpr double center_x = 0.5;
  constexpr double center_y = 0.5;
  constexpr double blast_radius = 0.1;

  constexpr double inner_pressure = 10.0;
  constexpr double outer_pressure = 0.1;

  constexpr double inverse_sqrt_two =
      0.70710678118654752440;

  int local_inside_count = 0;

  double local_minimum_pressure = 1.0e100;
  double local_maximum_pressure = -1.0e100;

  const t8_locidx_t number_of_local_trees =
      t8_forest_get_num_local_trees(forest);

  t8_locidx_t current_index = 0;

  for (t8_locidx_t tree_index = 0;
       tree_index < number_of_local_trees;
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

      double centroid[3] = {0.0, 0.0, 0.0};

      t8_forest_element_centroid(
          forest,
          tree_index,
          element,
          centroid);

      const double distance_x =
          centroid[0] - center_x;

      const double distance_y =
          centroid[1] - center_y;

      const double distance =
          std::sqrt(
              distance_x * distance_x
              + distance_y * distance_y);

      const bool inside_blast =
          distance <= blast_radius;

      if (inside_blast) {
        ++local_inside_count;
      }

      const double cell_pressure =
          inside_blast
          ? inner_pressure
          : outer_pressure;

      const PrimitiveState primitive{
          1.0,                   // density
          0.0, 0.0, 0.0,        // velocity
          cell_pressure,
          inverse_sqrt_two,      // Bx
          inverse_sqrt_two,      // By
          0.0,                   // Bz
          0.0                    // GLM psi
      };

      const ConservativeState state =
          primitive_to_conservative(primitive);

      if (!is_physical(state)) {
        throw std::runtime_error(
            "Nonphysical adaptive initial state.");
      }

      const std::size_t index =
          static_cast<std::size_t>(current_index);

      states[index] = state;

      density[index] =
          primitive.rho;

      pressure[index] =
          primitive.pressure;

      total_energy_values[index] =
          state[mhd::total_energy];

      velocity[3 * index] =
          primitive.vx;

      velocity[3 * index + 1] =
          primitive.vy;

      velocity[3 * index + 2] =
          primitive.vz;

      magnetic_field[3 * index] =
          primitive.bx;

      magnetic_field[3 * index + 1] =
          primitive.by;

      magnetic_field[3 * index + 2] =
          primitive.bz;

      local_minimum_pressure =
          std::min(
              local_minimum_pressure,
              primitive.pressure);

      local_maximum_pressure =
          std::max(
              local_maximum_pressure,
              primitive.pressure);
    }
  }

  if (current_index != local_element_count) {
    throw std::runtime_error(
        "Adaptive element iteration count is incorrect.");
  }
/*
 * std::array<double, 9> can safely be transmitted as a
 * contiguous block of bytes because it is trivially copyable.
 */
static_assert(
    std::is_trivially_copyable_v<ConservativeState>,
    "ConservativeState must be trivially copyable.");

/*
 * Wrap the existing vector in an sc_array. The wrapper does
 * not take ownership of the vector's memory.
 */
sc_array_t* state_array =
    sc_array_new_data(
        states.data(),
        sizeof(ConservativeState),
        static_cast<std::size_t>(total_element_count));

/*
 * This collective MPI operation fills every ghost entry with
 * the state stored on the MPI process that owns that leaf.
 */
t8_forest_ghost_exchange_data(
    forest,
    state_array);

/*
 * Destroy only the wrapper. states continues to own its
 * underlying memory.
 */
sc_array_destroy(state_array);

/*
 * Validate all states received into the local ghost layer.
 */
int local_invalid_ghost_states = 0;
int local_high_pressure_ghosts = 0;

double local_ghost_minimum_pressure = 1.0e100;
double local_ghost_maximum_pressure = -1.0e100;

for (t8_locidx_t ghost_number = 0;
     ghost_number < ghost_element_count;
     ++ghost_number) {
  const t8_locidx_t combined_index =
      local_element_count + ghost_number;

  const ConservativeState& ghost_state =
      states[static_cast<std::size_t>(combined_index)];

  if (!is_physical(ghost_state)) {
    ++local_invalid_ghost_states;
    continue;
  }

  const double ghost_pressure =
      gas_pressure(ghost_state);

  local_ghost_minimum_pressure =
      std::min(
          local_ghost_minimum_pressure,
          ghost_pressure);

  local_ghost_maximum_pressure =
      std::max(
          local_ghost_maximum_pressure,
          ghost_pressure);

  if (ghost_pressure > 1.0) {
    ++local_high_pressure_ghosts;
  }
}
(void) local_ghost_minimum_pressure;
(void) local_ghost_maximum_pressure;
int global_ghost_count = 0;
int global_invalid_ghost_states = 0;
int global_high_pressure_ghosts = 0;

const int local_ghost_count_as_int =
    static_cast<int>(ghost_element_count);

sc_MPI_Allreduce(
    &local_ghost_count_as_int,
    &global_ghost_count,
    1,
    sc_MPI_INT,
    sc_MPI_SUM,
    communicator);

sc_MPI_Allreduce(
    &local_invalid_ghost_states,
    &global_invalid_ghost_states,
    1,
    sc_MPI_INT,
    sc_MPI_SUM,
    communicator);

sc_MPI_Allreduce(
    &local_high_pressure_ghosts,
    &global_high_pressure_ghosts,
    1,
    sc_MPI_INT,
    sc_MPI_SUM,
    communicator);

t8_global_productionf(
    "\n"
    "MHD ghost-state exchange\n"
    "------------------------\n"
    "Total ghost copies:          %d\n"
    "High-pressure ghost copies:  %d\n"
    "Invalid ghost states:        %d\n",
    global_ghost_count,
    global_high_pressure_ghosts,
    global_invalid_ghost_states);

if (global_invalid_ghost_states != 0) {
  throw std::runtime_error(
      "MPI ghost exchange produced invalid MHD states.");
}

t8_global_productionf(
    "MHD ghost-state exchange passed.\n");
run_adaptive_mhd_simulation(
    forest,
    communicator,
    states,
    0.02, // final time
    0.25, // CFL number
    10);  // write every 10 steps
   verify_adaptive_hll_fluxes(
    forest,
    communicator,
    states);
  /*
   * Combine verification information across all MPI ranks.
   */
  int global_inside_count = 0;

  double global_minimum_pressure = 0.0;
  double global_maximum_pressure = 0.0;

  sc_MPI_Allreduce(
      &local_inside_count,
      &global_inside_count,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
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

  /*
   * Set up the fields passed to t8code's extended VTK writer.
   */
  constexpr int number_of_fields = 5;

  t8_vtk_data_field_t fields[number_of_fields];

  fields[0].type = T8_VTK_SCALAR;
  std::strcpy(
      fields[0].description,
      "density");
  fields[0].data = density.data();

  fields[1].type = T8_VTK_SCALAR;
  std::strcpy(
      fields[1].description,
      "pressure");
  fields[1].data = pressure.data();

  fields[2].type = T8_VTK_SCALAR;
  std::strcpy(
      fields[2].description,
      "total_energy");
  fields[2].data =
      total_energy_values.data();

  fields[3].type = T8_VTK_VECTOR;
  std::strcpy(
      fields[3].description,
      "velocity");
  fields[3].data = velocity.data();

  fields[4].type = T8_VTK_VECTOR;
  std::strcpy(
      fields[4].description,
      "magnetic_field");
  fields[4].data = magnetic_field.data();

  t8_forest_write_vtk_ext(
      forest,
      output_prefix,
      1, // tree ID
      1, // MPI rank
      1, // refinement level
      1, // element ID
      0, // ghost elements
      0, // curved geometry
      0, // VTK API output
      number_of_fields,
      fields);

  t8_global_productionf(
      "\n"
      "Adaptive MHD initialization\n"
      "---------------------------\n"
      "Cells inside blast:       %d\n"
      "Expected blast cells:     124\n"
      "Minimum pressure:         %.6f\n"
      "Maximum pressure:         %.6f\n",
      global_inside_count,
      global_minimum_pressure,
      global_maximum_pressure);
   

  if (global_inside_count != 124) {
    throw std::runtime_error(
        "Adaptive mesh does not reproduce the expected "
        "124 blast cells.");
  }

  if (std::abs(global_minimum_pressure - 0.1)
          > 1.0e-12 ||
      std::abs(global_maximum_pressure - 10.0)
          > 1.0e-12) {
    throw std::runtime_error(
        "Incorrect adaptive pressure range.");
  }

  t8_global_productionf(
      "Adaptive MHD state initialization passed.\n");
}

} // namespace mhd
