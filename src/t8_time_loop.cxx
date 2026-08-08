#include "t8_time_loop.hxx"

#include "t8_euler_step.hxx"
#include "t8_magnetic_divergence.hxx"

#include <t8_forest/t8_forest_io.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace mhd {

namespace {

void write_state_vtk(
    t8_forest_t forest,
    const sc_MPI_Comm communicator,
    const std::vector<ConservativeState>& states,
    const char* prefix,
    const double simulation_time)
{
  const t8_locidx_t local_element_count =
      t8_forest_get_local_num_leaf_elements(forest);

  if (states.size() <
      static_cast<std::size_t>(local_element_count)) {
    throw std::runtime_error(
        "VTK writer received an undersized state array.");
  }

  std::vector<double> density_values(
      static_cast<std::size_t>(local_element_count));

  std::vector<double> pressure_values(
      static_cast<std::size_t>(local_element_count));

  std::vector<double> energy_values(
      static_cast<std::size_t>(local_element_count));

  std::vector<double> psi_values(
      static_cast<std::size_t>(local_element_count));

  std::vector<double> velocity_values(
      3 * static_cast<std::size_t>(local_element_count));

  std::vector<double> magnetic_values(
      3 * static_cast<std::size_t>(local_element_count));

  /*
   * Calculate one div(B) value for every local adaptive cell.
   */
  std::vector<double> divergence_values;

  const MagneticDivergenceMetrics divergence_metrics =
      compute_magnetic_divergence(
          forest,
          communicator,
          states,
          divergence_values);

  for (t8_locidx_t element_index = 0;
       element_index < local_element_count;
       ++element_index) {
    const std::size_t index =
        static_cast<std::size_t>(element_index);

    const ConservativeState& state =
        states[index];

    const PrimitiveState primitive =
        conservative_to_primitive(state);

    density_values[index] =
        primitive.rho;

    pressure_values[index] =
        primitive.pressure;

    energy_values[index] =
        state[total_energy];

    psi_values[index] =
        state[glm_psi];

    velocity_values[3 * index] =
        primitive.vx;

    velocity_values[3 * index + 1] =
        primitive.vy;

    velocity_values[3 * index + 2] =
        primitive.vz;

    magnetic_values[3 * index] =
        primitive.bx;

    magnetic_values[3 * index + 1] =
        primitive.by;

    magnetic_values[3 * index + 2] =
        primitive.bz;
  }

  constexpr int number_of_fields = 7;

  t8_vtk_data_field_t fields[number_of_fields];

  fields[0].type = T8_VTK_SCALAR;
  std::strcpy(
      fields[0].description,
      "density");
  fields[0].data = density_values.data();

  fields[1].type = T8_VTK_SCALAR;
  std::strcpy(
      fields[1].description,
      "pressure");
  fields[1].data = pressure_values.data();

  fields[2].type = T8_VTK_SCALAR;
  std::strcpy(
      fields[2].description,
      "total_energy");
  fields[2].data = energy_values.data();

  fields[3].type = T8_VTK_SCALAR;
  std::strcpy(
      fields[3].description,
      "psi");
  fields[3].data = psi_values.data();

  fields[4].type = T8_VTK_SCALAR;
  std::strcpy(
      fields[4].description,
      "divergence_B");
  fields[4].data = divergence_values.data();

  fields[5].type = T8_VTK_VECTOR;
  std::strcpy(
      fields[5].description,
      "velocity");
  fields[5].data = velocity_values.data();

  fields[6].type = T8_VTK_VECTOR;
  std::strcpy(
      fields[6].description,
      "magnetic_field");
  fields[6].data = magnetic_values.data();

  t8_forest_write_vtk_ext(
      forest,
      prefix,
      1,  // Write tree ID.
      1,  // Write MPI rank.
      1,  // Write refinement level.
      1,  // Write element ID.
      0,  // Do not write ghosts.
      0,  // Do not use curved geometry.
      0,  // Do not use the VTK API.
      number_of_fields,
      fields);

  t8_global_productionf(
      "\n"
      "Magnetic divergence at t=%.8e\n"
      "  L1 div(B):             %.8e\n"
      "  L2 div(B):             %.8e\n"
      "  Maximum |div(B)|:      %.8e\n"
      "  Normalized L1:         %.8e\n"
      "  Normalized maximum:    %.8e\n",
      simulation_time,
      divergence_metrics.l1,
      divergence_metrics.l2,
      divergence_metrics.maximum,
      divergence_metrics.normalized_l1,
      divergence_metrics.normalized_maximum);
}

}  // namespace

void run_adaptive_mhd_simulation(
    t8_forest_t forest,
    const sc_MPI_Comm communicator,
    std::vector<ConservativeState>& states,
    const double final_time,
    const double cfl_number,
    const int output_every_steps)
{
  if (final_time <= 0.0) {
    throw std::invalid_argument(
        "Final time must be positive.");
  }

  if (output_every_steps <= 0) {
    throw std::invalid_argument(
        "Output interval must be positive.");
  }

  double simulation_time = 0.0;
  int step_number = 0;
  int output_number = 0;

  char output_prefix[128];

  /*
   * Write the numbered initial state.
   */
  std::snprintf(
      output_prefix,
      sizeof(output_prefix),
      "t8_mhd_%04d",
      output_number);

  write_state_vtk(
      forest,
      communicator,
      states,
      output_prefix,
      simulation_time);

  t8_global_productionf(
      "\n"
      "Adaptive MHD time integration\n"
      "-----------------------------\n"
      "Initial time:       %.8e\n"
      "Final time:         %.8e\n"
      "CFL number:         %.4f\n"
      "Output interval:    %d steps\n",
      simulation_time,
      final_time,
      cfl_number,
      output_every_steps);

  while (simulation_time < final_time) {
    const double remaining_time =
        final_time - simulation_time;

    const double time_step =
        perform_one_adaptive_euler_step(
            forest,
            communicator,
            states,
            cfl_number,
            remaining_time,
            false);

    simulation_time += time_step;
    ++step_number;

    const bool reached_final_time =
        simulation_time >=
        final_time - 1.0e-14;

    const bool write_this_step =
        step_number % output_every_steps == 0 ||
        reached_final_time;

    if (step_number % 10 == 0 ||
        reached_final_time) {
      t8_global_productionf(
          "Step %5d  time=%.8e  dt=%.8e\n",
          step_number,
          simulation_time,
          time_step);
    }

    if (write_this_step) {
      ++output_number;

      std::snprintf(
          output_prefix,
          sizeof(output_prefix),
          "t8_mhd_%04d",
          output_number);

      write_state_vtk(
          forest,
          communicator,
          states,
          output_prefix,
          simulation_time);

      t8_global_productionf(
          "  Wrote %s.pvtu\n",
          output_prefix);
    }
  }

  t8_global_productionf(
      "\n"
      "Adaptive MHD simulation completed.\n"
      "Total steps:        %d\n"
      "Final time:         %.16e\n"
      "VTK outputs:        %d\n",
      step_number,
      simulation_time,
      output_number + 1);
}

}  // namespace mhd