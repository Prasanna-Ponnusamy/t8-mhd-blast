#include "mhd_state.hxx"

#include <t8.h>
#include <t8_cmesh/t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_forest/t8_forest_general.h>
#include <t8_forest/t8_forest_geometrical.h>
#include <t8_forest/t8_forest_io.h>
#include <t8_schemes/t8_default/t8_default.hxx>

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv)
{
  int mpi_return = sc_MPI_Init(&argc, &argv);
  SC_CHECK_MPI(mpi_return);

  sc_init(
      sc_MPI_COMM_WORLD,
      1,
      1,
      nullptr,
      SC_LP_ESSENTIAL);

  t8_init(SC_LP_PRODUCTION);

  const sc_MPI_Comm communicator =
      sc_MPI_COMM_WORLD;

  int mpi_rank = 0;
  int mpi_size = 1;

  sc_MPI_Comm_rank(
      communicator,
      &mpi_rank);

  sc_MPI_Comm_size(
      communicator,
      &mpi_size);

  /*
   * Construct one quadrilateral coarse tree representing
   * [0,1] x [0,1].
   */
  t8_cmesh_t coarse_mesh;

  t8_cmesh_init(&coarse_mesh);

  t8_cmesh_new_hypercube(
      &coarse_mesh,
      T8_ECLASS_QUAD,
      communicator,
      0,
      0,
      0);

  constexpr int refinement_level = 6;

  const t8_scheme* scheme =
      t8_scheme_new_default();

  t8_forest_t forest =
      t8_forest_new_uniform(
          coarse_mesh,
          scheme,
          refinement_level,
          0,
          communicator);

  const t8_locidx_t local_element_count =
      t8_forest_get_local_num_leaf_elements(forest);

  const t8_gloidx_t global_element_count =
      t8_forest_get_global_num_leaf_elements(forest);

  /*
   * The position in these arrays is the local t8code leaf
   * index. Therefore, the numerical state remains associated
   * with the corresponding forest element.
   */
  std::vector<mhd::ConservativeState> states(
      static_cast<std::size_t>(local_element_count));

  std::vector<double> density(
      static_cast<std::size_t>(local_element_count));

  std::vector<double> pressure(
      static_cast<std::size_t>(local_element_count));

  std::vector<double> total_energy(
      static_cast<std::size_t>(local_element_count));

  /*
   * VTK vector fields need three consecutive values per
   * element.
   */
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

  const t8_locidx_t number_of_local_trees =
      t8_forest_get_num_local_trees(forest);

  t8_locidx_t current_index = 0;

  /*
   * Iterate over every local tree and every leaf element
   * inside that tree.
   */
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

      const mhd::PrimitiveState primitive{
          1.0,                   // density
          0.0, 0.0, 0.0,        // velocity
          cell_pressure,
          inverse_sqrt_two,      // Bx
          inverse_sqrt_two,      // By
          0.0,                   // Bz
          0.0                    // psi
      };

      const mhd::ConservativeState state =
          mhd::primitive_to_conservative(primitive);

      const std::size_t array_index =
          static_cast<std::size_t>(current_index);

      states[array_index] = state;

      density[array_index] =
          primitive.rho;

      pressure[array_index] =
          primitive.pressure;

      total_energy[array_index] =
          state[mhd::total_energy];

      velocity[3 * array_index] =
          primitive.vx;

      velocity[3 * array_index + 1] =
          primitive.vy;

      velocity[3 * array_index + 2] =
          primitive.vz;

      magnetic_field[3 * array_index] =
          primitive.bx;

      magnetic_field[3 * array_index + 1] =
          primitive.by;

      magnetic_field[3 * array_index + 2] =
          primitive.bz;
    }
  }

  if (current_index != local_element_count) {
    t8_global_errorf(
        "ERROR: Element iteration count is inconsistent.\n");

    t8_forest_unref(&forest);
    sc_finalize();

    mpi_return = sc_MPI_Finalize();
    SC_CHECK_MPI(mpi_return);

    return EXIT_FAILURE;
  }

  /*
   * Sum the number of blast cells across all MPI processes.
   */
  int global_inside_count = 0;

  sc_MPI_Allreduce(
      &local_inside_count,
      &global_inside_count,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  /*
   * Describe the user-defined VTK fields.
   */
  constexpr int number_of_vtk_fields = 5;

  t8_vtk_data_field_t vtk_fields[number_of_vtk_fields];

  vtk_fields[0].type = T8_VTK_SCALAR;
  std::strcpy(
      vtk_fields[0].description,
      "density");
  vtk_fields[0].data = density.data();

  vtk_fields[1].type = T8_VTK_SCALAR;
  std::strcpy(
      vtk_fields[1].description,
      "pressure");
  vtk_fields[1].data = pressure.data();

  vtk_fields[2].type = T8_VTK_SCALAR;
  std::strcpy(
      vtk_fields[2].description,
      "total_energy");
  vtk_fields[2].data = total_energy.data();

  vtk_fields[3].type = T8_VTK_VECTOR;
  std::strcpy(
      vtk_fields[3].description,
      "velocity");
  vtk_fields[3].data = velocity.data();

  vtk_fields[4].type = T8_VTK_VECTOR;
  std::strcpy(
      vtk_fields[4].description,
      "magnetic_field");
  vtk_fields[4].data = magnetic_field.data();

  /*
   * Write the forest and the physical fields.
   */
  t8_forest_write_vtk_ext(
      forest,
      "t8_mhd_initial",
      1, // write tree ID
      1, // write MPI rank
      1, // write refinement level
      1, // write element ID
      0, // do not write ghost elements
      0, // do not write curved geometry
      0, // do not use VTK API output
      number_of_vtk_fields,
      vtk_fields);

  t8_global_productionf(
      "\n"
      "t8code MHD initialization\n"
      "-------------------------\n"
      "MPI processes:            %d\n"
      "Refinement level:         %d\n"
      "Global elements:          %lld\n"
      "Cells inside blast:       %d\n"
      "Expected blast cells:     124\n"
      "Minimum pressure:         0.1\n"
      "Maximum pressure:         10.0\n"
      "\n"
      "Created t8_mhd_initial.pvtu\n",
      mpi_size,
      refinement_level,
      static_cast<long long>(global_element_count),
      global_inside_count);

  t8_productionf(
      "Rank %d: local elements=%lld, "
      "local blast cells=%d\n",
      mpi_rank,
      static_cast<long long>(local_element_count),
      local_inside_count);

  if (global_element_count != 4096 ||
      global_inside_count != 124) {
    t8_global_errorf(
        "ERROR: t8code initialization does not match "
        "the uniform-grid initialization.\n");

    t8_forest_unref(&forest);
    sc_finalize();

    mpi_return = sc_MPI_Finalize();
    SC_CHECK_MPI(mpi_return);

    return EXIT_FAILURE;
  }

  t8_global_productionf(
      "\nt8code MHD initialization test passed.\n");

  t8_forest_unref(&forest);

  sc_finalize();

  mpi_return = sc_MPI_Finalize();
  SC_CHECK_MPI(mpi_return);

  return EXIT_SUCCESS;
}
