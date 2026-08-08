#include "t8_geometry_diagnostic.hxx"

#include <t8_forest/t8_forest_geometrical.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mhd {

void verify_adaptive_geometry(
    t8_forest_t forest,
    const sc_MPI_Comm communicator)
{
  T8_ASSERT(t8_forest_is_committed(forest));

  const t8_locidx_t local_tree_count =
      t8_forest_get_num_local_trees(forest);

  double local_total_volume = 0.0;
  double local_boundary_length = 0.0;

  double local_minimum_volume = 1.0e100;
  double local_maximum_volume = 0.0;

  double local_minimum_face_area = 1.0e100;
  double local_maximum_face_area = 0.0;

  int local_invalid_volumes = 0;
  int local_invalid_face_areas = 0;
  int local_invalid_normals = 0;

  constexpr int number_of_faces = 4;

  for (t8_locidx_t tree_index = 0;
       tree_index < local_tree_count;
       ++tree_index) {
    const t8_eclass_t tree_class =
        t8_forest_get_tree_class(
            forest,
            tree_index);

    if (tree_class != T8_ECLASS_QUAD) {
      throw std::runtime_error(
          "Geometry diagnostic requires quadrilateral cells.");
    }

    const t8_locidx_t elements_in_tree =
        t8_forest_get_tree_num_leaf_elements(
            forest,
            tree_index);

    for (t8_locidx_t element_index = 0;
         element_index < elements_in_tree;
         ++element_index) {
      const t8_element_t* element =
          t8_forest_get_leaf_element_in_tree(
              forest,
              tree_index,
              element_index);

      /*
       * In two dimensions, t8code calls the cell area its
       * element volume.
       */
      const double cell_volume =
          t8_forest_element_volume(
              forest,
              tree_index,
              element);

      if (!(cell_volume > 0.0) ||
          !std::isfinite(cell_volume)) {
        ++local_invalid_volumes;
      }
      else {
        local_total_volume += cell_volume;

        local_minimum_volume =
            std::min(
                local_minimum_volume,
                cell_volume);

        local_maximum_volume =
            std::max(
                local_maximum_volume,
                cell_volume);
      }

      for (int face = 0;
           face < number_of_faces;
           ++face) {
        /*
         * In 2D, the face area is the physical edge length.
         */
        const double face_area =
            t8_forest_element_face_area(
                forest,
                tree_index,
                element,
                face);

        if (!(face_area > 0.0) ||
            !std::isfinite(face_area)) {
          ++local_invalid_face_areas;
        }
        else {
          local_minimum_face_area =
              std::min(
                  local_minimum_face_area,
                  face_area);

          local_maximum_face_area =
              std::max(
                  local_maximum_face_area,
                  face_area);
        }

        double normal[3] = {0.0, 0.0, 0.0};

        t8_forest_element_face_normal(
            forest,
            tree_index,
            element,
            face,
            normal);

        const double normal_length =
            std::sqrt(
                normal[0] * normal[0]
                + normal[1] * normal[1]
                + normal[2] * normal[2]);

        /*
         * For the unit-square geometry, every face normal
         * must be unit length, axis aligned and lie in the
         * x-y plane.
         */
        const double axis_measure =
            std::abs(normal[0])
            + std::abs(normal[1]);

        if (!std::isfinite(normal_length) ||
            std::abs(normal_length - 1.0) > 1.0e-12 ||
            std::abs(axis_measure - 1.0) > 1.0e-12 ||
            std::abs(normal[2]) > 1.0e-12) {
          ++local_invalid_normals;
        }

        /*
         * Determine whether this is a physical boundary face.
         */
        const t8_element_t** neighbor_leaves = nullptr;
        int* dual_faces = nullptr;
        t8_locidx_t* neighbor_indices = nullptr;

        int neighbor_count = 0;

        t8_eclass_t neighbor_eclass =
            T8_ECLASS_INVALID;

        t8_forest_leaf_face_neighbors(
            forest,
            tree_index,
            element,
            &neighbor_leaves,
            face,
            &dual_faces,
            &neighbor_count,
            &neighbor_indices,
            &neighbor_eclass);

        if (neighbor_count == 0) {
          local_boundary_length += face_area;
        }
        else {
          T8_FREE(neighbor_leaves);
          T8_FREE(neighbor_indices);
          T8_FREE(dual_faces);
        }
      }
    }
  }

  double global_total_volume = 0.0;
  double global_boundary_length = 0.0;

  double global_minimum_volume = 0.0;
  double global_maximum_volume = 0.0;

  double global_minimum_face_area = 0.0;
  double global_maximum_face_area = 0.0;

  int global_invalid_volumes = 0;
  int global_invalid_face_areas = 0;
  int global_invalid_normals = 0;

  sc_MPI_Allreduce(
      &local_total_volume,
      &global_total_volume,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_boundary_length,
      &global_boundary_length,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_minimum_volume,
      &global_minimum_volume,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_MIN,
      communicator);

  sc_MPI_Allreduce(
      &local_maximum_volume,
      &global_maximum_volume,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_MAX,
      communicator);

  sc_MPI_Allreduce(
      &local_minimum_face_area,
      &global_minimum_face_area,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_MIN,
      communicator);

  sc_MPI_Allreduce(
      &local_maximum_face_area,
      &global_maximum_face_area,
      1,
      sc_MPI_DOUBLE,
      sc_MPI_MAX,
      communicator);

  sc_MPI_Allreduce(
      &local_invalid_volumes,
      &global_invalid_volumes,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_invalid_face_areas,
      &global_invalid_face_areas,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_invalid_normals,
      &global_invalid_normals,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  t8_global_productionf(
      "\n"
      "Adaptive finite-volume geometry\n"
      "-------------------------------\n"
      "Total domain area:          %.16e\n"
      "Physical boundary length:   %.16e\n"
      "Minimum cell area:          %.16e\n"
      "Maximum cell area:          %.16e\n"
      "Minimum face length:        %.16e\n"
      "Maximum face length:        %.16e\n"
      "Invalid cell areas:         %d\n"
      "Invalid face lengths:       %d\n"
      "Invalid face normals:       %d\n",
      global_total_volume,
      global_boundary_length,
      global_minimum_volume,
      global_maximum_volume,
      global_minimum_face_area,
      global_maximum_face_area,
      global_invalid_volumes,
      global_invalid_face_areas,
      global_invalid_normals);

  /*
   * Unit-square reference values:
   *
   * domain area       = 1
   * boundary length   = 4
   * level-6 cell area = (1/64)^2 = 1/4096
   * level-3 cell area = (1/8)^2  = 1/64
   * level-6 edge      = 1/64
   * level-3 edge      = 1/8
   */
  constexpr double expected_minimum_volume =
      1.0 / 4096.0;

  constexpr double expected_maximum_volume =
      1.0 / 64.0;

  constexpr double expected_minimum_face =
      1.0 / 64.0;

  constexpr double expected_maximum_face =
      1.0 / 8.0;

  constexpr double tolerance = 1.0e-12;

  if (global_invalid_volumes != 0 ||
      global_invalid_face_areas != 0 ||
      global_invalid_normals != 0) {
    throw std::runtime_error(
        "Invalid adaptive geometry detected.");
  }

  if (std::abs(global_total_volume - 1.0)
          > tolerance ||
      std::abs(global_boundary_length - 4.0)
          > tolerance) {
    throw std::runtime_error(
        "Adaptive mesh does not cover the unit square.");
  }

  if (std::abs(
          global_minimum_volume
          - expected_minimum_volume) > tolerance ||
      std::abs(
          global_maximum_volume
          - expected_maximum_volume) > tolerance ||
      std::abs(
          global_minimum_face_area
          - expected_minimum_face) > tolerance ||
      std::abs(
          global_maximum_face_area
          - expected_maximum_face) > tolerance) {
    throw std::runtime_error(
        "Unexpected adaptive cell or face size.");
  }

  t8_global_productionf(
      "Adaptive finite-volume geometry passed.\n");
}

} // namespace mhd
