#include "t8_neighbor_diagnostic.hxx"

#include <t8_schemes/t8_default/t8_default.hxx>

#include <cmath>
#include <stdexcept>

namespace mhd {

void verify_adaptive_face_neighbors(
    t8_forest_t forest,
    const sc_MPI_Comm communicator)
{
  T8_ASSERT(t8_forest_is_committed(forest));

  const t8_locidx_t local_element_count =
      t8_forest_get_local_num_leaf_elements(forest);

  const t8_locidx_t ghost_element_count =
      t8_forest_get_num_ghosts(forest);

  const t8_locidx_t total_accessible_elements =
      local_element_count + ghost_element_count;

  const t8_locidx_t local_tree_count =
      t8_forest_get_num_local_trees(forest);

  const t8_scheme* scheme =
      t8_forest_get_scheme(forest);

  int local_boundary_faces = 0;
  int local_same_level_relations = 0;
  int local_coarse_to_fine_relations = 0;
  int local_fine_to_coarse_relations = 0;
  int local_ghost_relations = 0;

  int local_invalid_indices = 0;
  int local_balance_violations = 0;
  int local_invalid_neighbor_counts = 0;

  /*
   * Our mesh contains quadrilaterals, which have four faces.
   */
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
          "Neighbour diagnostic expected quadrilateral trees.");
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

      const int element_level =
          scheme->element_get_level(
              tree_class,
              element);

      for (int face = 0;
           face < number_of_faces;
           ++face) {
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
         * No neighbour means this is a physical domain
         * boundary.
         */
        if (number_of_neighbors == 0) {
          ++local_boundary_faces;

          if (neighbor_leaves != nullptr ||
              neighbor_indices != nullptr) {
            ++local_invalid_neighbor_counts;
          }

          continue;
        }

        /*
         * In a balanced 2D quadtree, a leaf face can touch:
         *
         * - one same-level neighbour,
         * - one coarser neighbour, or
         * - two finer neighbours.
         */
        if (number_of_neighbors != 1 &&
            number_of_neighbors != 2) {
          ++local_invalid_neighbor_counts;
        }

        for (int neighbor_number = 0;
             neighbor_number < number_of_neighbors;
             ++neighbor_number) {
          const t8_locidx_t neighbor_index =
              neighbor_indices[neighbor_number];

          if (neighbor_index < 0 ||
              neighbor_index >= total_accessible_elements) {
            ++local_invalid_indices;
            continue;
          }

          if (neighbor_index >= local_element_count) {
            ++local_ghost_relations;
          }

          const int neighbor_level =
              scheme->element_get_level(
                  neighbor_eclass,
                  neighbor_leaves[neighbor_number]);

          const int level_difference =
              neighbor_level - element_level;

          if (std::abs(level_difference) > 1) {
            ++local_balance_violations;
          }

          if (level_difference == 0) {
            ++local_same_level_relations;
          }
          else if (level_difference == 1) {
            /*
             * The current element is coarse and the returned
             * neighbour is finer.
             */
            ++local_coarse_to_fine_relations;
          }
          else if (level_difference == -1) {
            /*
             * The current element is fine and its neighbour
             * is coarser.
             */
            ++local_fine_to_coarse_relations;
          }
        }

        /*
         * t8code allocated these arrays. They must be freed
         * after processing this face.
         */
        T8_FREE(neighbor_leaves);
        T8_FREE(neighbor_indices);
        T8_FREE(dual_faces);
      }
    }
  }

  int global_boundary_faces = 0;
  int global_same_level_relations = 0;
  int global_coarse_to_fine_relations = 0;
  int global_fine_to_coarse_relations = 0;
  int global_ghost_relations = 0;

  int global_invalid_indices = 0;
  int global_balance_violations = 0;
  int global_invalid_neighbor_counts = 0;

  sc_MPI_Allreduce(
      &local_boundary_faces,
      &global_boundary_faces,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_same_level_relations,
      &global_same_level_relations,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_coarse_to_fine_relations,
      &global_coarse_to_fine_relations,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_fine_to_coarse_relations,
      &global_fine_to_coarse_relations,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_ghost_relations,
      &global_ghost_relations,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_invalid_indices,
      &global_invalid_indices,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_balance_violations,
      &global_balance_violations,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  sc_MPI_Allreduce(
      &local_invalid_neighbor_counts,
      &global_invalid_neighbor_counts,
      1,
      sc_MPI_INT,
      sc_MPI_SUM,
      communicator);

  t8_global_productionf(
      "\n"
      "Adaptive face-neighbour diagnostic\n"
      "----------------------------------\n"
      "Physical boundary faces:       %d\n"
      "Same-level relations:          %d\n"
      "Coarse-to-fine relations:      %d\n"
      "Fine-to-coarse relations:      %d\n"
      "MPI ghost relations:           %d\n"
      "Invalid neighbour indices:     %d\n"
      "2:1 balance violations:        %d\n"
      "Invalid neighbour counts:      %d\n",
      global_boundary_faces,
      global_same_level_relations,
      global_coarse_to_fine_relations,
      global_fine_to_coarse_relations,
      global_ghost_relations,
      global_invalid_indices,
      global_balance_violations,
      global_invalid_neighbor_counts);

  if (global_invalid_indices != 0) {
    throw std::runtime_error(
        "Invalid adaptive neighbour index detected.");
  }

  if (global_balance_violations != 0) {
    throw std::runtime_error(
        "The adaptive forest violates 2:1 balance.");
  }

  if (global_invalid_neighbor_counts != 0) {
    throw std::runtime_error(
        "Unexpected number of face neighbours.");
  }

  if (global_coarse_to_fine_relations == 0 ||
      global_fine_to_coarse_relations == 0) {
    throw std::runtime_error(
        "No coarse-fine face relationships were found.");
  }

  t8_global_productionf(
      "Adaptive face-neighbour diagnostic passed.\n");
}

} // namespace mhd
