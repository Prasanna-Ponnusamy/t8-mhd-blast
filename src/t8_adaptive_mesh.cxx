#include "t8_neighbor_diagnostic.hxx"
#include "t8_geometry_diagnostic.hxx"
#include "t8_adaptive_data.hxx"
#include <t8.h>

#include <t8_cmesh/t8_cmesh.h>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_forest/t8_forest_general.h>
#include <t8_forest/t8_forest_geometrical.h>
#include <t8_forest/t8_forest_io.h>
#include <t8_schemes/t8_default/t8_default.hxx>

#include <cmath>
#include <cstdlib>

namespace {

struct AdaptationData {
  double center_x;
  double center_y;
  double refinement_radius;
  int maximum_level;
};

/*
 * t8code calls this function for every element considered
 * during adaptation.
 *
 * Return:
 *   1  -> refine
 *   0  -> leave unchanged
 *  -1  -> coarsen a family
 */
int adapt_blast_region(
    t8_forest_t forest,
    t8_forest_t forest_from,
    t8_locidx_t which_tree,
    t8_eclass_t tree_class,
    t8_locidx_t,
    const t8_scheme* scheme,
    int,
    int,
    t8_element_t* elements[])
{
  const auto* adaptation =
      static_cast<const AdaptationData*>(
          t8_forest_get_user_data(forest));

  T8_ASSERT(adaptation != nullptr);

  const int level =
      scheme->element_get_level(
          tree_class,
          elements[0]);

  if (level >= adaptation->maximum_level) {
    return 0;
  }

  double centroid[3] = {0.0, 0.0, 0.0};

  t8_forest_element_centroid(
      forest_from,
      which_tree,
      elements[0],
      centroid);

  const double distance_x =
      centroid[0] - adaptation->center_x;

  const double distance_y =
      centroid[1] - adaptation->center_y;

  const double distance =
      std::sqrt(
          distance_x * distance_x
          + distance_y * distance_y);

  /*
   * Refine the central region containing the blast and its
   * initial expansion zone.
   */
  if (distance < adaptation->refinement_radius) {
    return 1;
  }

  return 0;
}

} // namespace

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

  int mpi_size = 1;

  sc_MPI_Comm_size(
      communicator,
      &mpi_size);

  /*
   * One quadrilateral coarse tree covering [0,1] x [0,1].
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

  const t8_scheme* scheme =
      t8_scheme_new_default();

  /*
   * Begin with an 8 x 8 grid:
   *
   * level 3 -> 4^3 = 64 cells.
   */
  constexpr int initial_level = 3;

  t8_forest_t uniform_forest =
      t8_forest_new_uniform(
          coarse_mesh,
          scheme,
          initial_level,
          0,
          communicator);

  const AdaptationData adaptation{
      0.5,  // blast centre x
      0.5,  // blast centre y
      0.25, // refine inside this radius
      6     // finest permitted level
  };

  /*
   * Create an uncommitted forest and configure a complete
   * adapt-partition-balance-ghost operation.
   */
  t8_forest_t adaptive_forest;

  t8_forest_init(&adaptive_forest);

  t8_forest_set_user_data(
      adaptive_forest,
      const_cast<AdaptationData*>(&adaptation));

  /*
   * The final argument 1 enables recursive adaptation.
   * Newly created children are passed back to the callback
   * until maximum_level stops the refinement.
   */
  t8_forest_set_adapt(
      adaptive_forest,
      uniform_forest,
      adapt_blast_region,
      1);

  /*
   * Repartition the adapted elements among MPI processes.
   */
  t8_forest_set_partition(
      adaptive_forest,
      nullptr,
      0);

  /*
   * Enforce 2:1 balance: face-neighbouring cells may differ
   * by at most one refinement level.
   */
  t8_forest_set_balance(
      adaptive_forest,
      nullptr,
      0);

  /*
   * Add one face-connected ghost layer for later flux
   * calculations between MPI processes.
   */
  t8_forest_set_ghost(
      adaptive_forest,
      1,
      T8_GHOST_FACES);

  t8_forest_commit(adaptive_forest);
mhd::verify_adaptive_face_neighbors(
    adaptive_forest,
    communicator);
    mhd::verify_adaptive_geometry(
    adaptive_forest,
    communicator);
  const t8_gloidx_t global_elements =
      t8_forest_get_global_num_leaf_elements(
          adaptive_forest);

  const t8_locidx_t local_elements =
      t8_forest_get_local_num_leaf_elements(
          adaptive_forest);

  const t8_locidx_t local_ghosts =
      t8_forest_get_num_ghosts(
          adaptive_forest);

  t8_global_productionf(
      "\n"
      "Adaptive MHD blast mesh\n"
      "-----------------------\n"
      "MPI processes:          %d\n"
      "Initial level:          %d\n"
      "Maximum level:          %d\n"
      "Uniform level-6 cells:  4096\n"
      "Adaptive global cells:  %lld\n",
      mpi_size,
      initial_level,
      adaptation.maximum_level,
      static_cast<long long>(global_elements));

  t8_productionf(
      "Local cells: %lld, ghosts: %lld\n",
      static_cast<long long>(local_elements),
      static_cast<long long>(local_ghosts));

  mhd::initialize_and_write_adaptive_data(
    adaptive_forest,
    communicator,
    "t8_mhd_adaptive_initial");

t8_global_productionf(
    "\nCreated t8_mhd_adaptive_initial.pvtu\n"
    "Adaptive mesh and MHD data construction passed.\n");
  /*
   * adaptive_forest owns the source forest, coarse mesh and
   * refinement scheme after the configured operations.
   */
  t8_forest_unref(&adaptive_forest);

  sc_finalize();

  mpi_return = sc_MPI_Finalize();
  SC_CHECK_MPI(mpi_return);

  return EXIT_SUCCESS;
}
