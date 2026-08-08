#include "finite_volume_solver.hxx"

#include "hll_solver.hxx"
#include "mhd_flux.hxx"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mhd {

FiniteVolumeSolver::FiniteVolumeSolver(
    const double cfl_number)
  : cfl_number_(cfl_number)
{
  if (cfl_number <= 0.0 || cfl_number > 1.0) {
    throw std::invalid_argument(
        "CFL number must be in the interval (0, 1].");
  }
}

double FiniteVolumeSolver::calculate_time_step(
    const UniformGrid& grid) const
{
  double maximum_speed_x = 0.0;
  double maximum_speed_y = 0.0;

  for (std::size_t index_y = 0;
       index_y < grid.number_of_cells_y();
       ++index_y) {
    for (std::size_t index_x = 0;
         index_x < grid.number_of_cells_x();
         ++index_x) {
      const ConservativeState& state =
          grid.cell(index_x, index_y);

      maximum_speed_x = std::max(
          maximum_speed_x,
          maximum_signal_speed(
              state,
              Direction::x));

      maximum_speed_y = std::max(
          maximum_speed_y,
          maximum_signal_speed(
              state,
              Direction::y));
    }
  }

  const double inverse_time_step =
      maximum_speed_x / grid.cell_width_x()
      + maximum_speed_y / grid.cell_width_y();

  if (inverse_time_step <= 0.0) {
    throw std::runtime_error(
        "Could not calculate a positive time step.");
  }

  /*
   * The multidimensional CFL condition is
   *
   * dt = CFL / (speed_x/dx + speed_y/dy).
   */
  return cfl_number_ / inverse_time_step;
}

void FiniteVolumeSolver::advance(
    UniformGrid& grid,
    const double time_step) const
{
  const std::size_t cells_x =
      grid.number_of_cells_x();

  const std::size_t cells_y =
      grid.number_of_cells_y();

  /*
   * There are nx + 1 vertical faces and ny + 1 horizontal
   * faces.
   */
  std::vector<ConservativeState> flux_x(
      (cells_x + 1) * cells_y);

  std::vector<ConservativeState> flux_y(
      cells_x * (cells_y + 1));

  const auto x_face_index =
      [cells_x](
          const std::size_t face_x,
          const std::size_t index_y) {
        return index_y * (cells_x + 1) + face_x;
      };

  const auto y_face_index =
      [cells_x](
          const std::size_t index_x,
          const std::size_t face_y) {
        return face_y * cells_x + index_x;
      };

  /*
   * Calculate fluxes at all vertical faces.
   *
   * At the domain boundary, the nearest interior state is
   * copied. This gives a transmissive/outflow boundary.
   */
  for (std::size_t index_y = 0;
       index_y < cells_y;
       ++index_y) {
    for (std::size_t face_x = 0;
         face_x <= cells_x;
         ++face_x) {
      const std::size_t left_x =
          face_x == 0 ? 0 : face_x - 1;

      const std::size_t right_x =
          face_x == cells_x
          ? cells_x - 1
          : face_x;

      flux_x[x_face_index(face_x, index_y)] =
          hll_flux(
              grid.cell(left_x, index_y),
              grid.cell(right_x, index_y),
              Direction::x);
    }
  }

  /*
   * Calculate fluxes at all horizontal faces.
   */
  for (std::size_t face_y = 0;
       face_y <= cells_y;
       ++face_y) {
    for (std::size_t index_x = 0;
         index_x < cells_x;
         ++index_x) {
      const std::size_t lower_y =
          face_y == 0 ? 0 : face_y - 1;

      const std::size_t upper_y =
          face_y == cells_y
          ? cells_y - 1
          : face_y;

      flux_y[y_face_index(index_x, face_y)] =
          hll_flux(
              grid.cell(index_x, lower_y),
              grid.cell(index_x, upper_y),
              Direction::y);
    }
  }

  /*
   * Store all updated states separately. We must not overwrite
   * a cell while neighbouring cells still need its old state.
   */
  std::vector<ConservativeState> updated_states(
      cells_x * cells_y);

  for (std::size_t index_y = 0;
       index_y < cells_y;
       ++index_y) {
    for (std::size_t index_x = 0;
         index_x < cells_x;
         ++index_x) {
      ConservativeState updated =
          grid.cell(index_x, index_y);

      const ConservativeState& flux_left =
          flux_x[x_face_index(index_x, index_y)];

      const ConservativeState& flux_right =
          flux_x[x_face_index(index_x + 1, index_y)];

      const ConservativeState& flux_lower =
          flux_y[y_face_index(index_x, index_y)];

      const ConservativeState& flux_upper =
          flux_y[y_face_index(index_x, index_y + 1)];

      for (std::size_t variable = 0;
           variable < number_of_variables;
           ++variable) {
        updated[variable] -=
            time_step / grid.cell_width_x()
            * (flux_right[variable]
               - flux_left[variable]);

        updated[variable] -=
            time_step / grid.cell_width_y()
            * (flux_upper[variable]
               - flux_lower[variable]);
      }

      if (!is_physical(updated)) {
        std::ostringstream message;

        message
            << "Nonphysical state after update in cell ("
            << index_x << ", " << index_y << ").";

        throw std::runtime_error(message.str());
      }

      updated_states[index_y * cells_x + index_x] =
          updated;
    }
  }

  /*
   * Commit the new states after every update has been
   * calculated successfully.
   */
  for (std::size_t index_y = 0;
       index_y < cells_y;
       ++index_y) {
    for (std::size_t index_x = 0;
         index_x < cells_x;
         ++index_x) {
      grid.cell(index_x, index_y) =
          updated_states[index_y * cells_x + index_x];
    }
  }
}

} // namespace mhd
