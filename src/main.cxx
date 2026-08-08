#include "finite_volume_solver.hxx"
#include "mhd_state.hxx"
#include "uniform_grid.hxx"
#include "vtk_writer.hxx"
#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>

int main()
{
  try {
    constexpr std::size_t cells_x = 64;
    constexpr std::size_t cells_y = 64;

    constexpr double final_time = 0.02;
    constexpr double cfl_number = 0.35;

    mhd::UniformGrid grid(
        cells_x,
        cells_y,
        0.0, 1.0,
        0.0, 1.0);

    grid.initialize_mhd_blast(
        0.5,
        0.5,
        0.1,
        10.0,
        0.1);

    grid.write_csv("mhd_blast_initial.csv");
mhd::write_vtu(
    grid,
    "mhd_blast_initial.vtu",
    0.0);
    mhd::FiniteVolumeSolver solver(cfl_number);

    double simulation_time = 0.0;
    std::size_t time_step_number = 0;

    std::cout << std::scientific
              << std::setprecision(8);

    while (simulation_time < final_time) {
      double time_step =
          solver.calculate_time_step(grid);

      /*
       * Shorten the final step so that the simulation stops
       * exactly at final_time.
       */
      time_step = std::min(
          time_step,
          final_time - simulation_time);

      solver.advance(grid, time_step);

      simulation_time += time_step;
      ++time_step_number;

      double minimum_density = 1.0e100;
      double maximum_density = -1.0e100;
      double minimum_pressure = 1.0e100;
      double maximum_pressure = -1.0e100;

      for (std::size_t index_y = 0;
           index_y < grid.number_of_cells_y();
           ++index_y) {
        for (std::size_t index_x = 0;
             index_x < grid.number_of_cells_x();
             ++index_x) {
          const auto primitive =
              mhd::conservative_to_primitive(
                  grid.cell(index_x, index_y));

          minimum_density = std::min(
              minimum_density,
              primitive.rho);

          maximum_density = std::max(
              maximum_density,
              primitive.rho);

          minimum_pressure = std::min(
              minimum_pressure,
              primitive.pressure);

          maximum_pressure = std::max(
              maximum_pressure,
              primitive.pressure);
        }
      }

      std::cout
          << "Step " << time_step_number
          << "  time=" << simulation_time
          << "  dt=" << time_step
          << "  rho=[" << minimum_density
          << ", " << maximum_density << "]"
          << "  p=[" << minimum_pressure
          << ", " << maximum_pressure << "]\n";
    }

    grid.write_csv("mhd_blast_final.csv");
mhd::write_vtu(
    grid,
    "mhd_blast_final.vtu",
    simulation_time);
    std::cout
        << "\nSimulation completed successfully.\n"
        << "Number of steps: "
        << time_step_number << '\n'
        << "Final time: "
        << simulation_time << '\n'
        << "Created mhd_blast_initial.csv\n"
        << "Created mhd_blast_final.csv\n"
<< "Created mhd_blast_initial.vtu\n"
    << "Created mhd_blast_final.vtu\n";
  }
  catch (const std::exception& exception) {
    std::cerr
        << "ERROR: " << exception.what() << '\n';
    return 1;
  }

  return 0;
}
