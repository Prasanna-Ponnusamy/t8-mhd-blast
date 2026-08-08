#include "uniform_grid.hxx"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace mhd {

UniformGrid::UniformGrid(
    const std::size_t number_of_cells_x,
    const std::size_t number_of_cells_y,
    const double x_min,
    const double x_max,
    const double y_min,
    const double y_max)
  : number_of_cells_x_(number_of_cells_x),
    number_of_cells_y_(number_of_cells_y),
    x_min_(x_min),
    x_max_(x_max),
    y_min_(y_min),
    y_max_(y_max),
    cell_width_x_(
        (x_max - x_min)
        / static_cast<double>(number_of_cells_x)),
    cell_width_y_(
        (y_max - y_min)
        / static_cast<double>(number_of_cells_y)),
    cells_(number_of_cells_x * number_of_cells_y)
{
  if (number_of_cells_x == 0 ||
      number_of_cells_y == 0) {
    throw std::invalid_argument(
        "Grid dimensions must be positive.");
  }

  if (x_max <= x_min || y_max <= y_min) {
    throw std::invalid_argument(
        "Invalid grid domain.");
  }
}

std::size_t UniformGrid::flatten_index(
    const std::size_t index_x,
    const std::size_t index_y) const
{
  if (index_x >= number_of_cells_x_ ||
      index_y >= number_of_cells_y_) {
    throw std::out_of_range(
        "UniformGrid cell index is out of range.");
  }

  return index_y * number_of_cells_x_ + index_x;
}

const ConservativeState& UniformGrid::cell(
    const std::size_t index_x,
    const std::size_t index_y) const
{
  return cells_[flatten_index(index_x, index_y)];
}

ConservativeState& UniformGrid::cell(
    const std::size_t index_x,
    const std::size_t index_y)
{
  return cells_[flatten_index(index_x, index_y)];
}

double UniformGrid::cell_center_x(
    const std::size_t index_x) const
{
  return x_min_
       + (static_cast<double>(index_x) + 0.5)
       * cell_width_x_;
}

double UniformGrid::cell_center_y(
    const std::size_t index_y) const
{
  return y_min_
       + (static_cast<double>(index_y) + 0.5)
       * cell_width_y_;
}

void UniformGrid::initialize_mhd_blast(
    const double center_x,
    const double center_y,
    const double blast_radius,
    const double inner_pressure,
    const double outer_pressure)
{
  constexpr double inverse_sqrt_two =
      0.70710678118654752440;

  for (std::size_t index_y = 0;
       index_y < number_of_cells_y_;
       ++index_y) {
    for (std::size_t index_x = 0;
         index_x < number_of_cells_x_;
         ++index_x) {
      const double x = cell_center_x(index_x);
      const double y = cell_center_y(index_y);

      const double distance_x = x - center_x;
      const double distance_y = y - center_y;

      const double distance =
          std::sqrt(
              distance_x * distance_x
              + distance_y * distance_y);

      const double pressure =
          distance <= blast_radius
          ? inner_pressure
          : outer_pressure;

      const PrimitiveState primitive{
          1.0,                   // density
          0.0, 0.0, 0.0,        // velocity
          pressure,
          inverse_sqrt_two,      // Bx
          inverse_sqrt_two,      // By
          0.0,                   // Bz
          0.0                    // psi
      };

      cell(index_x, index_y) =
          primitive_to_conservative(primitive);
    }
  }
}

void UniformGrid::write_csv(
    const std::string& filename) const
{
  std::ofstream output(filename);

  if (!output) {
    throw std::runtime_error(
        "Could not open CSV output file.");
  }

  output
      << "x,y,density,pressure,vx,vy,vz,"
      << "Bx,By,Bz,total_energy\n\n";

  for (std::size_t index_y = 0;
       index_y < number_of_cells_y_;
       ++index_y) {
    for (std::size_t index_x = 0;
         index_x < number_of_cells_x_;
         ++index_x) {
      const ConservativeState& conservative =
          cell(index_x, index_y);

      const PrimitiveState primitive =
          conservative_to_primitive(conservative);

      output
          << cell_center_x(index_x) << ','
          << cell_center_y(index_y) << ','
          << primitive.rho << ','
          << primitive.pressure << ','
          << primitive.vx << ','
          << primitive.vy << ','
          << primitive.vz << ','
          << primitive.bx << ','
          << primitive.by << ','
          << primitive.bz << ','
          << conservative[total_energy] << '\n';
    }
  }
}

std::size_t UniformGrid::number_of_cells_x() const
{
  return number_of_cells_x_;
}

std::size_t UniformGrid::number_of_cells_y() const
{
  return number_of_cells_y_;
}

double UniformGrid::cell_width_x() const
{
  return cell_width_x_;
}

double UniformGrid::cell_width_y() const
{
  return cell_width_y_;
}

} // namespace mhd
