#pragma once

#include "mhd_state.hxx"

#include <cstddef>
#include <string>
#include <vector>

namespace mhd {

class UniformGrid {
public:
  UniformGrid(
      std::size_t number_of_cells_x,
      std::size_t number_of_cells_y,
      double x_min,
      double x_max,
      double y_min,
      double y_max);

  void initialize_mhd_blast(
      double center_x,
      double center_y,
      double blast_radius,
      double inner_pressure,
      double outer_pressure);

  void write_csv(const std::string& filename) const;

  const ConservativeState& cell(
      std::size_t index_x,
      std::size_t index_y) const;

  ConservativeState& cell(
      std::size_t index_x,
      std::size_t index_y);

  double cell_center_x(std::size_t index_x) const;
  double cell_center_y(std::size_t index_y) const;

  std::size_t number_of_cells_x() const;
  std::size_t number_of_cells_y() const;

  double cell_width_x() const;
  double cell_width_y() const;

private:
  std::size_t flatten_index(
      std::size_t index_x,
      std::size_t index_y) const;

  std::size_t number_of_cells_x_;
  std::size_t number_of_cells_y_;

  double x_min_;
  double x_max_;
  double y_min_;
  double y_max_;

  double cell_width_x_;
  double cell_width_y_;

  std::vector<ConservativeState> cells_;
};

} // namespace mhd
