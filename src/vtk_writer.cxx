#include "vtk_writer.hxx"

#include "mhd_state.hxx"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace mhd {

void write_vtu(
    const UniformGrid& grid,
    const std::string& filename,
    const double simulation_time)
{
  std::ofstream output(filename);

  if (!output) {
    throw std::runtime_error(
        "Could not open VTU output file: " + filename);
  }

  output << std::scientific
         << std::setprecision(16);

  const std::size_t cells_x =
      grid.number_of_cells_x();

  const std::size_t cells_y =
      grid.number_of_cells_y();

  const std::size_t number_of_cells =
      cells_x * cells_y;

  const std::size_t points_x = cells_x + 1;
  const std::size_t points_y = cells_y + 1;

  const std::size_t number_of_points =
      points_x * points_y;

  const double x_min =
      grid.cell_center_x(0)
      - 0.5 * grid.cell_width_x();

  const double y_min =
      grid.cell_center_y(0)
      - 0.5 * grid.cell_width_y();

  output
      << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"UnstructuredGrid\" "
      << "version=\"0.1\" byte_order=\"LittleEndian\">\n"
      << "  <UnstructuredGrid>\n"
      << "    <FieldData>\n"
      << "      <DataArray type=\"Float64\" "
      << "Name=\"TimeValue\" NumberOfTuples=\"1\" "
      << "format=\"ascii\">\n"
      << "        " << simulation_time << '\n'
      << "      </DataArray>\n"
      << "    </FieldData>\n"
      << "    <Piece NumberOfPoints=\""
      << number_of_points
      << "\" NumberOfCells=\""
      << number_of_cells
      << "\">\n";

  /*
   * Write the grid vertices.
   */
  output
      << "      <Points>\n"
      << "        <DataArray type=\"Float64\" "
      << "NumberOfComponents=\"3\" format=\"ascii\">\n";

  for (std::size_t point_y = 0;
       point_y < points_y;
       ++point_y) {
    for (std::size_t point_x = 0;
         point_x < points_x;
         ++point_x) {
      const double x =
          x_min + point_x * grid.cell_width_x();

      const double y =
          y_min + point_y * grid.cell_width_y();

      output
          << "          "
          << x << ' ' << y << " 0.0\n";
    }
  }

  output
      << "        </DataArray>\n"
      << "      </Points>\n";

  /*
   * Write quadrilateral cell connectivity.
   *
   * Vertex order:
   *
   *   3 ----- 2
   *   |       |
   *   0 ----- 1
   */
  output
      << "      <Cells>\n"
      << "        <DataArray type=\"Int64\" "
      << "Name=\"connectivity\" format=\"ascii\">\n";

  for (std::size_t index_y = 0;
       index_y < cells_y;
       ++index_y) {
    for (std::size_t index_x = 0;
         index_x < cells_x;
         ++index_x) {
      const std::size_t point_0 =
          index_y * points_x + index_x;

      const std::size_t point_1 =
          point_0 + 1;

      const std::size_t point_3 =
          (index_y + 1) * points_x + index_x;

      const std::size_t point_2 =
          point_3 + 1;

      output
          << "          "
          << point_0 << ' '
          << point_1 << ' '
          << point_2 << ' '
          << point_3 << '\n';
    }
  }

  output
      << "        </DataArray>\n"
      << "        <DataArray type=\"Int64\" "
      << "Name=\"offsets\" format=\"ascii\">\n";

  for (std::size_t cell_index = 0;
       cell_index < number_of_cells;
       ++cell_index) {
    output << "          "
           << 4 * (cell_index + 1) << '\n';
  }

  output
      << "        </DataArray>\n"
      << "        <DataArray type=\"UInt8\" "
      << "Name=\"types\" format=\"ascii\">\n";

  /*
   * VTK cell type 9 is VTK_QUAD.
   */
  for (std::size_t cell_index = 0;
       cell_index < number_of_cells;
       ++cell_index) {
    output << "          9\n";
  }

  output
      << "        </DataArray>\n"
      << "      </Cells>\n";

  output
      << "      <CellData Scalars=\"pressure\">\n";

  const auto write_scalar =
      [&](const std::string& name,
          const auto& value_function) {
        output
            << "        <DataArray type=\"Float64\" "
            << "Name=\"" << name
            << "\" format=\"ascii\">\n";

        for (std::size_t index_y = 0;
             index_y < cells_y;
             ++index_y) {
          for (std::size_t index_x = 0;
               index_x < cells_x;
               ++index_x) {
            const ConservativeState& state =
                grid.cell(index_x, index_y);

            const PrimitiveState primitive =
                conservative_to_primitive(state);

            output
                << "          "
                << value_function(state, primitive)
                << '\n';
          }
        }

        output << "        </DataArray>\n";
      };

  write_scalar(
      "density",
      [](const ConservativeState&,
         const PrimitiveState& primitive) {
        return primitive.rho;
      });

  write_scalar(
      "pressure",
      [](const ConservativeState&,
         const PrimitiveState& primitive) {
        return primitive.pressure;
      });

  write_scalar(
      "total_energy",
      [](const ConservativeState& state,
         const PrimitiveState&) {
        return state[total_energy];
      });

  write_scalar(
      "psi",
      [](const ConservativeState& state,
         const PrimitiveState&) {
        return state[glm_psi];
      });

  output
      << "        <DataArray type=\"Float64\" "
      << "Name=\"velocity\" NumberOfComponents=\"3\" "
      << "format=\"ascii\">\n";

  for (std::size_t index_y = 0;
       index_y < cells_y;
       ++index_y) {
    for (std::size_t index_x = 0;
         index_x < cells_x;
         ++index_x) {
      const PrimitiveState primitive =
          conservative_to_primitive(
              grid.cell(index_x, index_y));

      output
          << "          "
          << primitive.vx << ' '
          << primitive.vy << ' '
          << primitive.vz << '\n';
    }
  }

  output
      << "        </DataArray>\n"
      << "        <DataArray type=\"Float64\" "
      << "Name=\"magnetic_field\" "
      << "NumberOfComponents=\"3\" format=\"ascii\">\n";

  for (std::size_t index_y = 0;
       index_y < cells_y;
       ++index_y) {
    for (std::size_t index_x = 0;
         index_x < cells_x;
         ++index_x) {
      const PrimitiveState primitive =
          conservative_to_primitive(
              grid.cell(index_x, index_y));

      output
          << "          "
          << primitive.bx << ' '
          << primitive.by << ' '
          << primitive.bz << '\n';
    }
  }

  output
      << "        </DataArray>\n"
      << "      </CellData>\n"
      << "    </Piece>\n"
      << "  </UnstructuredGrid>\n"
      << "</VTKFile>\n";
}

} // namespace mhd
