cmake_minimum_required(VERSION 3.16)

project(t8_mhd_blast LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(t8_mhd_blast
    src/main.cxx
    src/mhd_flux.cxx
    src/mhd_state.cxx
    src/hll_solver.cxx
    src/uniform_grid.cxx
    src/finite_volume_solver.cxx
    src/vtk_writer.cxx
)

target_include_directories(t8_mhd_blast PRIVATE src)

target_compile_options(t8_mhd_blast PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
find_package(MPI REQUIRED)
find_package(T8CODE CONFIG REQUIRED)

add_executable(t8_mhd_mesh
    src/t8_uniform_mesh.cxx
    src/mhd_state.cxx
)

target_link_libraries(t8_mhd_mesh PRIVATE
    MPI::MPI_CXX
    T8CODE::T8
)

target_compile_options(t8_mhd_mesh PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
# ------------------------------------------------------------
# Program 3: Adaptive t8code blast mesh
# ------------------------------------------------------------

add_executable(t8_mhd_adaptive
    src/t8_adaptive_mesh.cxx
    src/t8_adaptive_data.cxx
    src/t8_neighbor_diagnostic.cxx
    src/t8_geometry_diagnostic.cxx
    src/t8_flux_diagnostic.cxx
    src/t8_euler_step.cxx
    src/t8_time_loop.cxx
    src/t8_magnetic_divergence.cxx
    src/mhd_state.cxx
    src/mhd_flux.cxx
    src/hll_solver.cxx
)

target_link_libraries(t8_mhd_adaptive PRIVATE
    MPI::MPI_CXX
    T8CODE::T8
)

target_compile_options(t8_mhd_adaptive PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
