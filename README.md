# t8-mhd-blast

A parallel two-dimensional ideal magnetohydrodynamics (MHD) blast-wave simulation using [t8code](https://github.com/DLR-AMR/t8code) for adaptive mesh refinement.

This project demonstrates how to combine a cell-centred finite-volume MHD solver with a distributed, nonconforming t8code mesh.

## Current capabilities

- Two-dimensional ideal-MHD equations
- Nine-component conservative state
- Circular high-pressure blast initial condition
- HLL approximate Riemann solver
- Fast magnetosonic wave-speed calculation
- Explicit Euler time integration
- CFL-controlled timestep
- Adaptive quadrilateral t8code mesh
- 2:1 mesh balancing
- Coarse-fine face-flux handling
- MPI partitioning
- MPI ghost-state exchange
- VTK output for ParaView
- Magnetic-divergence diagnostics
- Uniform-grid reference solver
- Numerical and mesh diagnostic tests

GLM divergence cleaning and higher-order time integration are planned as future improvements.

## Physical problem

The simulation begins with a circular high-pressure region inside a uniform, magnetized plasma. The pressure imbalance launches an expanding MHD blast wave.

The computational domain is:

```text
(x, y) in [0, 1] x [0, 1]
```

The initial density and velocity are:

```text
rho = 1
v = (vx, vy, vz) = (0, 0, 0)
```

The gas pressure is:

```text
p = 10.0    when r < 0.1
p = 0.1     when r >= 0.1
```

The distance from the centre of the blast is:

```text
r = sqrt((x - 0.5)^2 + (y - 0.5)^2)
```

The initial magnetic field is:

```text
B = (Bx, By, Bz) = (1/sqrt(2), 1/sqrt(2), 0)
```

The ratio of specific heats is:

```text
gamma = 5/3
```

## MHD state

Each cell stores nine conservative variables:

```text
U = [rho, rho*vx, rho*vy, rho*vz, E, Bx, By, Bz, psi]
```

Here:

- `rho` is the mass density.
- `rho*vx`, `rho*vy`, and `rho*vz` are the momentum-density components.
- `E` is the total energy density.
- `Bx`, `By`, and `Bz` are the magnetic-field components.
- `psi` is reserved for GLM divergence cleaning.

The total energy density is:

```text
E = p/(gamma - 1)
    + 0.5*rho*(vx^2 + vy^2 + vz^2)
    + 0.5*(Bx^2 + By^2 + Bz^2)
```

The three terms represent internal energy, kinetic energy, and magnetic energy.

## Numerical method

The solver uses a cell-centred finite-volume update:

```text
U_i^(n+1) = U_i^n
            - (dt/V_i) * sum over faces [A_f * F_hat_f]
```

Here:

- `dt` is the timestep.
- `V_i` is the area of cell `i` in two dimensions.
- `A_f` is the length of face `f`.
- `F_hat_f` is the numerical flux through face `f`.

The numerical interface flux is calculated using the HLL approximate Riemann solver:

```text
F_HLL = [S_R*F_L - S_L*F_R
         + S_L*S_R*(U_R - U_L)]
        / (S_R - S_L)
```

Here:

- `U_L` and `U_R` are the states on the two sides of an interface.
- `F_L` and `F_R` are their corresponding physical fluxes.
- `S_L` is the estimated fastest left-going wave speed.
- `S_R` is the estimated fastest right-going wave speed.

The wave-speed estimates include the fast magnetosonic speed.

Time integration currently uses the explicit Euler method with a CFL-controlled timestep.

Transmissive boundary conditions are applied at the physical boundaries of the domain.

## Adaptive mesh

The mesh begins at refinement level 3 and is refined near the blast region up to level 6.

The current initial adaptive mesh has the following properties:

| Quantity | Value |
|---|---:|
| Initial refinement level | 3 |
| Maximum refinement level | 6 |
| Uniform level-6 cell count | 4096 |
| Adaptive cell count | 940 |
| Cells inside the blast | 124 |
| Minimum cell area | 1/4096 |
| Maximum cell area | 1/64 |

The adaptive mesh is balanced so that neighboring cells differ by no more than one refinement level.

At a coarse-fine interface, one coarse face can touch two fine cells. The solver weights the corresponding flux contributions using the appropriate subface areas.

The current mesh is constructed before the simulation and remains fixed during time integration. Solution-dependent refinement and coarsening will be added later.

## MPI parallelism

t8code partitions the adaptive mesh between MPI processes.

Each process stores:

- Its locally owned cells
- Ghost cells belonging to neighboring MPI processes
- Conservative MHD states for local and ghost cells

Before fluxes are calculated, the local states are exchanged so that every process has current data for its ghost cells.

A representative run with four MPI processes gives:

```text
MPI processes:          4
Adaptive global cells:  940
Local cells per rank:   235
Ghost cells per rank:   40
```

## Dependencies

The project requires:

- A C++ compiler with C++17 support
- CMake
- Ninja
- MPI
- t8code 4.x
- p4est
- libsc
- zlib

The project was developed using:

| Dependency | Version |
|---|---:|
| t8code | 4.0.0 |
| libsc | 2.8.7 |
| p4est | 4.0.0 |
| OpenMPI | 3.1 |
| CMake | 3.28 |
| GCC/G++ | 13 |

## Building

The following commands assume that t8code is installed in:

```text
$HOME/t8code-install
```

Configure the project:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_PREFIX_PATH="$HOME/t8code-install"
```

Compile the project:

```bash
cmake --build build
```

Both C and C++ must be enabled in `CMakeLists.txt` because the installed t8code dependencies require the MPI C component:

```cmake
project(t8_mhd_blast LANGUAGES C CXX)
```

## Running

### Uniform-grid MHD solver

Run the uniform-grid reference solver with:

```bash
./build/t8_mhd_blast
```

### Uniform t8code mesh test

Run the uniform t8code mesh example with:

```bash
mpirun -np 1 ./build/t8_mhd_mesh
```

### Adaptive parallel MHD solver

Run the adaptive solver using four MPI processes:

```bash
mpirun -np 4 ./build/t8_mhd_adaptive
```

The value following `-np` selects the number of MPI processes.

A serial adaptive run can be started with:

```bash
mpirun -np 1 ./build/t8_mhd_adaptive
```

## Simulation parameters

The current adaptive simulation uses:

| Parameter | Value |
|---|---:|
| Final time | 0.02 |
| CFL number | 0.25 |
| VTK output interval | 10 steps |
| Time integrator | Explicit Euler |
| Numerical flux | HLL |

A representative run reaches the final time in 88 steps.

## ParaView output

The adaptive simulation writes numbered parallel VTK files:

```text
t8_mhd_0000.pvtu
t8_mhd_0001.pvtu
t8_mhd_0002.pvtu
...
```

The initial state is stored in:

```text
t8_mhd_0000.pvtu
```

Open it in ParaView with:

```bash
paraview t8_mhd_0000.pvtu
```

To inspect the time evolution, open the numbered files as a file series in ParaView.

The VTK output contains the following cell fields:

- `density`
- `pressure`
- `total_energy`
- `psi`
- `divergence_B`
- `velocity`
- `magnetic_field`
- tree ID
- MPI rank
- refinement level
- element ID

Useful ParaView visualizations include:

- Pressure contours
- Density contours
- Magnetic-field glyphs
- Velocity glyphs
- Magnetic-divergence maps
- Refinement-level maps
- MPI partition maps

## Verification

The project contains diagnostic checks for:

- Primitive-to-conservative state conversion
- Conservative-to-primitive state conversion
- Positive density and pressure
- Fast magnetosonic wave speeds
- HLL interface fluxes
- Adaptive face neighbors
- Physical boundary faces
- Same-level neighbors
- Coarse-fine neighbors
- MPI ghost neighbors
- Invalid neighbor indices
- 2:1 mesh balance
- MPI ghost-state exchange
- Cell areas
- Face lengths
- Face-normal orientation
- Non-finite numerical fluxes
- Magnetic divergence

A representative adaptive face-neighbor diagnostic gives:

```text
Physical boundary faces:       32
Same-level relations:          3344
Coarse-to-fine relations:      256
Fine-to-coarse relations:      256
MPI ghost relations:           160
Invalid neighbour indices:     0
2:1 balance violations:        0
Invalid neighbour counts:      0
```

The corresponding adaptive HLL flux diagnostic gives:

```text
Total directed fluxes:       3888
Boundary fluxes:             32
MPI ghost fluxes:            160
Coarse-fine fluxes:          512
Invalid fluxes:              0
```

## Magnetic-divergence diagnostic

The initial uniform magnetic field satisfies the discrete divergence constraint exactly:

```text
L1 div(B):             0
L2 div(B):             0
Maximum |div(B)|:      0
Normalized L1:         0
Normalized maximum:    0
```

Without active divergence cleaning, numerical divergence grows during the simulation.

At the final time `t = 0.02`, a representative result is:

```text
L1 div(B):             2.70710759e-02
L2 div(B):             1.11168458e-01
Maximum |div(B)|:      1.03922969e+00
Normalized L1:         4.00845678e-04
Normalized maximum:    1.37032598e-02
```

These values provide a baseline for evaluating the planned GLM divergence-cleaning implementation.

## Project structure

```text
t8-mhd-blast/
|-- CMakeLists.txt
|-- README.md
`-- src/
    |-- main.cxx
    |-- mhd_state.hxx
    |-- mhd_state.cxx
    |-- mhd_flux.hxx
    |-- mhd_flux.cxx
    |-- hll_solver.hxx
    |-- hll_solver.cxx
    |-- finite_volume_solver.hxx
    |-- finite_volume_solver.cxx
    |-- uniform_grid.hxx
    |-- uniform_grid.cxx
    |-- vtk_writer.hxx
    |-- vtk_writer.cxx
    |-- t8_uniform_mesh.cxx
    |-- t8_adaptive_mesh.cxx
    |-- t8_adaptive_data.hxx
    |-- t8_adaptive_data.cxx
    |-- t8_neighbor_diagnostic.hxx
    |-- t8_neighbor_diagnostic.cxx
    |-- t8_geometry_diagnostic.hxx
    |-- t8_geometry_diagnostic.cxx
    |-- t8_flux_diagnostic.hxx
    |-- t8_flux_diagnostic.cxx
    |-- t8_euler_step.hxx
    |-- t8_euler_step.cxx
    |-- t8_time_loop.hxx
    |-- t8_time_loop.cxx
    |-- t8_magnetic_divergence.hxx
    `-- t8_magnetic_divergence.cxx
```

## Development roadmap

Planned improvements include:

1. Hyperbolic/parabolic GLM divergence cleaning
2. Comparison of divergence errors before and after cleaning
3. SSP-RK2 time integration
4. Solution-dependent adaptive refinement
5. Solution-dependent adaptive coarsening
6. Conservative state transfer during mesh adaptation
7. Gradient reconstruction
8. Slope limiting
9. Higher-order spatial accuracy
10. Automated regression tests
11. MPI scaling and performance measurements

## Acknowledgements

This project uses [t8code](https://github.com/DLR-AMR/t8code), an adaptive mesh refinement library developed by the DLR Institute of Software Technology and its contributors.
