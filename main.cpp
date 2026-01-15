#include <iostream>
#include <vector>
#include <cstdint>
#include <numeric>
#include <petsc.h>

#include "./src/mesh_core.h"
#include "./src/mesh_gen.h"
#include "./src/mesh_io.h"
#include "./src/material.h"
#include "./src/boundary.h"
#include "./src/assembler_petsc.h"
#include "./src/optimizer.h"

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);

    const uint32_t fac_mesh = 900; // ajustable
    const uint32_t Nx = 4 * fac_mesh;
    const uint32_t Ny = 3 * fac_mesh;
    const double Hx = 300.0;
    const double Hy = 120.0;

    msh::Mesh M = make_structured_quads_2D(Nx, Ny, /*x0*/0.0, /*y0*/0.0, Hx, Hy);
    std::cout << "Computing boundary elements..." << std::endl;
    M.compute_boundary_elements();
    M.compute_volume_cells();

    std::cout << "Number of nodes: " << M.geo.n_nodes() << std::endl;
    std::cout << "Number of elements: " << M.topo.n_cells() << std::endl;

    const size_t n_cells = M.topo.n_cells();
    const double vol_frac = 0.5;
    M.density.assign(n_cells, vol_frac);

    fem::BoundaryConditions bcs;
    bcs.add_dirichlet_line_x(0.0, 1e-6, {true, true, true});
    bcs.add_neumann_line_x(Hx, 1e-6, {0.0, -1000.0, 0.0});

    fem::LinearElasticityMaterial mat(200e3, 0.3, 1, 1.0);

    Optimizer opt;
    opt.setMaxIter(50);
    opt.setVolumeFractionConstraint(vol_frac);
    opt.setNDesignVariables(static_cast<int>(n_cells));
    opt.setPenalization(Penalization{3.0});

    std::vector<double> xmin(n_cells, 1e-2);
    std::vector<double> xmax(n_cells, 1.0);
    opt.setDesignVariableBounds(xmin, xmax);
    M.density.assign(M.topo.n_cells(), 0.5);
    opt.Optimize(mat, M, bcs);

    PetscFinalize();
    return 0; 
}
