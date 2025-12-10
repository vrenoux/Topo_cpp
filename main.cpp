#include <iostream>  // Pour afficher dans la console
#include <vector>    // Pour utiliser les vecteurs
#include <cstdint>   // Pour les types entiers à largeur fixe
#include <petsc.h>  // Inclure PETSc

#include "./src/mesh_core.h" 
#include "./src/mesh_gen.h" 
#include "./src/mesh_io.h" 
#include "./src/linalg.h"
#include "./src/material.h"
#include "./src/element.h"
#include "./src/assembler.h"
#include "./src/boundary.h"
#include "./src/assembler_petsc.h"

#include "./src/conjugate_gradient.h"

int main(int argc, char **argv) {
    std::cout << "Hello, C++ est bien configuré !" << std::endl;

    uint32_t const fac_mesh = 500;

    uint32_t const Nx = 4 * fac_mesh; // nombre d'éléments en x
    uint32_t const Ny = 3 * fac_mesh;  // nombre d'éléments en y
  
    double const Hx = 300.0;
    double const Hy = 120.0;

    msh::Mesh M = make_structured_quads_2D(Nx, Ny, /*x0*/0.0, /*y0*/0.0, Hx, Hy);
    std::cout << "Computing boundary elements..." << std::endl;
    M.compute_boundary_elements();

    std::cout << "nodes=" << M.geo.n_nodes() << std::endl;
    std::cout << "cells=" << M.topo.n_cells() << std::endl;

    fem::BoundaryConditions bcs;

    // Dirichlet sur x = 0
    bcs.add_dirichlet_line_x(0.0, 1e-6, {0.0, 0.0, 0.0});

    // Neumann sur x = L
    bcs.add_neumann_line_x(Hx, 1e-6, {0.0, -1000.0, 0.0});

    std::cout << "===========================\n";

    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscInt major, minor, subminor, release;
    PetscGetVersionNumber(&major, &minor, &subminor, &release);
    PetscPrintf(PETSC_COMM_WORLD, "PETSc version: %d.%d.%d\n", major, minor, subminor);

    std::clock_t t0_cpu = std::clock();

    std::vector<double> x_petsc = assembler_petsc(M, 1, fem::LinearElasticityMaterial(200e3, 0.3, 1), bcs);

    std::clock_t t1_cpu = std::clock();

    double dt_cpu  = double(t1_cpu - t0_cpu) / CLOCKS_PER_SEC;

    std::cout << "[TIMER] assembler_petsc - cpu: " << dt_cpu << " s\n";

    //Convert PETSc solution to VectorDense and export
    fem::VectorDense U_petsc(x_petsc.size());
    for (size_t i = 0; i < x_petsc.size(); ++i) U_petsc[i] = x_petsc[i];
    
    if (!msh::write_vtk(M, "mesh_with_u_petsc.vtk", U_petsc.data)) {
        std::cerr << "Failed to write mesh_with_u_petsc.vtk\n";
    } else {
        std::cout << "Wrote mesh_with_u_petsc.vtk (contains POINT_DATA U from PETSc)\n";
    }

    PetscFinalize();

    // Exporter en VTK (legacy ASCII)
    //const bool ok = msh::write_vtk(M, "mesh.vtk");
    //if (ok) std::cout << "Wrote mesh.vtk" << std::endl;
    //else std::cout << "Failed to write mesh.vtk" << std::endl;

    return 0; // Indique que le programme s'est terminé correctement
}
