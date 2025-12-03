#include <iostream>  // Pour afficher dans la console
#include <vector>    // Pour utiliser les vecteurs
#include <cstdint>   // Pour les types entiers à largeur fixe

#include "./src/mesh_core.h" 
#include "./src/mesh_gen.h" 
#include "./src/mesh_io.h" 
#include "./src/linalg.h"
#include "./src/material.h"
#include "./src/element.h"
#include "./src/assembler.h"
#include "./src/boundary.h"



int main() {
    std::cout << "Hello, C++ est bien configuré !" << std::endl;

    uint32_t const Nx = 5; // nombre d'éléments en x
    uint32_t const Ny = 5;  // nombre d'éléments en y

    double const Hx = 5.0;
    double const Hy = 3.0;

    msh::Mesh M = make_structured_quads_2D(Nx, Ny, /*x0*/0.0, /*y0*/0.0, Hx, Hy);

    std::vector<double> f(M.geo.n_nodes(),0.0);

    std::cout << "nodes=" << M.geo.n_nodes() << std::endl;
    std::cout << "cells=" << M.topo.n_cells() << std::endl;


    fem::SparseMatrixCSR K = fem::assemble_rigidity_matrix(M);

    std::cout << "K: nrows=" << K.nrows << ", ncols=" << K.ncols << ", nnz=" << K.col_indices.size() << "\n";  


    fem::BoundaryConditions bcs;

    // Dirichlet sur x = 0
    bcs.add_dirichlet_line_x(0.0, 1e-6, {0.0, 0.0, 0.0});

    // Neumann sur x = L
    bcs.add_neumann_line_x(Hx, 1e-6, {0.0, 100.0, 0.0});

    bcs.apply_dirichlet(K, bcs, M);
    fem::VectorDense F = bcs.apply_neumann(M);
    std::cout << "F size: " << F.size() << std::endl;
    fem::print_vector_dense(F);




    // Exporter en VTK (legacy ASCII)
    //const bool ok = msh::write_vtk(M, "mesh.vtk");
    //if (ok) std::cout << "Wrote mesh.vtk" << std::endl;
    //else std::cout << "Failed to write mesh.vtk" << std::endl;

    return 0; // Indique que le programme s'est terminé correctement
}
