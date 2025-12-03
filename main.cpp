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

    uint32_t const Nx = 2; // nombre d'éléments en x
    uint32_t const Ny = 2;  // nombre d'éléments en y

    double const hx = 5.0;
    double const hy = 3.0;

    msh::Mesh M = make_structured_quads_2D(Nx, Ny, /*x0*/0.0, /*y0*/0.0, hx, hy);

    std::vector<double> f(M.geo.n_nodes(),0.0);

    std::cout << "nodes=" << M.geo.n_nodes() << std::endl;
    std::cout << "cells=" << M.topo.n_cells() << std::endl;


    fem::SparseMatrixCSR K = fem::assemble_rigidity_matrix(M);

    std::cout << "K: nrows=" << K.nrows << ", ncols=" << K.ncols << ", nnz=" << K.col_indices.size() << "\n";  

    


    // Exporter en VTK (legacy ASCII)
    //const bool ok = msh::write_vtk(M, "mesh.vtk");
    //if (ok) std::cout << "Wrote mesh.vtk" << std::endl;
    //else std::cout << "Failed to write mesh.vtk" << std::endl;

    return 0; // Indique que le programme s'est terminé correctement
}
