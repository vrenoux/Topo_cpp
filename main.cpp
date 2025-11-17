#include <iostream>  // Pour afficher dans la console
#include <vector>    // Pour utiliser les vecteurs
#include <cstdint>   // Pour les types entiers à largeur fixe

#include "mesh_core.h"  
#include "mesh_gen.h"
#include "mesh_io.h"


int main() {
    std::cout << "Hello, C++ est bien configuré !" << std::endl;

    msh::Mesh M = make_structured_quads_2D(/*Nx*/1000, /*Ny*/1000, /*x0*/0.0, /*y0*/0.0, /*hx*/5.0, /*hy*/3.0);

    std::cout << "nodes=" << M.geo.x.size() << std::endl;

    // Exporter en VTK (legacy ASCII)
    const bool ok = msh::write_vtk(M, "mesh.vtk");
    if (ok) std::cout << "Wrote mesh.vtk" << std::endl;
    else std::cout << "Failed to write mesh.vtk" << std::endl;

    return 0; // Indique que le programme s'est terminé correctement
}
