#ifndef MESH_IO_H
#define MESH_IO_H

#include <string>
#include "mesh_core.h"
#include <vector>

namespace msh {

// Écrit le maillage M dans un fichier VTK (legacy, ASCII)
// Retourne true si l'écriture a réussi.
bool write_vtk(const Mesh &M, const std::string &filename);

// Écrit le maillage M dans un fichier VTK (legacy, ASCII) et ajoute un champ
// nodal vecteur `U` (taille = geo.ndof(), ordonnancement par noeud: [u0_x,u0_y(,u0_z), u1_x,...]).
// Retourne true si l'écriture a réussi.
bool write_vtk(const Mesh &M, const std::string &filename, const std::vector<double> &U);

} // namespace msh

#endif // MESH_IO_H
