#ifndef MESH_IO_H
#define MESH_IO_H

#include <string>
#include "mesh_core.h"

namespace msh {

// Écrit le maillage M dans un fichier VTK (legacy, ASCII)
// Retourne true si l'écriture a réussi.
bool write_vtk(const Mesh &M, const std::string &filename);

} // namespace msh

#endif // MESH_IO_H
