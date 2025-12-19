#ifndef MESH_IO_H
#define MESH_IO_H

#include <string>
#include <vector>
#include <variant> // C++17 requis
#include "mesh_core.h"

namespace msh {

    // Type de donnée pour l'export
    enum class FieldType { Scalar, Vector };
    enum class FieldLocation { Node, Cell };

    // Structure générique pour décrire un champ à exporter
    struct VtkField {
        std::string name;
        FieldType type;
        FieldLocation location;
        const std::vector<double>& data; // Référence vers les données (pas de copie)
    };

    // Fonction principale unique : Écrit le maillage + une liste arbitraire de champs
    bool write_vtk(const Mesh &M, const std::string &filename, 
                   const std::vector<VtkField>& fields = {});

} // namespace msh

#endif // MESH_IO_H