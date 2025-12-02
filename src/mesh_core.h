#ifndef MESH_CORE_H
#define MESH_CORE_H

#include <vector>
#include <cstdint>

namespace msh {

    struct Geometry {
        int dim = 2;
        std::vector<double> x, y, z;

        size_t n_nodes() const {
            return x.size();
        }

    };

    enum class CellType : uint8_t { Tri3, Quad4, Tet4, Hex8 }; // Type de cellule sur 8 bits

    struct Topology {
        std::vector<uint32_t> c2n_offsets;  // size = n_cells + 1
        std::vector<uint32_t> c2n_indices; // size = total_connectivity
        std::vector<CellType> ctype;       //size = n_cells

        size_t n_cells() const {
            return c2n_indices.size();
        }

    };

    struct Mesh {
        Geometry geo;
        Topology topo;

        //Vues utilitaires (indices des cellules actives, regroupées par type, etc.)
    };

} // namespace msh


#endif
