#ifndef MESH_CORE_H
#define MESH_CORE_H

#include <vector>
#include <cstdint>
#include <span>

namespace msh {

    struct Geometry {
        int dim = 2;
        std::vector<double> x, y, z;

        size_t n_nodes() const {
            return x.size();
        }

        uint32_t ndof() const {
            return n_nodes() * dim;
        }
    };

    enum class CellType : uint8_t {Tri3, Quad4Reg, Quad4, Tet4, Hex8 }; // Type de cellule sur 8 bits

    struct Topology {
        std::vector<uint32_t> c2n_offsets;  // size = n_cells + 1
        std::vector<uint32_t> c2n_indices; // size = total_connectivity
        std::vector<CellType> ctype;       //size = n_cells

        size_t n_cells() const {
            return c2n_offsets.size() - 1;
        }

        std::vector<uint32_t> get_nodes_cell(uint32_t cell_id) const {
            size_t start = c2n_offsets[cell_id];
            size_t end = c2n_offsets[cell_id + 1];
            return std::vector<uint32_t>(c2n_indices.begin() + start, c2n_indices.begin() + end);
        }
    };

    struct Mesh {
        Geometry geo;
        Topology topo;

        //Vues utilitaires (indices des cellules actives, regroupées par type, etc.)
    };

} // namespace msh


#endif
