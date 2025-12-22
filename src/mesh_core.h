#ifndef MESH_CORE_H
#define MESH_CORE_H

#include <vector>
#include <cstdint>
#include <span>
#include <cmath>
#include <array>
#include <algorithm>
#include <numeric>


namespace msh {

    enum class CellType : uint8_t {Tri3, Quad4Reg, Quad4, Tet4, Hex8 }; // Type de cellule sur 8 bits

    struct Geometry {
        int dim;
        std::vector<double> x, y, z;

        size_t n_nodes() const { return x.size(); }

        uint32_t ndof() const { return n_nodes() * dim; }
    };

    
    struct Topology {
        std::vector<uint32_t> c2n_offsets;  // size = n_cells + 1
        std::vector<uint32_t> c2n_indices; // size = total_connectivity
        std::vector<CellType> ctype;       //size = n_cells

        size_t n_cells() const { return c2n_offsets.size() - 1; }

        std::vector<uint32_t> get_nodes_cell(uint32_t cell_id) const {
            size_t start = c2n_offsets[cell_id];
            size_t end = c2n_offsets[cell_id + 1];
            std::vector<uint32_t> nodes;
            for(size_t k=start; k<end; ++k) nodes.push_back(c2n_indices[k]);
            return nodes;
        }
    };

    struct BoundaryElement {
        std::vector<uint32_t> node_ids;
        double measure; // longueur en 2D, surface en 3D
        double center_x, center_y, center_z;
    };

    struct Mesh {
        Geometry geo;
        Topology topo;

        std::vector<BoundaryElement> boundary;
        std::vector<double> volume_cells;
        double volume_total;
        std::vector<double> density;

        void compute_boundary_elements();

        void compute_volume_cells();

        private:
            std::vector<std::vector<uint32_t>> get_local_faces(CellType type, const std::vector<uint32_t>& cell_nodes) const;
            void  compute_measure_and_center(BoundaryElement& be);
    };

} // namespace msh


#endif
