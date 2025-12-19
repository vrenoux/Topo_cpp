#ifndef ELEMENT_H
#define ELEMENT_H

#include "material.h"

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <cstdint> 

namespace fem {

// ============================================================================
// Quad4 Regular Element (2D)
// ============================================================================

struct Quad4RegularElement {
    static constexpr uint32_t dim = 2; // 2D
    static constexpr int n_nodes = 4; // 4 nodes
    static constexpr int ndof = 8; // 4 nodes * 2 dof
    std::array<uint32_t, 4>node_ids; // indices des nœuds dans le Mesh
    const LinearElasticityMaterial& mat;
    
    Quad4RegularElement(const std::array<uint32_t, n_nodes>& nodes, const LinearElasticityMaterial& m)
        : node_ids(nodes), mat(m) {}

    void compute_stiffness(std::array<std::array<double, dim>, n_nodes>& coords,
                                  std::array<double, ndof * ndof>& Ke) const ;


    double compute_energy(const std::array<std::array<double, dim>, n_nodes>& coords,
                                    const std::array<double, ndof>& u_elem) const ;


    private:
        void accumulate_stiffness(std::array<double, ndof * ndof>& Ke,
                                const std::array<double, 24>& B,
                                const std::array<double, 36>& D,
                                double weight) const ;

        void compute_B_matrix(std::array<double, 24>& B, double elx, double ely, double xgp, double ygp) const;

};

} // namespace fem

#endif // ELEMENT_H