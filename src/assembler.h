#ifndef ASSEMBLER_H
#define ASSEMBLER_H

#include "linalg.h"
#include "element.h"
#include "mesh_core.h"
#include "material.h"

#include <cassert> 
#include <array>
#include <vector>


namespace fem {

// ============================================================================
// Assemblage d'un élément dans la matrice globale K (format COO)
// ============================================================================

template<typename ElementType>
inline void assemble_element(const ElementType& elem,
                              const msh::Mesh& mesh,
                              SparseMatrixCOO& K) {
    std::array<double, ElementType::ndof * ElementType::ndof> Ke;

    std::array<std::array<double, ElementType::dim>, ElementType::n_nodes> coords;
    for (int i = 0; i < ElementType::n_nodes; ++i) {
        coords[i][0] = mesh.geo.x[elem.node_ids[i]];
        coords[i][1] = mesh.geo.y[elem.node_ids[i]];
        if constexpr (ElementType::dim == 3)
            coords[i][2] = mesh.geo.z[elem.node_ids[i]];
    }

    elem.compute_stiffness(coords, Ke);

    constexpr uint32_t dim = ElementType::dim;

    for (int i = 0; i < ElementType::ndof; ++i) {
        for (int j = 0; j < ElementType::ndof; ++j) {
            uint32_t global_i = elem.node_ids[i / dim] * dim + (i % dim);
            uint32_t global_j = elem.node_ids[j / dim] * dim + (j % dim);
            K.add(global_i, global_j, Ke[i * ElementType::ndof + j]);
        }
    }
}

// ============================================================================
// Assembler de la matrice de rigidité globale
// ============================================================================

SparseMatrixCSR assemble_rigidity_matrix(const msh::Mesh& mesh){

    // Define material
    constexpr double young_modulus = 200e3; // Exemple
    constexpr double poisson_ratio = 0.3;   // Exemple
    constexpr uint32_t material_type = 1;    // Plane stress
    constexpr double thickness = 1.0;        // Épaisseur pour les éléments 2D

    LinearElasticityMaterial material(young_modulus, poisson_ratio, material_type);

    // Matrice de rigidité globale en format COO
    SparseMatrixCOO K_full_coo(mesh.geo.ndof());

    for (size_t e = 0; e < mesh.topo.n_cells(); ++e) {
        switch(mesh.topo.ctype[e]) {
            case msh::CellType::Quad4Reg: {
                auto nodes = mesh.topo.get_nodes_cell(e);
                assert(nodes.size() == Quad4RegularElement::n_nodes);

                std::array<uint32_t, Quad4RegularElement::n_nodes> nodesArr;
                std::copy(nodes.begin(), nodes.end(), nodesArr.begin());
                
                Quad4RegularElement elem(nodesArr, thickness, material);
                assemble_element(elem, mesh, K_full_coo);
                break;
            }
            // Ajouter d'autres types d'éléments ici si nécessaire
        }
    }

    SparseMatrixCSR K_full_csr;
    K_full_csr.build_from_coo(K_full_coo, true);

    return K_full_csr;
}




} // namespace fem

#endif
