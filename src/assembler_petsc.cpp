#include <petscmat.h>
#include <vector>

#include "assembler_petsc.h"
#include "mesh_core.h"
#include "material.h"
#include "element.h"


template <typename ElementType>
void process_element_batch(Mat A, const msh::Mesh& mesh, uint32_t cell_id, 
                           const fem::LinearElasticityMaterial& mat) {
    
    // 1. Récupération des indices des nœuds du maillage
    auto node_indices_vec = mesh.topo.get_nodes_cell(cell_id);
    
    // 2. Conversion en std::array pour la classe Element
    std::array<uint32_t, ElementType::n_nodes> node_ids;
    std::copy(node_indices_vec.begin(), node_indices_vec.end(), node_ids.begin());

    // 3. Extraction des Coordonnées
    std::array<std::array<double, ElementType::dim>, ElementType::n_nodes> coords;
    for (int i = 0; i < ElementType::n_nodes; ++i) {
        uint32_t nid = node_ids[i];
        coords[i][0] = mesh.geo.x[nid];
        coords[i][1] = mesh.geo.y[nid];
        if constexpr (ElementType::dim == 3)
            coords[i][2] = mesh.geo.z[nid];
    }

    // 4. Instanciation de l'élément et Calcul Ke
    ElementType elem(node_ids, mat);
    std::array<double, ElementType::ndof * ElementType::ndof> Ke;
    elem.compute_stiffness(coords, Ke);

    // 5. Préparation des indices pour PETSc
    std::array<PetscInt, ElementType::n_nodes> petsc_rows;
    for(size_t i=0; i<ElementType::n_nodes; ++i) {
        petsc_rows[i] = static_cast<PetscInt>(node_ids[i]);
    }

    // 6. Injection dans PETSc
    MatSetValuesBlocked(A, 
                        ElementType::n_nodes, petsc_rows.data(),
                        ElementType::n_nodes, petsc_rows.data(),
                        Ke.data(),
                        ADD_VALUES);
}

void assemble_rigidity(Mat A,
                       const msh::Mesh& mesh,
                       const fem::LinearElasticityMaterial& mat){
    
    std::cout << "Assembling rigidity matrix..." << std::endl;

    for (size_t e = 0; e < mesh.topo.n_cells(); ++e) {
        
        if (e % 1000 == 0) {
            std::cout << "Assembly: " << e << "/" << mesh.topo.n_cells() << "\r" << std::flush;
        }

        msh::CellType type = mesh.topo.ctype[e];

        switch (type) {
            case msh::CellType::Quad4Reg:
                process_element_batch<fem::Quad4RegularElement>(A, mesh, e, mat);
                break;

            default:
                // Type non supporté ou ignoré
                break;
        }
    }
}

template <typename ElementType>
double process_energy_batch(const PetscScalar* u_global_ptr, 
                            const msh::Mesh& mesh, 
                            uint32_t cell_id, 
                            const fem::LinearElasticityMaterial& mat) {
    
    auto node_indices_vec = mesh.topo.get_nodes_cell(cell_id);
    
    std::array<uint32_t, ElementType::n_nodes> node_ids;
    std::copy(node_indices_vec.begin(), node_indices_vec.end(), node_ids.begin());

    std::array<std::array<double, ElementType::dim>, ElementType::n_nodes> coords;
    for (int i = 0; i < ElementType::n_nodes; ++i) {
        uint32_t nid = node_ids[i];
        coords[i][0] = mesh.geo.x[nid];
        coords[i][1] = mesh.geo.y[nid];
        if constexpr (ElementType::dim == 3) {
            if (mesh.geo.z.size() > nid) coords[i][2] = mesh.geo.z[nid];
            else coords[i][2] = 0.0;
        }
    }

    std::array<double, ElementType::ndof> u_elem;
    
    for (int i = 0; i < ElementType::n_nodes; ++i) {
        uint32_t global_node_idx = node_ids[i];
        
        for (int d = 0; d < ElementType::dim; ++d) {
            PetscInt dof_idx = global_node_idx * ElementType::dim + d;
            
            u_elem[i * ElementType::dim + d] = static_cast<double>(u_global_ptr[dof_idx]);
        }
    }

    ElementType elem(node_ids, mat);
    
    return elem.compute_energy(coords, u_elem);
}

std::vector<double> compute_energy(const Vec& u,
                                   const msh::Mesh& mesh,
                                   const fem::LinearElasticityMaterial& mat) {
    
    const PetscScalar* u_ptr;
    VecGetArrayRead(u, &u_ptr); // Accès en lecture seule aux données de u

    std::cout << "Computing strain energy map..." << std::endl;

    std::vector<double> energy(mesh.topo.n_cells(), 0.0);

    for (size_t e = 0; e < mesh.topo.n_cells(); ++e) {

        if (e % 5000 == 0) {
            std::cout << "Assembly: " << e << "/" << mesh.topo.n_cells() << "\r" << std::flush;
        }

        msh::CellType type = mesh.topo.ctype[e];

        switch (type) {
            case msh::CellType::Quad4Reg:
                energy[e] = process_energy_batch<fem::Quad4RegularElement>(
                    u_ptr, mesh, e, mat);
                break;

            default:
                // Type non supporté ou ignoré
                break;
        }
    }         

    VecRestoreArrayRead(u, &u_ptr); // Libération de l'accès aux données de u
    
    return energy;
}
