#ifndef ASSEMBLER_PETSC_H
#define ASSEMBLER_PETSC_H

#include <petsc.h>
#include <vector>
#include "mesh_core.h"
#include "material.h"
#include "element.h"
#include "boundary.h"

inline std::vector<double> assembler_petsc(const msh::Mesh& mesh,
                            const double& thickness,
                            const fem::LinearElasticityMaterial& material,
                            const fem::BoundaryConditions& bcs){
    
    const PetscInt dim = static_cast<PetscInt>(mesh.geo.dim);
    const PetscInt N = static_cast<PetscInt>(mesh.geo.n_nodes() * dim); // 2 dof par nœud en 2D

    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N);
    MatSetFromOptions(A);   // Autorise -mat_type aij, baij, etc. en ligne de commande
    MatSetUp(A);            // Alloue la structure interne (réallocations possibles ensuite)

    Vec b, x;
    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, N);
    VecSetFromOptions(b);
    VecDuplicate(b, &x);

    //MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE); // En dev, tu peux activer cette option pour t’assurer que tu ne crées pas de nouveaux non-nuls

    // ASSEMBLAGE
    for (size_t e = 0; e < mesh.topo.n_cells(); ++e) {
        if (mesh.topo.ctype[e] != msh::CellType::Quad4Reg) {
            // si tu as d’autres éléments, gère-les ici
            continue;
        }

        auto nodes = mesh.topo.get_nodes_cell(e);
        if (nodes.size() != fem::Quad4RegularElement::n_nodes) {
            throw std::runtime_error("Quad4Reg: nombre de nœuds inattendu.");
        }

        std::array<uint32_t, fem::Quad4RegularElement::n_nodes> nodesArr;
        std::copy(nodes.begin(), nodes.end(), nodesArr.begin());

        fem::Quad4RegularElement elem(nodesArr, thickness, material);

        // coords des 4 nœuds (x,y)
        std::array<std::array<double, fem::Quad4RegularElement::dim>, fem::Quad4RegularElement::n_nodes> coords;
        for (int i = 0; i < fem::Quad4RegularElement::n_nodes; ++i) {
            const auto nid = elem.node_ids[i];
            coords[i][0] = mesh.geo.x[nid];
            coords[i][1] = mesh.geo.y[nid];
        }

        // matrice locale Ke (8×8)
        std::array<double, fem::Quad4RegularElement::ndof * fem::Quad4RegularElement::ndof> Ke;
        elem.compute_stiffness(coords, Ke);

        // indices DOF globaux (taille 8)
        std::array<PetscInt, fem::Quad4RegularElement::ndof> gidx;
        {
            constexpr int dim = fem::Quad4RegularElement::dim; // dim
            for (int i = 0; i < fem::Quad4RegularElement::ndof; ++i) {
                const uint32_t node = elem.node_ids[i / dim];
                const uint32_t comp = i % dim; // 0: ux, 1: uy
                gidx[i] = static_cast<PetscInt>(node * dim + comp);
            }
        }

        // insertion bloc (ADD_VALUES: contributions s’additionnent)
        MatSetValues(A,
                     fem::Quad4RegularElement::ndof, gidx.data(),
                     fem::Quad4RegularElement::ndof, gidx.data(),
                     Ke.data(),
                     ADD_VALUES);

        // si tu as un fe (second membre), insère-le ici via VecSetValues(b, ...)
        // (sinon b reste à 0, ce qui est OK pour un test)
    }

    // ASSEMBLAGE FIN (fusion doublons, tri, routing MPI)
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    // ASSEMBLAGE NEUMANN
    VecSet(b, 0.0);
    for (const auto& nbc : bcs.neumanns) {
        auto nodes = nbc.selector->select(mesh);
        for (auto node : nodes) {
            for (uint32_t d = 0; d < mesh.geo.dim; ++d) {
                // Ajout fx, fy (et fz si dim=3)
                const PetscInt dof = static_cast<PetscInt>(node * mesh.geo.dim + d);
                const PetscScalar val = static_cast<PetscScalar>(nbc.value[d]);
                VecSetValue(b, dof, val, ADD_VALUES);
            }
        }
    }
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    VecSet(x, 0.0);
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);

    std::vector<PetscInt> dirichlet_rows;
    dirichlet_rows.reserve(1024);

    for (const auto& dbc : bcs.dirichlets) {
        auto nodes = dbc.selector->select(mesh);
        for (auto node : nodes) {
            for (uint32_t d = 0; d < mesh.geo.dim; ++d) {
                const PetscInt dof = static_cast<PetscInt>(node * mesh.geo.dim + d);
                dirichlet_rows.push_back(dof);
            }
        }
    }

    if (!dirichlet_rows.empty()) {
        // Hard constraint: zéro des lignes + diag=1, RHS ajusté (valeur imposée = 0)
        MatZeroRows(A,
                    static_cast<PetscInt>(dirichlet_rows.size()),
                    dirichlet_rows.data(),
                    /*diag=*/1.0,
                    /*x=*/x, /*b=*/b);
    }

    //Solve 
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    // KSPSetType(ksp, KSPCG);
    KSPSetFromOptions(ksp); // permet de configurer en ligne de commande

    KSPSolve(ksp, b, x);
    PetscInt its; PetscReal r;
    KSPGetIterationNumber(ksp, &its);
    KSPGetResidualNorm(ksp, &r);
    PetscPrintf(PETSC_COMM_WORLD, "KSP its=%d, residual=%e\n", (int)its, (double)r);

    // Extract solution to std::vector before destroying
    std::vector<double> solution(N);
    PetscScalar *x_array;
    VecGetArray(x, &x_array);
    for (PetscInt i = 0; i < N; ++i) {
        solution[i] = static_cast<double>(x_array[i]);
    }
    VecRestoreArray(x, &x_array);

    // CLEAR
    KSPDestroy(&ksp);
    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x);

    return solution;
}

#endif // ASSEMBLER_PETSC_H