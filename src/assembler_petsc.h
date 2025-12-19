#ifndef ASSEMBLER_PETSC_H
#define ASSEMBLER_PETSC_H

#include <petsc.h>
#include <vector>
#include "mesh_core.h"
#include "material.h"
#include "element.h"
#include "boundary.h"

struct Solution{
    std::vector<double> displacements;
    std::vector<double> energy_map;
};


inline void save_petsc_binary(Mat A, const char* fname /* ex. "A.dat" */) {
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname, FILE_MODE_WRITE, &viewer);  // ouvre un viewer binaire
    MatView(A, viewer);                                                         // export binaire de la matrice
    PetscViewerDestroy(&viewer);
}

void process_element_batch(Mat A, const msh::Mesh& mesh, uint32_t cell_id, 
                           const fem::LinearElasticityMaterial& mat);

void assemble_rigidity(Mat A,
                       const msh::Mesh& mesh,
                       const fem::LinearElasticityMaterial& mat);

double process_energy_batch(const PetscScalar* u_global_ptr, 
                            const msh::Mesh& mesh, 
                            uint32_t cell_id, 
                            const fem::LinearElasticityMaterial& mat);

std::vector<double> compute_energy(const Vec& u,
                                   const msh::Mesh& mesh,
                                   const fem::LinearElasticityMaterial& mat);


inline Solution assembler_petsc(const msh::Mesh& mesh,
                                                                 const fem::LinearElasticityMaterial& mat,
                                                                 const fem::BoundaryConditions& bcs){
    
    const PetscInt dim = static_cast<PetscInt>(mesh.geo.dim);
    const PetscInt n_nodes = static_cast<PetscInt>(mesh.geo.n_nodes());
    const PetscInt N_dof = n_nodes * dim;

    Mat A;
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N_dof, N_dof);


    MatSetType(A, MATBAIJ); // MATSEQBAIJ ou MATMPIBAIJ choisi automatiquement
    MatSetBlockSize(A, dim);

    PetscInt nz_blocks = 9; // estimation grossière du nombre de blocs non-nuls par ligne (pour Quad4Reg)
    MatMPIBAIJSetPreallocation(A, dim, nz_blocks, NULL, nz_blocks, NULL); // Ignoré si sequentiel
    MatSeqBAIJSetPreallocation(A, dim, nz_blocks, NULL); // Ignoré si parallèle
    
    MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
    MatSetFromOptions(A);   // Autorise -mat_type aij, baij, etc. en ligne de commande
    MatSetUp(A);            // Alloue la structure interne (réallocations possibles ensuite)

    Vec b, x;
    VecCreate(PETSC_COMM_WORLD, &b);
    VecSetSizes(b, PETSC_DECIDE, N_dof);
    VecSetFromOptions(b);
    VecDuplicate(b, &x);
    
    // ========================================================================
    // ASSEMBLAGE
    // ========================================================================

    assemble_rigidity(A, mesh, mat);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    
    //save_petsc_binary(A, "A.dat"); // Sauvegarde binaire de la matrice A


    // NEUMANN
    VecSet(b, 0.0);
    auto all_loads = bcs.compute_neumann_loads(mesh);

    for (const auto& load : all_loads) {
        const PetscInt dof = static_cast<PetscInt>(load.node_index * mesh.geo.dim + load.dof_index);
        const PetscScalar val = static_cast<PetscScalar>(load.value);
        VecSetValue(b, dof, val, ADD_VALUES);
    }

    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    VecSet(x, 0.0);
    VecAssemblyBegin(x);
    VecAssemblyEnd(x);

    // DIRICHLET
    std::vector<int32_t> dirichlet_dofs_int32 = bcs.compute_dirichlet_dofs(mesh);

    if (!dirichlet_dofs_int32.empty()) {
        std::vector<PetscInt> dirichlet_rows(dirichlet_dofs_int32.begin(), dirichlet_dofs_int32.end());

        MatZeroRowsColumns(A, 
                           static_cast<PetscInt>(dirichlet_rows.size()),
                           dirichlet_rows.data(),
                           1.0, // Valeur diagonale (1.0 est standard)
                           x,   // Vecteur solution (sera mis à 0 sur les lignes bloquées)
                           b);  // Vecteur RHS (sera modifié pour compenser)
    }
    
    // ========================================================================
    //Solve 
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);

    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPCG); 
    PC pc;
    KSPGetPC(ksp, &pc);
    
    PCSetType(pc, PCHYPRE);               // On active Hypre
    PCHYPRESetType(pc, "boomeramg");      // On choisit BoomerAMG
    PetscOptionsSetValue(NULL, "-pc_hypre_boomeramg_strong_threshold", "0.5");
    KSPSetTolerances(ksp, 1e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); // tol, ksp, rtol, abstol, dtol, maxits


    KSPSetFromOptions(ksp); // permet de configurer en ligne de commande

    // ========================================================================

    std::cout << "Solving linear system with PETSc KSP...\n";
    KSPSolve(ksp, b, x);
    PetscInt its; PetscReal r;
    KSPGetIterationNumber(ksp, &its);
    KSPGetResidualNorm(ksp, &r);
    PetscPrintf(PETSC_COMM_WORLD, "KSP its=%d, residual=%e\n", (int)its, (double)r);

    std::cout << "Iterations: " << its << ", Residual: " << r << "\n";

    // Post process:

    std::vector<double> energy_map = compute_energy(x, mesh, mat);

    // Extract solution to std::vector before destroying
    std::vector<double> solution(N_dof);
    PetscScalar *x_array;
    VecGetArray(x, &x_array);
    for (PetscInt i = 0; i < N_dof; ++i) {
        solution[i] = static_cast<double>(x_array[i]);
    }
    VecRestoreArray(x, &x_array);

    Solution sol;
    sol.displacements = std::move(solution);
    sol.energy_map = std::move(energy_map);

    // CLEAR
    KSPDestroy(&ksp);
    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x);

    return sol;
}

#endif // ASSEMBLER_PETSC_H