#include "optimizer.h"

#include <vector>
#include <iostream>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <chrono>

#include "assembler_petsc.h"
#include "material.h"
#include "mesh_core.h"
#include "boundary.h"
#include "MMA.h"
#include "mesh_io.h"

// ============================================================================
// Public
// ============================================================================

void Optimizer::Optimize(const fem::LinearElasticityMaterial& mat,
                         msh::Mesh& M,
                         const fem::BoundaryConditions& bcs) {
    
    // Créer le dossier des résultats
    std::filesystem::create_directories("results");
    
    const size_t n_cells = M.topo.n_cells();

    std::vector<double> volume_dens_cell(n_cells);  
    std::vector<double> constraint1(1);

    double volume_dens_total = 0.0;            
    MMA optimizer(n,1);

    std::vector<double> dc1dx(n);

    for (size_t i = 0; i < n; ++i) {
        dc1dx[i] = M.volume_cells[i];
    }
    
    // Vecteur PETSc persistant pour warm start
    Vec x_warmstart = nullptr;

    for (iter=0; iter < max_iter; ++iter) {
        auto time_iter_start = std::chrono::high_resolution_clock::now();
        // SOLVE FEA avec warm start
        Solution sol = assembler_petsc(M, mat, bcs, penalization, x_warmstart);

        // COMPUTE CONSTRAINT
        for (size_t i = 0; i < n_cells; ++i) {
            volume_dens_cell[i] = M.volume_cells[i] * M.density[i];
        }
        volume_dens_total = std::accumulate(volume_dens_cell.begin(), volume_dens_cell.end(), 0.0);
        constraint1[0] = volume_dens_total / M.volume_total - volume_fraction_constraint;
        

        // PRINT INFO
        

        // WRITE VTK OUTPUT
        std::ostringstream vtk_filename;
        vtk_filename << "results/iter_" << std::setfill('0') << std::setw(6) << iter << ".vtk";
        std::vector<msh::VtkField> fields;
        fields.emplace_back(msh::VtkField{"Displacement",msh::FieldType::Vector,msh::FieldLocation::Node,sol.displacements});
        fields.emplace_back(msh::VtkField{"Energy",msh::FieldType::Scalar,msh::FieldLocation::Cell,sol.energy_map});
        fields.emplace_back(msh::VtkField{"Density",msh::FieldType::Scalar,msh::FieldLocation::Cell,M.density});
        msh::write_vtk(M, vtk_filename.str(), fields);

        // UPDATE DESIGN VARIABLES USING MMA
        std::vector<double> dfdc(n);
        for (size_t i = 0; i < n_cells; ++i) {
            dfdc[i] = -sol.energy_map[i] * penalization.get_penalization_derivative(M.density[i]); 
        }

        auto time_MMA_start = std::chrono::high_resolution_clock::now();
        optimizer.Update(M.density, dfdc, constraint1, dc1dx, xmin, xmax);
        auto time_MMA_end = std::chrono::high_resolution_clock::now();


        auto time_iter_end = std::chrono::high_resolution_clock::now();

        std::cout << "It: " << std::setw(3) << iter 
                  << " - f0: " << std::setw(10) << (int)sol.compliance 
                  << " - f1: " << std::fixed << std::setprecision(3) << std::setw(8) << constraint1[0]
                  << " - FEA ite: " << std::setw(4) << sol.FEM_iterations
                  << " - FEA time: " << std::fixed << std::setprecision(3) << std::setw(7) << sol.solve_time << "s"
                  << " - MMA time: " << std::fixed << std::setprecision(3) << std::setw(7) << std::chrono::duration<double>(time_MMA_end - time_MMA_start).count() << "s"
                  << " - Total time: " << std::fixed << std::setprecision(3) << std::setw(7) << std::chrono::duration<double>(time_iter_end - time_iter_start).count() << "s"
                  << std::endl;

        // Sauvegarder la solution pour warm start à l'itération suivante
        if (x_warmstart != nullptr) {
            VecDestroy(&x_warmstart);
        }
        // Créer un vecteur pour la solution
        PetscInt N_dof = M.geo.n_nodes() * M.geo.dim;
        VecCreate(PETSC_COMM_WORLD, &x_warmstart);
        VecSetSizes(x_warmstart, PETSC_DECIDE, N_dof);
        VecSetFromOptions(x_warmstart);
        // Copier les déplacements dans le vecteur PETSc
        for (size_t i = 0; i < sol.displacements.size(); ++i) {
            VecSetValue(x_warmstart, i, sol.displacements[i], INSERT_VALUES);
        }
        VecAssemblyBegin(x_warmstart);
        VecAssemblyEnd(x_warmstart);

    }

    // Nettoyer le vecteur PETSc
    if (x_warmstart != nullptr) {
        VecDestroy(&x_warmstart);
    }

}