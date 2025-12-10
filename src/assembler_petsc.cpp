#include "assembler_petsc.h"
#include <petscmat.h>
#include "mesh_core.h"

PetscErrorCode AttachMechanicsNullSpace(Mat A, const msh::Mesh& mesh) {
    PetscErrorCode ierr;
    MatNullSpace nullSpace;
    
    int dim = mesh.geo.dim;
    int n_modes = (dim == 2) ? 3 : 6; // 3 modes en 2D, 6 modes en 3D
    
    // 1. Création des vecteurs PETSc pour stocker les modes
    Vec rigidModes[6]; 
    ierr = MatCreateVecs(A, &rigidModes[0], NULL); CHKERRQ(ierr);
    for (int i = 1; i < n_modes; i++) {
        ierr = VecDuplicate(rigidModes[0], &rigidModes[i]); CHKERRQ(ierr);
    }

    // =========================================================
    // 2. Remplissage des Translations (Commun 2D et 3D)
    // =========================================================
    // Mode 0: Translation X (1.0 sur Ux, 0 ailleurs)
    ierr = VecSet(rigidModes[0], 0.0); CHKERRQ(ierr);
    ierr = VecStrideSet(rigidModes[0], 0, 1.0); CHKERRQ(ierr);

    // Mode 1: Translation Y (1.0 sur Uy, 0 ailleurs)
    ierr = VecSet(rigidModes[1], 0.0); CHKERRQ(ierr);
    ierr = VecStrideSet(rigidModes[1], 1, 1.0); CHKERRQ(ierr);

    if (dim == 3) {
        // Mode 2: Translation Z (1.0 sur Uz, 0 ailleurs)
        ierr = VecSet(rigidModes[2], 0.0); CHKERRQ(ierr);
        ierr = VecStrideSet(rigidModes[2], 2, 1.0); CHKERRQ(ierr);
    }

    // =========================================================
    // 3. Remplissage des Rotations (Dépend de x, y, z)
    // =========================================================
    
    PetscInt Istart, Iend;
    ierr = VecGetOwnershipRange(rigidModes[0], &Istart, &Iend); CHKERRQ(ierr);

    // On prépare l'accès aux tableaux PETSc pour les modes de rotation
    // En 2D, la rotation est le mode [2]. En 3D, ce sont [3], [4], [5].
    PetscScalar *rotPtrs[3]; 
    int rotStartIndex = (dim == 2) ? 2 : 3;
    int nRotations = (dim == 2) ? 1 : 3;

    for(int i=0; i<nRotations; i++) {
        ierr = VecGetArray(rigidModes[rotStartIndex + i], &rotPtrs[i]); CHKERRQ(ierr);
    }

    // Boucle locale sur les DoFs que ce processeur possède
    for (PetscInt row = Istart; row < Iend; row++) {
        // Décodage de l'indice : Quel noeud ? Quel DoF (x, y ou z) ?
        PetscInt globalNodeIdx = row / dim; 
        PetscInt dof = row % dim; 

        // Sécurité : On s'assure qu'on ne déborde pas du maillage local
        if (globalNodeIdx >= mesh.geo.n_nodes()) continue; 

        // Récupération des coordonnés depuis ta structure
        double x = mesh.geo.x[globalNodeIdx];
        double y = mesh.geo.y[globalNodeIdx];
        double z = (dim == 3) ? mesh.geo.z[globalNodeIdx] : 0.0;

        // Indice local dans le tableau PETSc
        PetscInt localIdx = row - Istart;

        if (dim == 2) {
            // --- 2D : Une seule rotation (autour de Z) ---
            // Mode [2] : (-y, x)
            if (dof == 0) rotPtrs[0][localIdx] = -y; // Ux
            if (dof == 1) rotPtrs[0][localIdx] =  x; // Uy
        } 
        else if (dim == 3) {
            // --- 3D : Trois rotations ---
            
            // Mode [3] : Rot X (0, -z, y)
            if (dof == 1) rotPtrs[0][localIdx] = -z; // Uy
            if (dof == 2) rotPtrs[0][localIdx] =  y; // Uz

            // Mode [4] : Rot Y (z, 0, -x)
            if (dof == 0) rotPtrs[1][localIdx] =  z; // Ux
            if (dof == 2) rotPtrs[1][localIdx] = -x; // Uz

            // Mode [5] : Rot Z (-y, x, 0)
            if (dof == 0) rotPtrs[2][localIdx] = -y; // Ux
            if (dof == 1) rotPtrs[2][localIdx] =  x; // Uy
        }
    }

    // Restauration des tableaux
    for(int i=0; i<nRotations; i++) {
        ierr = VecRestoreArray(rigidModes[rotStartIndex + i], &rotPtrs[i]); CHKERRQ(ierr);
    }

    // =========================================================
    // 4. Normalisation et Création
    // =========================================================
    
    // Très important pour Hypre : Normaliser les vecteurs
    for (int i = 0; i < n_modes; i++) {
        ierr = VecNormalize(rigidModes[i], NULL); CHKERRQ(ierr);
    }

    // Création de l'objet NullSpace
    // PETSC_FALSE car c'est le "Near" NullSpace (on a des Dirichlet ailleurs)
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)A), PETSC_FALSE, n_modes, rigidModes, &nullSpace); CHKERRQ(ierr);

    // Attachement à la matrice
    ierr = MatSetNearNullSpace(A, nullSpace); CHKERRQ(ierr);

    // Nettoyage
    ierr = MatNullSpaceDestroy(&nullSpace); CHKERRQ(ierr);
    for (int i = 0; i < n_modes; i++) {
        ierr = VecDestroy(&rigidModes[i]); CHKERRQ(ierr);
    }

    return 0;
}