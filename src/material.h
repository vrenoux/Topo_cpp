
#ifndef MATERIAL_H
#define MATERIAL_H

#include <array>
#include <stdexcept>

namespace fem {

struct LinearElasticityMaterial {
    // ------------------------------------------------------------------------
    // Renvoie la matrice constitutive D pour le matériau élastique linéaire isotrope
    // ------------------------------------------------------------------------
    double E; // Module de Young
    double v; // Coefficient de Poisson
    int ptype; // 1: plane stress, 2: plane strain, 3: 3D
    double thickness = 1.0; // Épaisseur (pour éléments 2D)

    // Matrice constitutive D (max 6x6 pour 3D)
    std::array<double, 36> D{}; // Stockée en row-major
    int mat_size; // Taille réelle (3 pour 2D, 6 pour 3D)

    LinearElasticityMaterial(double E_, double v_, int ptype_, double thickness_ = 1.0) : E(E_), v(v_), ptype(ptype_), thickness(thickness_) {
        compute_constitutive();
    }

    void compute_constitutive() {
        if (ptype == 1) { // Plane stress
            mat_size = 3;
            double coef = E / (1 - v * v);
            D = {coef, coef * v, 0,
                 coef * v, coef, 0,
                 0, 0, coef * (1 - v) / 2};
        } else if (ptype == 2) { // Plane strain
            mat_size = 3;
            double coef = E / ((1 + v) * (1 - 2 * v));
            D = {coef * (1 - v), coef * v, 0,
                 coef * v, coef * (1 - v), 0,
                 0, 0, coef * (1 - 2 * v) / 2};
        } else if (ptype == 3) { // 3D
            mat_size = 6;
            double coef = E / ((1 + v) * (1 - 2 * v));
            D = {coef * (1 - v), coef * v, coef * v, 0, 0, 0,
                 coef * v, coef * (1 - v), coef * v, 0, 0, 0,
                 coef * v, coef * v, coef * (1 - v), 0, 0, 0,
                 0, 0, 0, coef * (1 - 2 * v) / 2, 0, 0,
                 0, 0, 0, 0, coef * (1 - 2 * v) / 2, 0,
                 0, 0, 0, 0, 0, coef * (1 - 2 * v) / 2};
        } else {
            throw std::invalid_argument("LinearElasticityMaterial::compute_constitutive() - ptype must be 1, 2, or 3");
        }
    }
};

} // namespace fem

#endif
