#include "element.h"
#include "material.h"

#include <vector>
#include <array>
#include <cmath>

namespace fem {

void Quad4RegularElement::compute_stiffness(std::array<std::array<double, dim>, n_nodes>& coords,
                                std::array<double, ndof * ndof>& Ke) const {
    Ke.fill(0.0);

    // Calcul de la taille approximative (pour un élément régulier)
    double xmin = coords[0][0], xmax = coords[0][0];
    double ymin = coords[0][1], ymax = coords[0][1];
    for (int i = 1; i < n_nodes; ++i) {
        xmin = std::min(xmin, coords[i][0]);
        xmax = std::max(xmax, coords[i][0]);
        ymin = std::min(ymin, coords[i][1]);
        ymax = std::max(ymax, coords[i][1]);
    }
    double elx = xmax - xmin;
    double ely = ymax - ymin;

    constexpr double inv_sqrt3 = 1.0 / std::sqrt(3.0);
    std::array<double, 4> xsi = {-1, 1, 1, -1};
    std::array<double, 4> eta = {-1, -1, 1, 1};

    for (int i = 0; i < 4; ++i) {
        double xgp = xsi[i] * inv_sqrt3 * elx;
        double ygp = eta[i] * inv_sqrt3 * ely;

        // Matrice B (3x8)
        std::array<double, 24> B = {
            -(elx - ygp), 0, elx - ygp, 0, elx + ygp, 0, -(elx + ygp), 0,
            0, -(ely - xgp), 0, -(ely + xgp), 0, ely + xgp, 0, ely - xgp,
            -(ely - xgp), -(elx - ygp), -(ely + xgp), elx - ygp,
            ely + xgp, elx + ygp, ely - xgp, -(elx + ygp)
        };

        for (double& val : B) val /= (4.0 * elx * ely);

        accumulate_stiffness(Ke, B, mat.D, elx * ely * mat.thickness);
    }
}

void Quad4RegularElement::accumulate_stiffness(std::array<double, ndof * ndof>& Ke,
                            const std::array<double, 24>& B,
                            const std::array<double, 36>& D,
                            double weight) const {
    // Bᵀ * D * B
    // B: 3x8, D: 3x3
    std::array<double, 24> DB{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 8; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += D[i * 3 + k] * B[k * 8 + j];
            }
            DB[i * 8 + j] = sum;
        }
    }

    // Ke += Bᵀ * (DB)
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                sum += B[k * 8 + i] * DB[k * 8 + j];
            }
            Ke[i * 8 + j] += sum * weight;
        }
    }
}


double Quad4RegularElement::compute_energy(const std::array<std::array<double, dim>, n_nodes>& coords,
                                const std::array<double, ndof>& u_elem) const {
    
    double energy = 0.0;

    // 1. Géométrie
    double xmin = coords[0][0], xmax = coords[0][0];
    double ymin = coords[0][1], ymax = coords[0][1];
    for (int i = 1; i < n_nodes; ++i) {
        if(coords[i][0] < xmin) xmin = coords[i][0];
        if(coords[i][0] > xmax) xmax = coords[i][0];
        if(coords[i][1] < ymin) ymin = coords[i][1];
        if(coords[i][1] > ymax) ymax = coords[i][1];
    }
    double elx = xmax - xmin;
    double ely = ymax - ymin;

    // 2. Boucle Gauss
    constexpr double inv_sqrt3 = 0.577350269189626;
    const double xsi_ref[4] = {-1, 1, 1, -1};
    const double eta_ref[4] = {-1, -1, 1, 1};

    for (int i = 0; i < 4; ++i) {
        double xsi = xsi_ref[i];
        double eta = eta_ref[i];
        
        // Coordonnées Gauss locales
        double xgp = xsi * inv_sqrt3 * (elx / 2.0);
        double ygp = eta * inv_sqrt3 * (ely / 2.0);

        // Calcul de B
        std::array<double, 24> B;
        compute_B_matrix(B, elx, ely, xgp, ygp);

        // Poids d'intégration
        double detJ_w = (elx * ely / 4.0) * mat.thickness; // Poids w=1 inclus

        // Calcul de la Déformation : Epsilon = B * u (Vecteur size 3)
        std::array<double, 3> eps = {0.0, 0.0, 0.0};
        for(int k=0; k<8; ++k) {
            double u_val = u_elem[k];
            eps[0] += B[0*8 + k] * u_val;
            eps[1] += B[1*8 + k] * u_val;
            eps[2] += B[2*8 + k] * u_val;
        }

        // Calcul de la Contrainte : Sigma = D * Epsilon (Vecteur size 3)
        std::array<double, 3> sig = {0.0, 0.0, 0.0};
        
        for(int r=0; r<3; ++r) {
            for(int c=0; c<3; ++c) {
                sig[r] += mat.D[r*3 + c] * eps[c];
            }
        }

        // C. Densité d'énergie = 0.5 * (Sigma : Epsilon)
        double strain_energy_density = 0.0;
        for(int k=0; k<3; ++k) {
            strain_energy_density += sig[k] * eps[k];
        }
        
        energy += 0.5 * strain_energy_density * detJ_w;
    }

    return energy;
}

void Quad4RegularElement::compute_B_matrix(std::array<double, 24>& B, double elx, double ely, double xgp, double ygp) const {
    // Formule analytique pour Quad4 Régulier
    // Basée sur les dérivées des fonctions de forme
    
    // Row 1 (Strain XX -> dN/dx)
    B[0] = -(elx - ygp); B[1] = 0;
    B[2] =  (elx - ygp); B[3] = 0;
    B[4] =  (elx + ygp); B[5] = 0;
    B[6] = -(elx + ygp); B[7] = 0;

    // Row 2 (Strain YY -> dN/dy)
    B[8]  = 0; B[9]  = -(ely - xgp);
    B[10] = 0; B[11] = -(ely + xgp);
    B[12] = 0; B[13] =  (ely + xgp);
    B[14] = 0; B[15] =  (ely - xgp);

    // Row 3 (Shear XY -> dN/dy + dN/dx)
    B[16] = -(ely - xgp); B[17] = -(elx - ygp);
    B[18] = -(ely + xgp); B[19] =  (elx - ygp);
    B[20] =  (ely + xgp); B[21] =  (elx + ygp);
    B[22] =  (ely - xgp); B[23] = -(elx + ygp);

    // Scaling global (Jacobien inverse partiel)
    double factor = 1.0 / (4.0 * elx * ely);
    for(double& val : B) val *= factor;
}


} // namespace fem