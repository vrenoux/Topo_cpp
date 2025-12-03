#ifndef ELEMENT_H
#define ELEMENT_H

#include "material.h"

#include <vector>
#include <array>
#include <cmath>

namespace fem {

struct Quad4RegularElement {
    static constexpr uint32_t dim = 2; // 2D
    static constexpr int n_nodes = 4; // 4 nodes
    static constexpr int ndof = 8; // 4 nodes * 2 dof
    std::array<uint32_t, 4>node_ids; // indices des nœuds dans le Mesh
    double thickness;                // utile pour 2D
    const LinearElasticityMaterial& mat;
    
    Quad4RegularElement(const std::array<uint32_t, n_nodes>& nodes, double t, const LinearElasticityMaterial& m)
        : node_ids(nodes), thickness(t), mat(m) {}

    inline void compute_stiffness(std::array<std::array<double, dim>, n_nodes>& coords,
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

            accumulate_stiffness(Ke, B, mat.D, elx * ely * thickness);
        }
    }


private:
    inline void accumulate_stiffness(std::array<double, ndof * ndof>& Ke,
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
};


} // namespace fem

#endif // ELEMENT_H