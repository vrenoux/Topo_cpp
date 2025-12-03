#ifndef FEM_H
#define FEM_H

#include <vector>
#include <memory>
#include <array>
#include <cmath>

#include "mesh_core.h"

namespace fem {

    // ------------------------------------------------------------------------
    // Primitives pour Boundary Conditions
    // ------------------------------------------------------------------------

    // Selecteur de noeuds (interface abstraite)

    struct NodeSelector {
        virtual ~NodeSelector() = default;
        virtual std::vector<uint32_t> select(const msh::Mesh& M) const = 0;
    };

    // Primitives

    // Ligne infinie vertical x = x0 avec tolérance tol
    struct LineXSelector : NodeSelector {
        double x0;
        double tol;

        LineXSelector(double x0_, double tol_) : x0(x0_), tol(tol_) {}

        std::vector<uint32_t> select(const msh::Mesh& M) const override {
            std::vector<uint32_t> ids;
            for (uint32_t i = 0; i < M.geo.x.size(); ++i) {
                if (std::abs(M.geo.x[i] - x0) <= tol) {
                    ids.push_back(i);
                }
            }
            return ids;
        }
    };

    // ------------------------------------------------------------------------
    // Boundary conditions
    // ------------------------------------------------------------------------

    struct DirichletBC {
        std::array<double,3> value; // ux, uy, uz
        std::unique_ptr<NodeSelector> selector;

        static DirichletBC constant(std::unique_ptr<NodeSelector> sel, std::array<double,3> val) {
            DirichletBC bc;
            bc.selector = std::move(sel);
            bc.value = val;
            return bc;
        }
    };

    struct NeumannBC {
        std::array<double,3> value; // fx, fy, fz
        std::unique_ptr<NodeSelector> selector;

        static NeumannBC constant(std::unique_ptr<NodeSelector> sel, std::array<double,3> val) {
            NeumannBC bc;
            bc.selector = std::move(sel);
            bc.value = val;
            return bc;
        }

    };

    struct BoundaryConditions {
        std::vector<DirichletBC> dirichlets;
        std::vector<NeumannBC>   neumanns;

        void add(DirichletBC bc) { dirichlets.push_back(std::move(bc)); }
        void add(NeumannBC bc)   { neumanns.push_back(std::move(bc));   }

        };


} // namespace fem

#endif
