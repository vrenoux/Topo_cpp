#ifndef FEM_H
#define FEM_H

#include <vector>
#include <memory>
#include <array>
#include <cmath>

#include "mesh_core.h"
#include "linalg.h"

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

        void add_dirichlet(std::unique_ptr<NodeSelector> sel, std::array<double,3> val) {
            dirichlets.push_back(DirichletBC::constant(std::move(sel), val));
        }

        void add_neumann(std::unique_ptr<NodeSelector> sel, std::array<double,3> val) {
            neumanns.push_back(NeumannBC::constant(std::move(sel), val));
        }

        void apply_dirichlet(SparseMatrixCSR& K, const BoundaryConditions& bcs,const msh::Mesh& mesh) {
            std::vector<uint32_t> dofs;
            uint32_t dim = mesh.geo.dim;
            for (const auto& bc : bcs.dirichlets) {
                auto nodes = bc.selector->select(mesh);
                for (auto node : nodes) {
                    for (uint32_t d = 0; d < dim; ++d) {
                        dofs.push_back(node * dim + d);
                    }
                }
            }
            K.apply_dirichlet_batch(dofs);
        }


        VectorDense apply_neumann(const msh::Mesh& mesh) const {
            uint32_t ndofs = mesh.geo.x.size() * mesh.geo.dim;
            VectorDense F(ndofs); // initialisé à 0.0

            for (const auto& bc : neumanns) {
                auto nodes = bc.selector->select(mesh);
                for (auto node : nodes) {
                    for (uint32_t d = 0; d < mesh.geo.dim; ++d) {
                        uint32_t dof = node * mesh.geo.dim + d;
                        F[dof] += bc.value[d];
                    }
                }
            }
            return F;
        }


        // Helpers pour éviter make_unique dans le main
        void add_dirichlet_line_x(double x0, double tol, std::array<double,3> val) {
            add_dirichlet(std::make_unique<LineXSelector>(x0, tol), val);
        }

        void add_neumann_line_x(double x0, double tol, std::array<double,3> val) {
            add_neumann(std::make_unique<LineXSelector>(x0, tol), val);
        }

        };


} // namespace fem

#endif
