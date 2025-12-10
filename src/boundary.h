#ifndef BOUNDARY_H
#define BOUNDARY_H

#include <vector>
#include <memory>
#include <array>
#include <cmath>
#include <algorithm>

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

    struct NodalLoad {
        uint32_t node_index;
        uint32_t dof_index; // 0=x, 1=y, 2=z
        double   value;     // Force en Newtons
    };

    struct NeumannBC {
        std::array<double,3> force_density; // N/m (2D) ou N/m² (3D)
        std::unique_ptr<NodeSelector> selector;

        static NeumannBC constant(std::unique_ptr<NodeSelector> sel, std::array<double,3> val) {
            NeumannBC bc;
            bc.selector = std::move(sel);
            bc.force_density = val;
            return bc;
        }

        // return un vecteur de NodalLoad correspondant aux charges nodales de bc Neumann
        std::vector<NodalLoad> compute_loads(const msh::Mesh& mesh) const {
            std::vector<NodalLoad> loads;

            auto selected_nodes_ids = selector->select(mesh);

            if (selected_nodes_ids.empty()) return loads;

            std::vector<bool> is_selected(mesh.geo.n_nodes(), false);
            for(auto idx : selected_nodes_ids) is_selected[idx] = true;

            for (const auto& elem : mesh.boundary) {
                bool all_nodes_in = true;
                for(auto n : elem.node_ids) {
                    if(!is_selected[n]) {
                        all_nodes_in = false;
                        break;
                    }
                }
                
                if (all_nodes_in) {
                    double weight = 1.0 / static_cast<double>(elem.node_ids.size());

                    for (int d = 0; d < mesh.geo.dim; ++d) {
                        if (std::abs(force_density[d]) < 1e-12) continue;

                        double nodal_force = force_density[d] * elem.measure * weight;

                        for(auto node_idx : elem.node_ids) {
                            loads.push_back({node_idx, (uint32_t)d, nodal_force});
                        }
                    }
                }
            }
            return loads;
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

        // Helpers pour éviter make_unique dans le main
        void add_dirichlet_line_x(double x0, double tol, std::array<double,3> val) {
            add_dirichlet(std::make_unique<LineXSelector>(x0, tol), val);
        }

        void add_neumann_line_x(double x0, double tol, std::array<double,3> val) {
            add_neumann(std::make_unique<LineXSelector>(x0, tol), val);
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

        std::vector<NodalLoad> compute_neumann_loads(const msh::Mesh& mesh) const {
            std::vector<NodalLoad> all_loads;

            all_loads.reserve(neumanns.size() * static_cast<size_t>(std::sqrt(mesh.geo.n_nodes())));
            
            for (const auto& bc : neumanns) {
                auto local_loads = bc.compute_loads(mesh);
                all_loads.insert(all_loads.end(), local_loads.begin(), local_loads.end());
            }
            return all_loads;
        }
    };

} // namespace fem

#endif
