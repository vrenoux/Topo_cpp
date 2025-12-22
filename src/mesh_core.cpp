#include "mesh_core.h"

#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <array>
#include <iostream>

namespace msh {

struct FaceKey {
    std::array<uint32_t, 4> nodes; // Max 4 noeuds (Hex8 face = Quad4)
    uint8_t n_cnt;                 // Nombre de noeuds (2, 3 ou 4)

    // Constructeur optimisé : prend une liste d'initialisation {a, b, c...}
    FaceKey(std::initializer_list<uint32_t> list) {
        n_cnt = static_cast<uint8_t>(list.size());
        auto it = list.begin();
        for(int i = 0; i < n_cnt; ++i) {
            nodes[i] = *it;
            ++it;
        }
        // TRI LOCAL : Indispensable pour l'unicité (1-2-3 == 3-1-2)
        std::sort(nodes.begin(), nodes.begin() + n_cnt);
    }

    // Opérateur < pour std::sort
    bool operator<(const FaceKey& other) const {
        if (n_cnt != other.n_cnt) return n_cnt < other.n_cnt;
        for(int i = 0; i < n_cnt; ++i) {
            if (nodes[i] != other.nodes[i]) return nodes[i] < other.nodes[i];
        }
        return false;
    }

    // Opérateur == pour détecter les doublons
    bool operator==(const FaceKey& other) const {
        if (n_cnt != other.n_cnt) return false;
        for(int i = 0; i < n_cnt; ++i) {
            if (nodes[i] != other.nodes[i]) return false;
        }
        return true;
    }
};

void Mesh::compute_boundary_elements() {
    boundary.clear();

    std::vector<FaceKey> all_faces;
    all_faces.reserve(topo.n_cells() * 6);

    // 1 : Extraction de toutes les faces (soupe de faces)

    for (size_t i = 0; i < topo.n_cells(); ++i) {
        // On récupère les indices bruts des noeuds de l'élément i
        // (Supposons que get_nodes_cell renvoie un const std::vector<uint32_t>& ou un span)
        const auto& n = topo.get_nodes_cell(i);
        CellType type = topo.ctype[i];

        // On "pousse" les faces directement dans le vecteur.
        // Les indices ci-dessous suivent la numérotation standard (VTK/Gmsh).
        
        switch (type) {
            // --- 2D ---
            case CellType::Tri3: // Faces = Arêtes (2 noeuds)
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[0], n[1]});
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[1], n[2]});
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[2], n[0]});
                break;

            case CellType::Quad4:
            case CellType::Quad4Reg: // Faces = Arêtes (2 noeuds)
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[0], n[1]});
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[1], n[2]});
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[2], n[3]});
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[3], n[0]});
                break;

            // --- 3D ---
            case CellType::Tet4: // Faces = Triangles (3 noeuds)
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[0], n[2], n[1]});
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[0], n[1], n[3]});
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[1], n[2], n[3]});
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[0], n[3], n[2]});
                break;

            case CellType::Hex8: // Faces = Quads (4 noeuds)
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[0], n[3], n[2], n[1]}); // Bottom
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[4], n[5], n[6], n[7]}); // Top
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[0], n[1], n[5], n[4]}); // Front
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[1], n[2], n[6], n[5]}); // Right
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[2], n[3], n[7], n[6]}); // Back
                all_faces.emplace_back(std::initializer_list<uint32_t>{n[3], n[0], n[4], n[7]}); // Left 
                break;

            default:

                break;
        }
    }

    // 2 : Le Tri (Sort) - O(N log N)
    std::sort(all_faces.begin(), all_faces.end());

    // 3 : Le Scan (Linear) - O(N)
    size_t i = 0;
    while (i < all_faces.size()) {
        bool is_internal = (i + 1 < all_faces.size()) && (all_faces[i] == all_faces[i + 1]);

        if (is_internal) {
            // C'est une face INTERNE (partagée par 2 éléments).
            const auto& current_ref = all_faces[i];
            while (i < all_faces.size() && all_faces[i] == current_ref) {
                i++;
            }
        } else {
            // C'est une face de BORD (unique).
            BoundaryElement be;
            be.node_ids.assign(all_faces[i].nodes.begin(), all_faces[i].nodes.begin() + all_faces[i].n_cnt);
            
            compute_measure_and_center(be); 
            
            boundary.push_back(std::move(be));
            i++; 
        }
    }
}

void Mesh::compute_measure_and_center(BoundaryElement& be) {
    // Calcul du centre simple (moyenne)
    be.center_x = 0; be.center_y = 0; be.center_z = 0;
    for(auto idx : be.node_ids) {
        be.center_x += geo.x[idx];
        be.center_y += geo.y[idx];
        if(geo.z.size() > idx) be.center_z += geo.z[idx];
    }
    double n = (double)be.node_ids.size();
    be.center_x /= n; be.center_y /= n; be.center_z /= n;

    // Calcul de la mesure (Longueur ou Aire)
    if (be.node_ids.size() == 2) { 
        // --- LIGNE (2D) ---
        double dx = geo.x[be.node_ids[0]] - geo.x[be.node_ids[1]];
        double dy = geo.y[be.node_ids[0]] - geo.y[be.node_ids[1]];
        be.measure = std::sqrt(dx*dx + dy*dy);
    } 
    else if (be.node_ids.size() == 3) {
        // --- TRIANGLE (3D) ---
        // Aire = 0.5 * || AB x AC ||
        // Vecteurs AB et AC
        double ABx = geo.x[be.node_ids[1]] - geo.x[be.node_ids[0]];
        double ABy = geo.y[be.node_ids[1]] - geo.y[be.node_ids[0]];
        double ABz = geo.z[be.node_ids[1]] - geo.z[be.node_ids[0]];

        double ACx = geo.x[be.node_ids[2]] - geo.x[be.node_ids[0]];
        double ACy = geo.y[be.node_ids[2]] - geo.y[be.node_ids[0]];
        double ACz = geo.z[be.node_ids[2]] - geo.z[be.node_ids[0]];

        // Produit vectoriel
        double cx = ABy*ACz - ABz*ACy;
        double cy = ABz*ACx - ABx*ACz;
        double cz = ABx*ACy - ABy*ACx;
        
        be.measure = 0.5 * std::sqrt(cx*cx + cy*cy + cz*cz);
    }
    else if (be.node_ids.size() == 4) {
        // --- QUAD (3D) ---
        // Approx simple : Aire = somme de 2 triangles ou produit diagonales
        // D1 = P2 - P0, D2 = P3 - P1
        double D1x = geo.x[be.node_ids[2]] - geo.x[be.node_ids[0]];
        double D1y = geo.y[be.node_ids[2]] - geo.y[be.node_ids[0]];
        double D1z = geo.z[be.node_ids[2]] - geo.z[be.node_ids[0]];

        double D2x = geo.x[be.node_ids[3]] - geo.x[be.node_ids[1]];
        double D2y = geo.y[be.node_ids[3]] - geo.y[be.node_ids[1]];
        double D2z = geo.z[be.node_ids[3]] - geo.z[be.node_ids[1]];

        double cx = D1y*D2z - D1z*D2y;
        double cy = D1z*D2x - D1x*D2z;
        double cz = D1x*D2y - D1y*D2x;

        be.measure = 0.5 * std::sqrt(cx*cx + cy*cy + cz*cz);
    }
}

void Mesh::compute_volume_cells() {
    volume_cells.resize(topo.n_cells(), 0.0);

    for (size_t i = 0; i < topo.n_cells(); ++i) {
        CellType type = topo.ctype[i];
        const auto& n = topo.get_nodes_cell(i);

        double cell_volume = 0.0;

        switch (type) {
            case CellType::Quad4Reg: {

                const uint32_t n0 = n[0];
                const uint32_t n1 = n[1];
                const uint32_t n2 = n[2];
                const uint32_t n3 = n[3];

                const double x0 = geo.x[n0], y0 = geo.y[n0];
                const double x1 = geo.x[n1], y1 = geo.y[n1];
                const double x2 = geo.x[n2], y2 = geo.y[n2];
                const double x3 = geo.x[n3], y3 = geo.y[n3];

                double twice_area = (x0 * y1 + x1 * y2 + x2 * y3 + x3 * y0)
                                   - (y0 * x1 + y1 * x2 + y2 * x3 + y3 * x0);

                cell_volume = 0.5 * std::abs(twice_area);
                break;
            }


        }

        volume_cells[i] = cell_volume;
    }

    volume_total = std::accumulate(volume_cells.begin(), volume_cells.end(), 0.0);
}
} // namespace msh