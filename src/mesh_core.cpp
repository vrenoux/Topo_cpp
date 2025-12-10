#include "mesh_core.h"

#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <array>
#include <iostream>

namespace msh {

void Mesh::compute_boundary_elements() {
    boundary.clear();

    std::map<std::vector<uint32_t>, int> face_counts;

    // 1. Compter les occurences des faces
    for(size_t i=0; i<topo.n_cells(); ++i) {
        auto nodes = topo.get_nodes_cell(i);
        CellType type = topo.ctype[i];
        
        // Récupérer les faces locales brutes
        auto faces = get_local_faces(type, nodes);

        for(auto& face : faces) {
            // On trie les indices pour que la face (1,5,2) soit égale à (1,2,5)
            std::vector<uint32_t> sorted_face = face;
            std::sort(sorted_face.begin(), sorted_face.end());
            face_counts[sorted_face]++;
        }
    }
    // 2. Garder celles qui apparaissent 1 seule fois
    for(auto const& [key_nodes, count] : face_counts) {
        if(count == 1) {
            // C'est une frontière !
            BoundaryElement be;
            be.node_ids = key_nodes; // (Note: elles sont triées ici, pour l'intégration ça ne change rien si shape function simple)
            
            compute_measure_and_center(be);
            boundary.push_back(be);
        }
    }
}

std::vector<std::vector<uint32_t>> Mesh::get_local_faces(CellType type, const std::vector<uint32_t>& n) const {
    std::vector<std::vector<uint32_t>> faces;
            
    switch(type) {
        case CellType::Tri3: // 2D -> Faces sont des lignes
            faces = {{n[0], n[1]}, {n[1], n[2]}, {n[2], n[0]}}; 
            break;
        case CellType::Quad4: // 2D -> Faces sont des lignes
            faces = {{n[0], n[1]}, {n[1], n[2]}, {n[2], n[3]}, {n[3], n[0]}};
            break;
        case CellType::Quad4Reg: // 2D -> Faces sont des lignes
            faces = {{n[0], n[1]}, {n[1], n[2]}, {n[2], n[3]}, {n[3], n[0]}};
            break;
        case CellType::Tet4: // 3D -> Faces sont des Tri3
            // Faces d'un tétraèdre : (0,1,2), (0,1,3), (1,2,3), (0,2,3)
            faces = {{n[0],n[2],n[1]}, {n[0],n[1],n[3]}, {n[1],n[2],n[3]}, {n[0],n[3],n[2]}};
            break;
        case CellType::Hex8: // 3D -> Faces sont des Quad4
            // 6 faces (indices standards VTK/Gmsh)
            faces = {
                {n[0],n[3],n[2],n[1]}, {n[4],n[5],n[6],n[7]}, // Bottom, Top
                {n[0],n[1],n[5],n[4]}, {n[1],n[2],n[6],n[5]}, // Side
                {n[2],n[3],n[7],n[6]}, {n[3],n[0],n[4],n[7]}  // Side
            };
            break;
        default: break;
    }
    return faces;
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


} // namespace msh