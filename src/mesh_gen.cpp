#include "mesh_gen.h"
#include <cassert>      // Pour les assertions
#include <stdexcept>    // Pour les exceptions

msh::Mesh make_structured_quads_2D(uint32_t Nx, uint32_t Ny, double x0, double y0, double hx, double hy)
{
    msh::Mesh M;
    msh::Geometry& G = M.geo;
    msh::Topology& T = M.topo;

    const uint32_t nx = Nx + 1;
    const uint32_t ny = Ny + 1;
    const uint32_t Nn = nx * ny;     // nb de noeuds
    const uint32_t Nc = Nx * Ny;     // nb de cellules

    // --- Geometry: coord. SoA (z vide en 2D) ---
    G.x.resize(Nn);
    G.y.resize(Nn);
    G.z.clear();

    auto nid = [nx](uint32_t i, uint32_t j) -> uint32_t {
        return i + j * nx;
    };

    // --- Remplissage des coordonnées ---
    for (uint32_t j = 0; j < ny; ++j) {
        for (uint32_t i = 0; i < nx; ++i) {
            const uint32_t n = nid(i, j);
            G.x[n] = x0 + i * hx;
            G.y[n] = y0 + j * hy;
        }
    }

    // --- Topology: c2n (CSR ragged), ctype ---
    T.ctype.resize(Nc, msh::CellType::Quad4Reg);
    T.c2n_offsets.resize(Nc + 1);
    T.c2n_indices.resize(4 * Nc);

    uint32_t off = 0;
    for (uint32_t j = 0; j < Ny; ++j) {
        for (uint32_t i = 0; i < Nx; ++i) {
            const uint32_t c = i + j * Nx;
            T.c2n_offsets[c] = off;
            // Quad local (orientation CCW) :
            //  n3 --- n2
            //  |       |
            //  n0 --- n1
            const uint32_t n0 = nid(i,     j);
            const uint32_t n1 = nid(i + 1, j);
            const uint32_t n2 = nid(i + 1, j + 1);
            const uint32_t n3 = nid(i,     j + 1);

            T.c2n_indices[off + 0] = n0;
            T.c2n_indices[off + 1] = n1;
            T.c2n_indices[off + 2] = n2;
            T.c2n_indices[off + 3] = n3;
            off += 4;
        }
    }
    T.c2n_offsets[Nc] = off;

    return M;
}
