#ifndef MESH__GEN_H
#define MESH__GEN_H

#include "mesh_core.h"
#include <vector>
#include <cstdint>

msh::Mesh make_structured_quads_2D(uint32_t Nx, uint32_t Ny,
                              double x0 = 0.0, double y0 = 0.0,
                              double hx = 1.0, double hy = 1.0);

#endif
