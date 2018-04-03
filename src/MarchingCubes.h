#ifndef MARCHING_CUBES_H
#define MARCHING_CUBES_H

#include "TSDFVolumn.h"
#include <vector>

void extract_surface(TSDFVolumn & volumn, std::vector<float3>& vertices, std::vector<int3>& triangles);


#endif
