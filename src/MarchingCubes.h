#ifndef MARCHING_CUBES_H
#define MARCHING_CUBES_H

#include "TSDFVolume.h"
#include <vector>

void extract_surface(TSDFVolume & volume, std::vector<float3>& vertices, std::vector<int3>& triangles);


#endif
