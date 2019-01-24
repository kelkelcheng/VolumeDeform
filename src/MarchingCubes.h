#ifndef MARCHING_CUBES_H
#define MARCHING_CUBES_H

#include "TSDFVolume.h"
#include <vector>

void extract_surface(TSDFVolume & volume, std::vector<float3>& vertices, 
                    std::vector<int3>& triangles, std::vector<float3>& normals, std::vector<int3>& vol_idx, std::vector<float3>& rel_coors,
                    bool if_current=true);


#endif
