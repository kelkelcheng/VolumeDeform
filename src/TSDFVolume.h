#ifndef TSDFVolume_h
#define TSDFVolume_h

#include <iostream>
#include <string>
#include <vector>

class TSDFVolume{
public:
	TSDFVolume(int x, int y, int z, float3 ori, float3 size);

	dim3 get_size(){
		return m_size;
	}

	~TSDFVolume();

	void deallocate();

	void Integrate(float* depth_map,float* cam_K, float* cam2base);

	float* get_grid(){
		return m_distances;
	}

	float3 get_origin(){
		return origin;
	}
	
	float3 get_voxelsize(){
		return voxel_size;
	}

	float3* get_deform(){
		return grid_coord;

	}

	void InitSubGrid(std::vector<float3>& sg_pos, int3 sg_dims);
	void Upsample(std::vector<float3>& sg_pos, int3 sg_dims);
	/*
	float3* get_deform(){
		return m_deform;
	}*/

private:
	float3 origin;

	dim3 m_size;

	float3 voxel_size;
	float trunc_margin;
	// Per grid point data
    float *m_distances;
    //  Confidence weight for distance and colour
    float *m_weights;
	// translation vector for each node
	//float3 *m_deform;
	float3 *grid_coord;

	int max_threads;
};
#endif
