#ifndef TSDFVolume_h
#define TSDFVolume_h

#include <iostream>
#include <string>
#include <vector>

class TSDFVolume{
public:
	TSDFVolume(int x, int y, int z, float3 ori, float3 size, int3 sg_scale, int H, int W, bool is_perspective);

	dim3 get_size(){
		return m_size;
	}

	~TSDFVolume();

	void deallocate();

	void Integrate(float* depth_map,float* cam_K, float* cam2base, bool start);

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

	float3* get_coord_ori(){
		return grid_coord_ori;
	}

	unsigned int* get_state(){
		return m_state;
	}
	
	int3 get_scale(){
		return m_sg_scale;
	}
	
	void Reset();
	void InitSubGrid(std::vector<float3>& sg_pos, int3 sg_dims);
	void Upsample(std::vector<float3>& sg_pos);

private:
	float3 origin;
	int3 m_sg_scale;
	dim3 sg_size;
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

	float3 *grid_coord_ori;
	
	// state for each grid point
	unsigned int* m_state;

	int max_threads;

	int im_h;
	int im_w;
	bool m_is_perspective;
};
#endif
