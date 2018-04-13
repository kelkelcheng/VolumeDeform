#include "TSDFVolume.h"
#include <iostream>
#include <string>
#include "cudaUtil.h"
#include <math.h>
//using namespace std;

__device__
float3 bilinear( float3 c00, float3 c01, float3 c10, float3 c11){
    float3  a = c00 * 0.5 + c10 * 0.5; 
    float3  b = c01 * 0.5 + c11 * 0.5; 
    return (a * 0.5 + b * 0.5); 
}

__device__
float3 trilinear(float3 c000, float3 c001, float3 c010, float3 c011, float3 c100, float3 c101, float3 c110, float3 c111){
	float3 c0, c1;
	c0 = bilinear(c000, c100, c010, c110);
	c1 = bilinear(c001, c101, c011, c111);
	return (c0 * 0.5 + c1 * 0.5);
}

__device__
int3 index_1to3(int idx, dim3 grid_size) {
	int3 r_idx;
	int xy = (grid_size.x * grid_size.y);
	r_idx.z = idx / xy;
	int remainder = idx % xy;
	
	r_idx.y = remainder / grid_size.x;
	r_idx.x = remainder % grid_size.x;
	return r_idx;
}

/*__device__
int3 index_1to3(int idx, dim3 grid_size) {
	int3 r_idx;
	int yz = (grid_size.y * grid_size.z);
	r_idx.x = idx / yz;
	int remainder = idx % yz;
	
	r_idx.y = remainder / grid_size.z;
	r_idx.z = remainder % grid_size.z;
	return r_idx;
}*/

/*__global__
void initialize_grid(float3 * grid, dim3 grid_size, float voxel_size, float3 grid_origin ){
	int vy = threadIdx.x;
	int vz = blockIdx.x;

	// If this thread is in range
    if ( vy < grid_size.y+1 && vz < grid_size.z+1 ) {

        // The next (x_size) elements from here are the x coords
        size_t base_grid_index = (grid_size.x+1) * (grid_size.y+1) * vz + (grid_size.x+1) * vy;

        size_t grid_index = base_grid_index;
        for ( int vx = 0; vx < grid_size.x+1; vx++ ) {
            grid[grid_index].x = (float)vx * voxel_size + grid_origin.x;
            grid[grid_index].y = (float)vy * voxel_size + grid_origin.y;
            grid[grid_index].z = (float)vz * voxel_size + grid_origin.z;

            grid_index++;
        }
    }
}*/

/*__global__
void initialize_grid(float3 * grid, dim3 grid_size, float voxel_size, float3 grid_origin ){
	int vy = threadIdx.x;
	int vz = blockIdx.x;

	// If this thread is in range
    if ( vy < grid_size.y && vz < grid_size.z ) {

        // The next (x_size) elements from here are the x coords
        size_t base_grid_index = (grid_size.x) * (grid_size.y) * vz + (grid_size.x) * vy;

        size_t grid_index = base_grid_index;
        for ( int vx = 0; vx < grid_size.x; vx++ ) {
            grid[grid_index].x = (float)vx * voxel_size + grid_origin.x;
            grid[grid_index].y = (float)vy * voxel_size + grid_origin.y;
            grid[grid_index].z = (float)vz * voxel_size + grid_origin.z;

            grid_index++;
        }
    }
}*/

__global__
void initialize_grid(float3 * grid, dim3 grid_size, float3 voxel_size, float3 grid_origin ){
	// index 1D to 3D	
	int grid_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (grid_index < grid_size.x * grid_size.y * grid_size.z) {
		int3 idx = index_1to3(grid_index, grid_size);
        grid[grid_index].x = (float)idx.x * voxel_size.x + grid_origin.x;
        grid[grid_index].y = (float)idx.y * voxel_size.y + grid_origin.y;
        grid[grid_index].z = (float)idx.z * voxel_size.z + grid_origin.z;
	}
}

/*
__global__
void deformation( float3* grid, float3 * deformation, dim3 grid_size ) {

    // Extract the voxel Y and Z coordinates we then iterate over X
    int vy = threadIdx.x;
    int vz = blockIdx.x;
	float3 c000, c001,c010,c011,c100,c101,c110,c111;
	size_t layer_size =  (grid_size.x + 1) * (grid_size.y + 1);
    // If this thread is in range
    if ( vy < grid_size.y && vz < grid_size.z ) {

        // The next (x_size) elements from here are the x coords
		size_t base_grid_index = (grid_size.x+1) * (grid_size.y+1) * vz + (grid_size.x+1) * vy;
        size_t base_voxel_index =  ((grid_size.x * grid_size.y) * vz ) + (grid_size.x * vy);

        size_t voxel_index = base_voxel_index;
		size_t grid_index = base_grid_index;
        for ( int vx = 0; vx < grid_size.x; vx++ ) {
			c000 = grid[grid_index + grid_size.x +1];
			c001 = grid[grid_index];
			c010 = grid[grid_index + grid_size.x +1 + layer_size];
			c011 = grid[grid_index + layer_size];
			c100 = grid[grid_index + grid_size.x +2];
			c101 = grid[grid_index + 1];
			c110 = grid[grid_index + grid_size.x + 2 + layer_size];
			c111 = grid[grid_index + layer_size + 1];
            
			deformation[voxel_index] = trilinear(c000, c001,c010,c011,c100,c101,c110,c111);


            voxel_index++;
			grid_index++;
        }
    }
}*/

__host__
TSDFVolume::TSDFVolume(int x, int y, int z, float3 ori, float3 size){
		max_threads = 512;
		x+=1; y+=1; z+=1;

		m_size.x = x;
		m_size.y = y;
		m_size.z = z;

		origin = ori;
		/*origin.x = -1.5f;
		origin.y = -1.5f;
		origin.z = 0.5f;*/

		voxel_size = size;
		trunc_margin = voxel_size.x * 5; //5;

		int xyz = x*y*z;
		size_t data_size = xyz * sizeof( float );

        cudaSafeCall(cudaMalloc( &m_distances, data_size ));

		float * voxel_grid_TSDF = new float[xyz];
		for(int i = 0; i< xyz;i++)
			voxel_grid_TSDF[i] = 99.0f; //1.0f; 
		cudaMemcpy(m_distances, voxel_grid_TSDF, data_size, cudaMemcpyHostToDevice);
		delete voxel_grid_TSDF;

        cudaSafeCall(cudaMalloc( &m_weights, data_size ));
		cudaMemset(m_weights,0,data_size);

		/*cudaSafeCall(cudaMalloc(&grid_coord,(x+1) * (y+1) * (z+1) * sizeof(float3)));
		initialize_grid<<< (x+1), (y+1) >>>(grid_coord, m_size, voxel_size, origin);
		cudaDeviceSynchronize( );*/
 
		cudaSafeCall(cudaMalloc(&grid_coord, xyz * sizeof(float3)));
		initialize_grid<<< ceil(xyz / (float)max_threads), max_threads >>>(grid_coord, m_size, voxel_size, origin);
		cudaDeviceSynchronize( );

		/*cudaSafeCall(cudaMalloc(&m_deform, x * y * z * sizeof( float3 )));
		deformation<<< x, y >>>(grid_coord, m_deform, m_size);
		cudaDeviceSynchronize( );*/
	}

TSDFVolume::~TSDFVolume() {
    std::cout << "Destroying TSDFVolume" << std::endl;
    deallocate( );
}


/**
 * Deallocate storage for this TSDF
 */
void TSDFVolume::deallocate( ) {
    // Remove existing data
    if ( m_distances ) {
        cudaFree( m_distances );
        m_distances = 0;
    }
    if ( m_weights ) {
        cudaFree( m_weights );
        m_weights = 0;
    }
    /*if ( m_deform ) {
        cudaFree( m_deform );
        m_deform = 0;
    }*/
    if ( grid_coord ) {
        cudaFree( grid_coord );
        grid_coord = 0;
    }

}


__global__
void Integrate_kernal(float * cam_K, float * cam2base, float * depth_im,
               dim3 size, float3 origin, float3 voxel_size, float trunc_margin,
               float * voxel_grid_TSDF, float * voxel_grid_weight, float3* grid_c) {

	int volume_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (volume_idx < size.x * size.y * size.z){
		// Convert voxel center from grid coordinates to base frame camera coordinates
		float pt_base_x = grid_c[volume_idx].x;
		float pt_base_y = grid_c[volume_idx].y;
		float pt_base_z = grid_c[volume_idx].z;

		// Convert from base frame camera coordinates to current frame camera coordinates
		float tmp_pt[3] = {0};
		tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
		tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
		tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
		float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
		float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
		float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

		if (pt_cam_z <= 0) //0
			return;

		int pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
		int pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
		if (pt_pix_x < 0 || pt_pix_x >= 640 || pt_pix_y < 0 || pt_pix_y >= 480)
			return;

		float depth_val = depth_im[pt_pix_y * 640 + pt_pix_x];

		if (depth_val <= 0 || depth_val > 1) //0, 6
			return;

		float diff = depth_val - pt_cam_z;

		if (diff <= -trunc_margin) {
			voxel_grid_TSDF[volume_idx] = -1.0f; //voxel_grid_TSDF[volume_idx] = 99.0f; //
			voxel_grid_weight[volume_idx] += 1.0f;
			return;
		}
			
		if ( voxel_grid_TSDF[volume_idx] >= 90.0f )	{ //90.0f
			voxel_grid_TSDF[volume_idx] = fmin(1.0f, diff / trunc_margin); //fmin(diff, trunc_margin);
			voxel_grid_weight[volume_idx] = 1.0f;
			return;
		}

		// Integrate

		float dist = fmin(1.0f, diff / trunc_margin); //fmin(diff, trunc_margin);
		float weight_old = voxel_grid_weight[volume_idx];
		float weight_new = weight_old + 1.0f;
		voxel_grid_weight[volume_idx] = weight_new;
		voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
	}
}

__host__
void TSDFVolume::Integrate(float* depth_map,float* cam_K, float* cam2base){
	float * gpu_cam_K;
	float * gpu_cam2base;
	float * gpu_depth_im;

	cudaMalloc(&gpu_depth_im, 480 * 640 * sizeof(float));
	cudaMemcpy(gpu_depth_im, depth_map, 480 * 640 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_cam_K, 3 * 3 * sizeof(float));
	cudaMemcpy(gpu_cam_K, cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_cam2base, 4 * 4 * sizeof(float));
	cudaMemcpy(gpu_cam2base, cam2base, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);

	int blocknum = ceil(m_size.x * m_size.y * m_size.z / (float)max_threads);
	Integrate_kernal<<< blocknum,max_threads >>>(gpu_cam_K, gpu_cam2base, gpu_depth_im, m_size, origin, voxel_size, trunc_margin,m_distances, m_weights, grid_coord); //m_deform
	cudaDeviceSynchronize( );

	cudaSafeCall(cudaFree(gpu_cam_K));
	cudaSafeCall(cudaFree(gpu_cam2base));
	cudaSafeCall(cudaFree(gpu_depth_im));
}
 
__host__
void TSDFVolume::InitSubGrid(std::vector<float3>& sg_pos, int3 sg_dims){
	float3 * temp_grid = new float3[m_size.x * m_size.y * m_size.z];
	cudaMemcpy(temp_grid, grid_coord, m_size.x * m_size.y * m_size.z * sizeof(float3), cudaMemcpyDeviceToHost);
	int3 grid_scale;
	int sg_index, volume_index;
	grid_scale.x = 15; //m_size.x / sg_dims.x;
	grid_scale.y = 15; //m_size.y / sg_dims.y;
	grid_scale.z = 20; //m_size.z / sg_dims.z;
	std::cout<<"test scale "<< grid_scale.x<<" "<<grid_scale.y<<" "<<grid_scale.z<<" "<<std::endl;
	for (int i = 0; i < sg_dims.x; i++) {
		for (int j = 0; j < sg_dims.y; j++) {
			for (int k = 0; k < sg_dims.z; k++){	
				sg_index = k * (sg_dims.y * sg_dims.x) + j * sg_dims.x + i;
				volume_index = k*grid_scale.z * (m_size.y * m_size.x) + j*grid_scale.y * m_size.x + i*grid_scale.x;
				//std::cout << "test " << i<<" "<<j<<" "<<k <<" "<<tmp.x<<" "<<tmp.y<<" "<<tmp.z<<std::endl;
				sg_pos[sg_index] = temp_grid[volume_index];
			}
		}
	}

	delete temp_grid;
}

int index_3to1(int3 idx, int3 grid_dims)
{
	return idx.z * (grid_dims.y * grid_dims.x) + idx.y * grid_dims.x + idx.x;
}

__host__
void TSDFVolume::Upsample(std::vector<float3>& sg_pos, int3 sg_dims){
	float3 * temp_grid = new float3[m_size.x * m_size.y * m_size.z];
	cudaMemcpy(temp_grid, grid_coord, m_size.x * m_size.y * m_size.z * sizeof(float3), cudaMemcpyDeviceToHost);

	int3 grid_scale;
	grid_scale.x = 15; //m_size.x / sg_dims.x;
	grid_scale.y = 15; //m_size.y / sg_dims.y;
	grid_scale.z = 20; //m_size.z / sg_dims.z;
	std::cout<<"test scale "<< grid_scale.x<<" "<<grid_scale.y<<" "<<grid_scale.z<<" "<<std::endl;
	for (int i = 0; i < m_size.x; i++) {
		for (int j = 0; j < m_size.y; j++) {
			for (int k = 0; k < m_size.z; k++){	

				float x = i/(float)grid_scale.x;
				float y = j/(float)grid_scale.y;
				float z = k/(float)grid_scale.z;
				int x0 = floor(x); int x1 = ceil(x);
				int y0 = floor(y); int y1 = ceil(y);
				int z0 = floor(z); int z1 = ceil(z);

				x = x - x0;
				y = y - y0;
				z = z - z0;
				float3 p000 = sg_pos[index_3to1(make_int3(x0, y0, z0), sg_dims)];
				float3 p001 = sg_pos[index_3to1(make_int3(x0, y0, z1), sg_dims)];
				float3 p010 = sg_pos[index_3to1(make_int3(x0, y1, z0), sg_dims)];
				float3 p011 = sg_pos[index_3to1(make_int3(x0, y1, z1), sg_dims)];
				float3 p100 = sg_pos[index_3to1(make_int3(x1, y0, z0), sg_dims)];
				float3 p101 = sg_pos[index_3to1(make_int3(x1, y0, z1), sg_dims)];
				float3 p110 = sg_pos[index_3to1(make_int3(x1, y1, z0), sg_dims)];
				float3 p111 = sg_pos[index_3to1(make_int3(x1, y1, z1), sg_dims)];

				float3 px00 = (1.0f - x)*p000 + x*p100;
				float3 px01 = (1.0f - x)*p001 + x*p101;
				float3 px10 = (1.0f - x)*p010 + x*p110;
				float3 px11 = (1.0f - x)*p011 + x*p111;

				float3 pxx0 = (1.0f - y)*px00 + y*px10;
				float3 pxx1 = (1.0f - y)*px01 + y*px11;

				float3 p = (1.0f - z)*pxx0 + z*pxx1;

				int3 g_dims;
				g_dims.x = m_size.x; g_dims.y = m_size.y; g_dims.z = m_size.z;
				temp_grid[index_3to1(make_int3(i,j,k), g_dims)] = p;

				//std::cout << "test " << i<<" "<<j<<" "<<k <<" "<<tmp.x<<" "<<tmp.y<<" "<<tmp.z<<std::endl;
			}
		}
	}
	cudaMemcpy(grid_coord, temp_grid, m_size.x * m_size.y * m_size.z * sizeof(float3), cudaMemcpyHostToDevice);
	delete temp_grid;
}
