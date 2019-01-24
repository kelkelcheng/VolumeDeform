#include "TSDFVolume.h"
#include <iostream>
#include <string>
#include "cudaUtil.h"
#include <math.h>
#include <stdio.h>
//using namespace std;

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

__device__
int sg_index_3to1(int3 idx, dim3 grid_size)
{
	return idx.z * (grid_size.y * grid_size.x) + idx.y * grid_size.x + idx.x;
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

__device__
bool if_valid(int3 idx, dim3 size, unsigned int* state) {
	if ((idx.x >= 0) && (idx.x < size.x) && 
		(idx.y >= 0) && (idx.y < size.y) && 
		(idx.z >= 0) && (idx.z < size.z)) {
		return (state[sg_index_3to1(idx, size)] == 2);
	}
	return 0;
}

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

__host__
TSDFVolume::TSDFVolume(int x, int y, int z, float3 ori, float3 size, int3 sg_scale, int H, int W, bool is_perspective){
		max_threads = 512;

		im_h = H;
		im_w = W;
		m_is_perspective = is_perspective;

		m_sg_scale = sg_scale;
		
		m_size.x = x;
		m_size.y = y;
		m_size.z = z;

		sg_size.x = (m_size.x - 1) / m_sg_scale.x + 1;
		sg_size.y = (m_size.y - 1) / m_sg_scale.y + 1;
		sg_size.z = (m_size.z - 1) / m_sg_scale.z + 1;

		origin = ori;

		voxel_size = size;
		trunc_margin = voxel_size.x * 5;//0.2;// * 5; //5;

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
 
		cudaSafeCall(cudaMalloc(&grid_coord, xyz * sizeof(float3)));
		initialize_grid<<< ceil(xyz / (float)max_threads), max_threads >>>(grid_coord, m_size, voxel_size, origin);
		cudaDeviceSynchronize( );

		cudaSafeCall(cudaMalloc(&grid_coord_ori, xyz * sizeof(float3)));
		initialize_grid<<< ceil(xyz / (float)max_threads), max_threads >>>(grid_coord_ori, m_size, voxel_size, origin);
		cudaDeviceSynchronize( );

		cudaSafeCall(cudaMalloc(&m_state, xyz * sizeof(unsigned int)));
		cudaMemset(m_state, 0, xyz * sizeof(unsigned int));
		cudaDeviceSynchronize( );

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
    if ( grid_coord ) {
        cudaFree( grid_coord );
        grid_coord = 0;
    }
    if ( grid_coord_ori ) {
        cudaFree( grid_coord_ori );
        grid_coord_ori = 0;
    }
	if ( m_state ) {
		cudaFree(m_state);
		m_state = 0;
	}
}

__global__
void Update_state(unsigned int * state, dim3 size){
	int volume_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (volume_idx < size.x * size.y * size.z && volume_idx >= 0){
		if (state[volume_idx]==3) {state[volume_idx]=2;}
	}
}

__global__
void Integrate_kernal(float * cam_K, float * cam2base, float * depth_im,
               dim3 size, float3 origin, float3 voxel_size, float trunc_margin,
               float * voxel_grid_TSDF, float * voxel_grid_weight, float3* grid_c, unsigned int* state, bool check_start,
			   int H, int W, bool is_perspective=true) {

	int volume_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (volume_idx < size.x * size.y * size.z){
		// Convert voxel center from grid coordinates to base frame camera coordinates
		float pt_base_x = grid_c[volume_idx].x;
		float pt_base_y = grid_c[volume_idx].y;
		float pt_base_z = grid_c[volume_idx].z;

		//if (volume_idx < 10) {printf("hello0 volume idx: %d, x: %f, y: %f, z_2: %f \n", volume_idx, pt_base_x, pt_base_y, pt_base_z);}

		// Convert from base frame camera coordinates to current frame camera coordinates
		float tmp_pt[3] = {0};
		tmp_pt[0] = pt_base_x - cam2base[0 * 4 + 3];
		tmp_pt[1] = pt_base_y - cam2base[1 * 4 + 3];
		tmp_pt[2] = pt_base_z - cam2base[2 * 4 + 3];
		float pt_cam_x = cam2base[0 * 4 + 0] * tmp_pt[0] + cam2base[1 * 4 + 0] * tmp_pt[1] + cam2base[2 * 4 + 0] * tmp_pt[2];
		float pt_cam_y = cam2base[0 * 4 + 1] * tmp_pt[0] + cam2base[1 * 4 + 1] * tmp_pt[1] + cam2base[2 * 4 + 1] * tmp_pt[2];
		float pt_cam_z = cam2base[0 * 4 + 2] * tmp_pt[0] + cam2base[1 * 4 + 2] * tmp_pt[1] + cam2base[2 * 4 + 2] * tmp_pt[2];

		//if (volume_idx < 10) {printf("hello1 volume idx: %d, x: %f, y: %f, z_2: %f \n", volume_idx, pt_cam_x, pt_cam_y, pt_cam_z);}

		if (pt_cam_z <= 0.01f) //0
			return;

		//TODO: add orthographic projection
		int pt_pix_x, pt_pix_y;
		if (is_perspective) {
			pt_pix_x = roundf(cam_K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + cam_K[0 * 3 + 2]);
			pt_pix_y = roundf(cam_K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + cam_K[1 * 3 + 2]);
		}
		else {
			pt_pix_x = roundf(cam_K[0 * 3 + 0] * pt_cam_x + cam_K[0 * 3 + 2]);
			pt_pix_y = roundf(cam_K[1 * 3 + 1] * pt_cam_y + cam_K[1 * 3 + 2]);
		}

		//if (volume_idx < 10) {printf("hello2 volume idx: %d, x: %d, y: %d, z_2: %f \n", volume_idx, pt_pix_x, pt_pix_y, pt_cam_z);}

		if (pt_pix_x < 0 || pt_pix_x >= W || pt_pix_y < 0 || pt_pix_y >= H)
			return;
		
		//if (volume_idx > 750000 && volume_idx < 800000) {printf("hello2 volume idx: %d, x: %d, y: %d, z_2: %f \n", volume_idx, pt_pix_x, pt_pix_y, pt_cam_z);}
		//if (volume_idx < 10) {printf("hello2 volume idx: %d, x: %d, y: %d, z_2: %f \n", volume_idx, pt_pix_x, pt_pix_y, pt_cam_z);}

		float depth_val = depth_im[pt_pix_y * W + pt_pix_x];

		if (depth_val <= 0 || depth_val > 1.2) //0, 6
			return;

		//if (volume_idx < 1000000) {printf("hello3 volume idx: %d, x: %d, y: %d, z: %f, dep: %f \n", volume_idx, pt_pix_x, pt_pix_y, pt_cam_z, depth_val);}
		//if (volume_idx < 10) {printf("hello3 volume idx: %d, x: %d, y: %d, z: %f, dep: %f \n", volume_idx, pt_pix_x, pt_pix_y, pt_cam_z, depth_val);}

		float diff = depth_val - pt_cam_z;

		/*if (diff <= -trunc_margin) {
			voxel_grid_TSDF[volume_idx] = -1.0f; //voxel_grid_TSDF[volume_idx] = 99.0f; //
			voxel_grid_weight[volume_idx] += 1.0f;
			return;
		}*/
		diff = fmax(diff, -trunc_margin); // new
		
		//if (volume_idx < 10) {printf("hello3 volume idx: %d, x: %d, y: %d, z: %d, z_2: %d \n", volume_idx, pt_pix_x, pt_pix_y, depth_val, pt_cam_z);}
		
		// update one-ring only
		if ( voxel_grid_TSDF[volume_idx] >= 90.0f )	{ //90.0f

			//TODO !! remove later... use to disable adding new grid points
			if (!check_start) {
				return;
			}		

			int3 v_idx3 = index_1to3(volume_idx, size);
			bool check_iso = 0;
			//int num_grid = size.x * size.y * size.z;
			
			int3 v_idx3_temp = v_idx3; 
			v_idx3_temp.x = v_idx3.x + 1; 
			int v_idx1 = sg_index_3to1(v_idx3_temp, size);
			check_iso = check_iso || if_valid(v_idx3_temp, size, state);

			v_idx3_temp = v_idx3; 
			v_idx3_temp.x = v_idx3.x - 1; 
			v_idx1 = sg_index_3to1(v_idx3_temp, size);
			check_iso = check_iso || if_valid(v_idx3_temp, size, state);
			
			v_idx3_temp = v_idx3; 
			v_idx3_temp.y = v_idx3.y + 1; 
			v_idx1 = sg_index_3to1(v_idx3_temp, size);
			check_iso = check_iso || if_valid(v_idx3_temp, size, state);

			v_idx3_temp = v_idx3; 
			v_idx3_temp.y = v_idx3.y - 1; 
			v_idx1 = sg_index_3to1(v_idx3_temp, size);
			check_iso = check_iso || if_valid(v_idx3_temp, size, state);

			v_idx3_temp = v_idx3; 
			v_idx3_temp.z = v_idx3.z + 1; 
			v_idx1 = sg_index_3to1(v_idx3_temp, size);
			check_iso = check_iso || if_valid(v_idx3_temp, size, state);

			v_idx3_temp = v_idx3; 
			v_idx3_temp.z = v_idx3.z - 1; 
			check_iso = check_iso || if_valid(v_idx3_temp, size, state);
			
			if (check_iso && (!check_start)) {
				float dist = fmin(1.0f, diff / trunc_margin); //fmin(diff, trunc_margin);	
				if (state[volume_idx] == 0) {												 				
					state[volume_idx] = 3;//1;
				} else {				
					//dist = (voxel_grid_TSDF[volume_idx] * voxel_grid_weight[volume_idx] + dist) / (voxel_grid_weight[volume_idx] + 1.0f);
					//state[volume_idx] += 1;
					state[volume_idx] = 3;
				}
				voxel_grid_weight[volume_idx] += 1.0f;
				voxel_grid_TSDF[volume_idx] = dist;
				return;
			} 
			
			if (!check_start) {
				return;
			}		
		}

		// Integrate
		// update only voxel used in optimization
		/*if (state[volume_idx]!=2 && !check_start) {
			return;
		}*/
		
		float dist = fmin(1.0f, diff / trunc_margin); //fmin(diff, trunc_margin);
		float weight_old = voxel_grid_weight[volume_idx];
		float weight_new = weight_old + 1.0f;
		if (weight_new < 30.0f) { //max weight
			voxel_grid_weight[volume_idx] = weight_new;
		}
		voxel_grid_TSDF[volume_idx] = (voxel_grid_TSDF[volume_idx] * weight_old + dist) / weight_new;
		state[volume_idx] = 3; //0 to enable integration control, 2 to disregard, use 3
	}
}

__host__
void TSDFVolume::Integrate(float* depth_map,float* cam_K, float* cam2base, bool check_start){
	float * gpu_cam_K;
	float * gpu_cam2base;
	float * gpu_depth_im;

	cudaMalloc(&gpu_depth_im, im_h * im_w * sizeof(float));
	cudaMemcpy(gpu_depth_im, depth_map, im_h * im_w * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_cam_K, 3 * 3 * sizeof(float));
	cudaMemcpy(gpu_cam_K, cam_K, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc(&gpu_cam2base, 4 * 4 * sizeof(float));
	cudaMemcpy(gpu_cam2base, cam2base, 4 * 4 * sizeof(float), cudaMemcpyHostToDevice);

	int blocknum = ceil(m_size.x * m_size.y * m_size.z / (float)max_threads);
	Integrate_kernal<<< blocknum,max_threads >>>(gpu_cam_K, gpu_cam2base, gpu_depth_im, m_size, origin, voxel_size, trunc_margin,m_distances, m_weights, grid_coord, m_state, check_start, im_h, im_w, m_is_perspective); 
	cudaDeviceSynchronize( );

	Update_state<<< blocknum,max_threads >>>(m_state, m_size);
	cudaDeviceSynchronize( );
	
	cudaSafeCall(cudaFree(gpu_cam_K));
	cudaSafeCall(cudaFree(gpu_cam2base));
	cudaSafeCall(cudaFree(gpu_depth_im));
}

// initialize sparse subgrid
__host__
void TSDFVolume::InitSubGrid(std::vector<float3>& sg_pos, int3 sg_dims){
	float3 * temp_grid = new float3[m_size.x * m_size.y * m_size.z];
	cudaMemcpy(temp_grid, grid_coord, m_size.x * m_size.y * m_size.z * sizeof(float3), cudaMemcpyDeviceToHost);
	int3 grid_scale = m_sg_scale;
	int sg_index, volume_index;

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

/*int index_3to1(int3 idx, int3 grid_dims)
{
	return idx.z * (grid_dims.y * grid_dims.x) + idx.y * grid_dims.x + idx.x;
}*/

// upsampling sparse subgrid to update the whole grid
__global__
void upsample(float3 * grid_coor, float3 * sg_pos, int3 grid_scale, dim3 size, dim3 sg_size) {
	int grid_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (grid_index < size.x * size.y * size.z) {
		int3 idx = index_1to3(grid_index, size);
		
		float x = idx.x/(float)grid_scale.x;
		float y = idx.y/(float)grid_scale.y;
		float z = idx.z/(float)grid_scale.z;
		int x0 = floor(x); int x1 = ceil(x);
		int y0 = floor(y); int y1 = ceil(y);
		int z0 = floor(z); int z1 = ceil(z);

		x = x - x0;
		y = y - y0;
		z = z - z0;
		float3 p000 = sg_pos[sg_index_3to1(make_int3(x0, y0, z0), sg_size)];
		float3 p001 = sg_pos[sg_index_3to1(make_int3(x0, y0, z1), sg_size)];
		float3 p010 = sg_pos[sg_index_3to1(make_int3(x0, y1, z0), sg_size)];
		float3 p011 = sg_pos[sg_index_3to1(make_int3(x0, y1, z1), sg_size)];
		float3 p100 = sg_pos[sg_index_3to1(make_int3(x1, y0, z0), sg_size)];
		float3 p101 = sg_pos[sg_index_3to1(make_int3(x1, y0, z1), sg_size)];
		float3 p110 = sg_pos[sg_index_3to1(make_int3(x1, y1, z0), sg_size)];
		float3 p111 = sg_pos[sg_index_3to1(make_int3(x1, y1, z1), sg_size)];

		float3 px00 = (1.0f - x)*p000 + x*p100;
		float3 px01 = (1.0f - x)*p001 + x*p101;
		float3 px10 = (1.0f - x)*p010 + x*p110;
		float3 px11 = (1.0f - x)*p011 + x*p111;

		float3 pxx0 = (1.0f - y)*px00 + y*px10;
		float3 pxx1 = (1.0f - y)*px01 + y*px11;

		float3 p = (1.0f - z)*pxx0 + z*pxx1;

		grid_coor[sg_index_3to1(make_int3(idx.x, idx.y, idx.z), size)] = p;		
	}	
}

__host__
void TSDFVolume::Upsample(std::vector<float3>& sg_pos){
	// pass subgrid positions into gpu
	float3 * sg_pos_gpu;
	cudaSafeCall( cudaMalloc(&sg_pos_gpu, sg_pos.size() * sizeof(float3)) );
	cudaSafeCall( cudaMemcpy(sg_pos_gpu, sg_pos.data(), sg_pos.size() * sizeof(float3), cudaMemcpyHostToDevice) );
	
	int blocknum = ceil(m_size.x*m_size.y*m_size.z / (float)max_threads);
	upsample<<< blocknum, max_threads >>>(grid_coord, sg_pos_gpu, m_sg_scale, m_size, sg_size);
	cudaSafeCall( cudaDeviceSynchronize() );
	cudaSafeCall( cudaFree(sg_pos_gpu) );
}
