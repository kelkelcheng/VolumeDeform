#include <iostream>
#include "MarchingCubes.h"
#include "GPUTiktok.h"
#include "MC_table.cu"
#include "cudaUtil.h"
#include <math.h>
#include <assert.h>

#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
using namespace std;

#define CUDART_NAN_F            __int_as_float(0x7fffffff)
// CUDA kernel function to integrate a TSDF voxel volume given depth images

inline __host__ __device__ uint2 operator+(uint2 a, uint2 b) {
	return make_uint2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ uint3 operator+(uint3 a, uint3 b) {
	return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

void ThrustScanWrapperUint3(uint3* output, uint3* input, unsigned int numElements) {
    const uint3 zero = make_uint3(0, 0, 0);
    thrust::exclusive_scan(thrust::device_ptr<uint3>(input),
                           thrust::device_ptr<uint3>(input + numElements),
                           thrust::device_ptr<uint3>(output),
                           zero);
}


__device__
float3 compute_rel_coor(int i, int x, int y, int z, int3 sc, float ratio) {
	switch(i) {
		case 0 :	//between v0 and v1					
			return make_float3((x%sc.x+ratio)/(float)sc.x, (y%sc.y)/(float)sc.y, (z%sc.z)/(float)sc.z);											
		case 1 : 	//between v2 and v1						
			return make_float3( (x%sc.x+1)/(float)sc.x, (y%sc.y)/(float)sc.y, (z%sc.z+ratio)/(float)sc.z);							
		case 2 :	//between v3 and v2					
			return make_float3( (x%sc.x+ratio)/(float)sc.x, (y%sc.y)/(float)sc.y, (z%sc.z+1)/(float)sc.z);						
		case 3 :	//between v3 and v0					
			return make_float3( (x%sc.x)/(float)sc.x, (y%sc.y)/(float)sc.y, (z%sc.z+ratio)/(float)sc.z);										
		case 4 : 	//between v4 and v5					
			return make_float3( (x%sc.x+ratio)/(float)sc.x, (y%sc.y+1)/(float)sc.y, (z%sc.z)/(float)sc.z);													
		case 5 :  	//between v6 and v5				
			return make_float3( (x%sc.x+1)/(float)sc.x, (y%sc.y+1)/(float)sc.y, (z%sc.z+ratio)/(float)sc.z);								
		case 6 :	//between v7 and v6						
			return make_float3( (x%sc.x+ratio)/(float)sc.x, (y%sc.y+1)/(float)sc.y, (z%sc.z+1)/(float)sc.z);														
		case 7 :	//between v7 and v4					
			return make_float3( (x%sc.x)/(float)sc.x, (y%sc.y+1)/(float)sc.y, (z%sc.z+ratio)/(float)sc.z);						
		case 8 :	//between v0 and v4			
			return make_float3( (x%sc.x)/(float)sc.x, (y%sc.y+ratio)/(float)sc.y, (z%sc.z)/(float)sc.z);													
		case 9 :	//between v1 and v5					
			return make_float3( (x%sc.x+1)/(float)sc.x, (y%sc.y+ratio)/(float)sc.y, (z%sc.z)/(float)sc.z);												
		case 10:	//between v2 and v6	
			return make_float3( (x%sc.x+1)/(float)sc.x, (y%sc.y+ratio)/(float)sc.y, (z%sc.z+1)/(float)sc.z);							
		case 11:	//between v3 and v7				
			return make_float3( (x%sc.x)/(float)sc.x, (y%sc.y+ratio)/(float)sc.y, (z%sc.z+1)/(float)sc.z);						
	}
	return make_float3(0,0,0);
}
/**
 * @param cube_index the value descrbining which cube this is
 * @param voxel_values The distance values from the TSDF corresponding to the 8 voxels forming this cube
 * @param cube_vertices The vertex coordinates in real space of the cube being considered
 * @param intersects Populated by this function, the point on each of the 12 edges where an intersection occurs
 * There are a maximum of 12. Non-intersected edges are skipped that is, if only edge 12 is intersected then intersects[11]
 * will have a value the other values will be NaN
 * @return The number of intersects found
 */
__device__
void compute_edge_intersects( uint8_t cube_index,
                             const float voxel_values[8],
                             const float3 cube_vertices[8],
                             float3 intersects[12], float ratios[12]) {
	// Get the edges impacted
	//int num_edges_impacted = 0;
	if ( ( cube_index != 0x00) && ( cube_index != 0xFF ) ) {
		uint16_t intersected_edge_flags = EDGE_TABLE[cube_index];
		uint16_t mask = 0x01;
		int v1_index, v2_index;
		bool flip;
		float3 start_vertex, end_vertex, temp3, edge, delta;
		float start_weight, end_weight, temp, ratio;
		for ( int i = 0; i < 12; i++ ) {
			if ( ( intersected_edge_flags & mask ) > 0 ) {

				//intersects[i] = compute_intersection_for_edge( i, voxel_values, cube_vertices);
				v1_index = EDGE_VERTICES[i][0];
				v2_index = EDGE_VERTICES[i][1];

				start_vertex = cube_vertices[v1_index];
				start_weight = voxel_values[v1_index];

				end_vertex = cube_vertices[v2_index];
				end_weight = voxel_values[v2_index];

				flip = 0;
				if (  (start_weight > 0 ) &&  (end_weight <  0 ) ) {
					// Swap start and end
					flip = 1;
					temp3 = start_vertex;
					start_vertex = end_vertex;
					end_vertex = temp3;

					temp = start_weight;
					start_weight = end_weight;
					end_weight = temp;
				} 
				assert( ( start_weight * end_weight ) <= 0 );

				ratio = ( 0.0f- start_weight ) / ( end_weight - start_weight);

				// Work out where this lies
				edge = make_float3(end_vertex.x - start_vertex.x, end_vertex.y -start_vertex.y, end_vertex.z - start_vertex.z);
				delta = make_float3(ratio * edge.x, ratio * edge.y, ratio * edge.z);
				intersects[i] = make_float3(start_vertex.x + delta.x, start_vertex.y + delta.y, start_vertex.z + delta.z); 
				if (flip==1) {ratio = 1.0f - ratio;}
				ratios[i] = ratio;
									
				//num_edges_impacted++;
			} else {
				intersects[i].x = CUDART_NAN_F;
				intersects[i].y = CUDART_NAN_F;
				intersects[i].z = CUDART_NAN_F;
			}
			mask = mask << 1;
		}
	}
	//return num_edges_impacted;
}

/**
 * @param values An array of eight values from the TSDF indexed per the edge_table include file
 * return an 8 bit value representing the cube type (where a bit is set if the value in tha location
 * is negative i.e. behind the surface otherwise it's clear
 */
__device__
uint8_t cube_type_for_values( const float values[8] ) {
	uint8_t mask = 0x01;
	uint8_t cube_type = 0x00;
	for ( int i = 0; i < 8; i++ ) {
		if (values[i] < 0) {
			cube_type = cube_type | mask;
		}
		mask = mask << 1;
	
		if (values[i] >=10.0f) {
			return 0x00;
		}
	}
	return cube_type;
}


__device__
int3 index_1dto3d(int idx, dim3 grid_size) {
	int3 r_idx;
	int xy = (grid_size.x * grid_size.y);
	r_idx.z = idx / xy;
	int remainder = idx % xy;
	
	r_idx.y = remainder / grid_size.x;
	r_idx.x = remainder % grid_size.x;
	return r_idx;
}

__global__
void classify_voxel(const float * tsdf_dis, dim3 size,

                	// Output variables
				 	uint3 * voxel_verts,
                
                	// voxel state
                	unsigned int* state) {


	// Extract the voxel X and Y coordinates which describe the position in the layers
	// We use layer1 = z0, layer2 = z1
	int idx_1d = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx_1d < size.x * size.y * size.z) {
		int3 idx_3d = index_1dto3d(idx_1d, size);
		int vx = idx_3d.x;
		int vy = idx_3d.y;
		int vz = idx_3d.z;

		// If this thread is in range (we only go up to penultimate X and Y values)
		if ( ( vx < size.x - 1 ) && ( vy <size.y - 1 ) && ( vz <size.z - 1 ) ) {

			// Compute index of the voxel to address (used to index voxel data)
			//int voxel_index =  vz * (size.y * size.x) + vy * size.x + vx;
			int voxel_index = idx_1d;

			// Load voxel values for the cube
			float voxel_values[8] = {
				tsdf_dis[voxel_index],									//	vx,   	vy,   	vz
				tsdf_dis[voxel_index + 1],								//	vx + 1, vy,   	vz
				tsdf_dis[voxel_index + 1 + (size.y * size.x)],			//	vx + 1, vy, 	vz+1
				tsdf_dis[voxel_index + + (size.y * size.x)],			//	vx, 	vy , 	vz+1
				tsdf_dis[voxel_index + size.x],							//	vx, 	vy+1,	vz
				tsdf_dis[voxel_index + size.x+ 1],						//	vx+1, 	vy+1, 	vz
				tsdf_dis[voxel_index + size.x+ 1 + (size.y * size.x)],	//	vx+1, 	vy+1, 	vz+1
				tsdf_dis[voxel_index + size.x + (size.y * size.x)]		//	vx, 	vy+1, 	vz + 1
			};
		
			// Compute the cube type
			uint8_t cube_type = cube_type_for_values( voxel_values );
			//if (vz >= 58 && vx>= 358 && vy>=373) printf("hello x: %d, y: %d, z: %d\n", vx, vy, vz);
			if (state[idx_1d] != 2) {cube_type = 0x00;}
			unsigned int num_vert = (unsigned int)VERTICES_FOR_CUBE_TYPE[cube_type];
			int cube_index = vz * ((size.y - 1) * (size.x - 1)) + ((size.x - 1) * vy) + vx;
			voxel_verts[cube_index].x = num_vert;
			voxel_verts[cube_index].y = num_vert/3; 
			voxel_verts[cube_index].z = num_vert > 0;				
		}
	}
}

__global__
void compact_voxel(unsigned int * compactedVoxelArray, 
              const uint3 * voxelOccupied, 
              unsigned int lastVoxel, unsigned int numVoxels, 
              unsigned int numVoxelsm1) {
              
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
 
	if ((i < numVoxels) && ((i < numVoxelsm1) ? voxelOccupied[i].z < voxelOccupied[i+1].z : lastVoxel)) {
		compactedVoxelArray[ voxelOccupied[i].z ] = i;
	}
}

__global__
void mc_kernel( const float * tsdf_dis, const float3 * tsdf_pos, dim3 size, int3 sc,

                // Output variables
				float3 * vertices, int3 * triangles, float3 * normals, int3 * vol_idx, float3 * rel_coors,
                
                // voxel state
                unsigned int* state,
                
                // compacted voxel array and scanned voxel array
                unsigned int * comp_voxel, uint3 * voxel_verts, int active_voxels) {


	// Extract cube index and corresponding x,y,z
	int kernel_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (!(kernel_idx < active_voxels)) {return;}
	int cube_index = comp_voxel[blockIdx.x * blockDim.x + threadIdx.x];
	dim3 cube_size = size;
	cube_size.x--; cube_size.y--; cube_size.z--;
	if (cube_index < cube_size.x * cube_size.y * cube_size.z) {
		int3 idx_3d = index_1dto3d(cube_index, cube_size);
		int vx = idx_3d.x;
		int vy = idx_3d.y;
		int vz = idx_3d.z;

		// If this thread is in range (we only go up to penultimate X and Y values)
		if ( ( vx < size.x - 1 ) && ( vy <size.y - 1 ) && ( vz <size.z - 1 ) ) {

			// Compute index of the voxel to address (used to index voxel data)
			int voxel_index =  vz * (size.y * size.x) + vy * size.x + vx;

			// Compute cube index (ised to index output tris and verts)
			//int cube_index =      vz * ((size.y - 1) * (size.x - 1)) + ((size.x - 1) * vy) + vx;
			assert(cube_index == vz * ((size.y - 1) * (size.x - 1)) + ((size.x - 1) * vy) + vx);
			
			// Load voxel values for the cube
			float voxel_values[8] = {
				tsdf_dis[voxel_index],									//	vx,   	vy,   	vz
				tsdf_dis[voxel_index + 1],								//	vx + 1, vy,   	vz
				tsdf_dis[voxel_index + 1 + (size.y * size.x)],			//	vx + 1, vy, 	vz+1
				tsdf_dis[voxel_index + + (size.y * size.x)],			//	vx, 	vy , 	vz+1
				tsdf_dis[voxel_index + size.x],							//	vx, 	vy+1,	vz
				tsdf_dis[voxel_index + size.x+ 1],						//	vx+1, 	vy+1, 	vz
				tsdf_dis[voxel_index + size.x+ 1 + (size.y * size.x)],	//	vx+1, 	vy+1, 	vz+1
				tsdf_dis[voxel_index + size.x + (size.y * size.x)]		//	vx, 	vy+1, 	vz + 1
			};
		
			// Compute the cube type
			uint8_t cube_type = cube_type_for_values( voxel_values );
			//int3 idx = make_int3(vx, vy, vz);
			//int idx_1D = idx.z * (size.y * size.x) + idx.y * size.x + idx.x;
			if (state[voxel_index] != 2) {cube_type =0x00;}
		
			// If it's a non-trivial cube_type, process it
			if ( ( cube_type != 0 ) && ( cube_type != 0xFF ) ) {

				// Compute the coordinates of the vertices of the cube
				float3 cube_vertices[8] = {
					tsdf_pos[voxel_index],									//	vx,   	vy,   	vz
					tsdf_pos[voxel_index + 1],								//	vx+1, 	vy,   	vz
					tsdf_pos[voxel_index + 1 + (size.y * size.x)],			//	vx+1, 	vy, 	vz+1
					tsdf_pos[voxel_index + + (size.y * size.x)],			//	vx, 	vy, 	vz+1
					tsdf_pos[voxel_index + size.x],							//	vx, 	vy+1,	vz
					tsdf_pos[voxel_index + size.x+ 1],						//	vx+1, 	vy+1, 	vz
					tsdf_pos[voxel_index + size.x+ 1 + (size.y * size.x)],	//	vx+1, 	vy+1, 	vz+1
					tsdf_pos[voxel_index + size.x + (size.y * size.x)]		//	vx, 	vy+1, 	vz+1
				};
				//compute_cube_vertices(vx,vy,size.x, centers_layer_1, centers_layer_2, cube_vertices);

				// Compute intersects (up to 12 per cube)
				float3 	intersects[12]; float ratios[12];
				compute_edge_intersects( cube_type, voxel_values, cube_vertices, intersects, ratios);

				// Copy these back into the return vaue array at the appropriate point for this thread
				uint3 voxel_scan = voxel_verts[cube_index];
				float3 uVec, vVec, v_nrml;
				int i0, i1, i2;
				for (int i=0; i<VERTICES_FOR_CUBE_TYPE[cube_type]; i+=3) {
					i0 = TRIANGLE_TABLE[cube_type][i + 0];
					i1 = TRIANGLE_TABLE[cube_type][i + 1];
					i2 = TRIANGLE_TABLE[cube_type][i + 2];
					
					vertices[voxel_scan.x + i + 0] = intersects[i0];
					vertices[voxel_scan.x + i + 1] = intersects[i1];
					vertices[voxel_scan.x + i + 2] = intersects[i2];
					
					triangles[voxel_scan.y + i/3].x = voxel_scan.x + i + 0;
					triangles[voxel_scan.y + i/3].y = voxel_scan.x + i + 1;
					triangles[voxel_scan.y + i/3].z = voxel_scan.x + i + 2;
	
					// calculate per-vertex normal
					uVec = vertices[voxel_scan.x + i + 1] - vertices[voxel_scan.x + i + 0];
					vVec = vertices[voxel_scan.x + i + 2] - vertices[voxel_scan.x + i + 0];
					v_nrml = make_float3(uVec.y*vVec.z-uVec.z*vVec.y, uVec.z*vVec.x-uVec.x*vVec.z, uVec.x*vVec.y-uVec.y*vVec.x);

					if (v_nrml.z > 0) {
						v_nrml.x = -v_nrml.x; v_nrml.y = -v_nrml.y; v_nrml.z = -v_nrml.z;
					}
					
					v_nrml = normalize(v_nrml);		
					normals[voxel_scan.x + i + 0] = v_nrml;	
					normals[voxel_scan.x + i + 1] = v_nrml;
					normals[voxel_scan.x + i + 2] = v_nrml;	
					
					vol_idx[voxel_scan.x + i + 0] = idx_3d;
					vol_idx[voxel_scan.x + i + 1] = idx_3d;
					vol_idx[voxel_scan.x + i + 2] = idx_3d;
					
					rel_coors[voxel_scan.x + i + 0] = compute_rel_coor(i0, vx, vy, vz, sc, ratios[i0]);
					rel_coors[voxel_scan.x + i + 1] = compute_rel_coor(i1, vx, vy, vz, sc, ratios[i1]);
					rel_coors[voxel_scan.x + i + 2] = compute_rel_coor(i2, vx, vy, vz, sc, ratios[i2]);
				}
			}
		}
	}
}


__host__
void extract_surface(TSDFVolume & volume, vector<float3>& vertices, vector<int3>& triangles, vector<float3>& normals, vector<int3>& vol_idx, vector<float3>& rel_coors){
	int3 sc = make_int3(15, 15, 20); 
	// Allocate storage on device and locally
	// Fail if not possible
	dim3 size = volume.get_size();
	//int tmp = size.x; size.x = size.z; size.z = tmp;

	size_t num_cubes = (size.x - 1) * (size.y - 1) * (size.z - 1); //(size.x - 1) * (size.y - 1);

	
	util::GPUTiktok clock; 
	clock.tik();

	// Now iterate over each slice
	size_t layer_size =  size.x * size.y * size.z; //size.x * size.y
	const float* grid = volume.get_grid();
	const float3* centers = volume.get_deform();
	
	// new
	// temporary host vertices
	//float3* h_center = new float3 [layer_size];
	
	int max_threads = 512;
	int blocknum = ceil(layer_size / (float)max_threads);
	
	// Classify voxel
	// Device vertices numbers and if occupied
	uint3* d_voxel_verts; 
	cudaSafeCall( cudaMalloc( &d_voxel_verts, num_cubes * sizeof( uint3 ) ) );
	classify_voxel <<< blocknum,max_threads >>>(grid, size, d_voxel_verts, volume.get_state()); 
	cudaSafeCall( cudaDeviceSynchronize() );
	
	uint3 lastElement, lastScanElement;
	cudaMemcpy((void *) &lastElement, (void *)(d_voxel_verts + num_cubes-1), sizeof(uint3), cudaMemcpyDeviceToHost);
	std::cout << "lastElement: " <<lastElement.x<<" "<<lastElement.y<<" "<<lastElement.z<<std::endl;
	
	ThrustScanWrapperUint3(d_voxel_verts, d_voxel_verts, num_cubes);

	cudaMemcpy((void *) &lastScanElement, (void *)(d_voxel_verts + num_cubes-1), sizeof(uint3), cudaMemcpyDeviceToHost);
	std::cout << "lastScanElement: " <<lastScanElement.x<<" "<<lastScanElement.y<<" "<<lastScanElement.z<<std::endl;
	
	int active_voxels = lastElement.z + lastScanElement.z;
	int total_verts = lastElement.x + lastScanElement.x;
	int total_trigs = lastElement.y + lastScanElement.y;
	std::cout << "active_voxels: " <<active_voxels<<" total_verts "<<total_verts<<" total_trigs "<<total_trigs<<std::endl;
	
	uint* d_comp_voxel;
	cudaSafeCall( cudaMalloc( &d_comp_voxel, num_cubes * sizeof( uint ) ) );
	compact_voxel<<< blocknum, max_threads >>>(d_comp_voxel, d_voxel_verts, lastElement.y, num_cubes, num_cubes - 1);
	cudaSafeCall( cudaDeviceSynchronize() );
	
	float3* test_vertices;
	cudaSafeCall( cudaMalloc( &test_vertices, total_verts * sizeof( float3 ) ) );
	
	int3* test_triangles;
	cudaSafeCall( cudaMalloc( &test_triangles, total_trigs * sizeof( int3 ) ) );

	float3* test_normals;
	cudaSafeCall( cudaMalloc( &test_normals, total_verts * sizeof( float3 ) ) );

	float3* test_rel_coors;
	cudaSafeCall( cudaMalloc( &test_rel_coors, total_verts * sizeof( float3 ) ) );

	int3* test_vol_idx;
	cudaSafeCall( cudaMalloc( &test_vol_idx, total_verts * sizeof( int3 ) ) );
				
	int cube_num = ceil(active_voxels / (float)max_threads);
    mc_kernel<<< cube_num, max_threads >>>(grid, centers, size, sc, 
    										test_vertices, test_triangles, test_normals, test_vol_idx, test_rel_coors,
    										volume.get_state(), d_comp_voxel, d_voxel_verts, active_voxels);
	cudaSafeCall( cudaDeviceSynchronize() );
	
	vertices.resize(total_verts);
	cudaSafeCall( cudaMemcpy( vertices.data(), test_vertices, total_verts * sizeof( float3 ), cudaMemcpyDeviceToHost) );
	
	triangles.resize(total_trigs);
	cudaSafeCall( cudaMemcpy( triangles.data(), test_triangles, total_trigs * sizeof( int3 ), cudaMemcpyDeviceToHost) );

	normals.resize(total_verts);
	cudaSafeCall( cudaMemcpy( normals.data(), test_normals, total_verts * sizeof( float3 ), cudaMemcpyDeviceToHost) );

	rel_coors.resize(total_verts);
	cudaSafeCall( cudaMemcpy( rel_coors.data(), test_rel_coors, total_verts * sizeof( float3 ), cudaMemcpyDeviceToHost) );

	vol_idx.resize(total_verts);
	cudaSafeCall( cudaMemcpy( vol_idx.data(), test_vol_idx, total_verts * sizeof( int3 ), cudaMemcpyDeviceToHost) );
				
	cudaSafeCall( cudaFree(test_vertices) );
	cudaSafeCall( cudaFree(test_triangles) );
	cudaSafeCall( cudaFree(test_normals) );
	cudaSafeCall( cudaFree(test_rel_coors) );
	cudaSafeCall( cudaFree(test_vol_idx) );
		
	cudaSafeCall( cudaFree(d_voxel_verts) );
	cudaSafeCall( cudaFree(d_comp_voxel) );
	
	clock.tok();
	std::cout << "Total elapsed time for mc_kernel:" << clock.toMilliseconds() << " ms" << std::endl;		
	

}
