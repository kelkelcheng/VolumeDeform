#include <iostream>
#include "MarchingCubes.h"
#include "GPUTiktok.h"
//#include "MC_edge_table.cu"
//#include "MC_triangle_table.cu"
#include "MC_table.cu"
#include "cudaUtil.h"
using namespace std;

#define CUDART_NAN_F            __int_as_float(0x7fffffff)
// CUDA kernel function to integrate a TSDF voxel volume given depth images

__device__
float3 compute_intersection_for_edge( int edge_index,
                                      const float voxel_values[8],
                                      const float3 cube_vertices[8] ) {
	// The expectation here is that 
	// : Cube vertices is populated with real world coordinates of the cube being marched
	// : voxel values are the corresponding weights of the vertices
	// : edge index is an edge to be intersected
	// : The vertex at one end of that edge has a negative weigt and at the otehr, a positive weight
	// : The intersection should be at the zero crossing

	// Check assumptions
	int v1_index = EDGE_VERTICES[edge_index][0];
	int v2_index = EDGE_VERTICES[edge_index][1];

	float3 start_vertex = cube_vertices[v1_index];
	float start_weight = voxel_values[v1_index];

	float3 end_vertex   = cube_vertices[v2_index];
	float end_weight = voxel_values[v2_index];

	if (  (start_weight > 0 ) &&  (end_weight <  0 ) ) {
		// Swap start and end
		float3 temp3 = start_vertex;
		start_vertex = end_vertex;
		end_vertex = temp3;

		float temp = start_weight;
		start_weight = end_weight;
		end_weight = temp;
	} else if ( ( start_weight * end_weight ) > 0 ) {
		printf( "Intersected edge expected to have differenlty signed weights at each end\n");
		asm("trap;");
	}

	float ratio = ( 0 - start_weight ) / ( end_weight - start_weight);


	// Work out where this lies
	float3 edge = make_float3(end_vertex.x - start_vertex.x, end_vertex.y -start_vertex.y, end_vertex.z - start_vertex.z);
	float3 delta = make_float3(ratio * edge.x, ratio * edge.y, ratio * edge.z);
	float3 intersection = make_float3(start_vertex.x + delta.x, start_vertex.y + delta.y, start_vertex.z + delta.z); 

	return intersection;
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
int compute_edge_intersects( uint8_t cube_index,
                             const float voxel_values[8],
                             const float3 cube_vertices[8],
                             float3 intersects[12]) {
	// Get the edges impacted
	int num_edges_impacted = 0;
	if ( ( cube_index != 0x00) && ( cube_index != 0xFF ) ) {
		uint16_t intersected_edge_flags = EDGE_TABLE[cube_index];
		uint16_t mask = 0x01;
		for ( int i = 0; i < 12; i++ ) {
			if ( ( intersected_edge_flags & mask ) > 0 ) {

				intersects[i] = compute_intersection_for_edge( i, voxel_values, cube_vertices);

				num_edges_impacted++;
			} else {
				intersects[i].x = CUDART_NAN_F;
				intersects[i].y = CUDART_NAN_F;
				intersects[i].z = CUDART_NAN_F;
			}
			mask = mask << 1;
		}
	}
	return num_edges_impacted;
}

__device__
void compute_cube_vertices(int x, int y, int width, const float3* layer1, const float3* layer2, float3 cube_vertices[8]){
	  int base = y * width + x;
	  cube_vertices[0] = layer1[base];
	  cube_vertices[1] = layer1[base + 1];
	  cube_vertices[2] = layer2[base + 1];
	  cube_vertices[3] = layer2[base];
	  cube_vertices[4] = layer1[base + width];
	  cube_vertices[5] = layer1[base + width + 1];
      cube_vertices[6] = layer2[base + width + 1];
	  cube_vertices[7] = layer2[base + width];
	  
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

/**
 * Compute the triangles for two planes of data
 * @param tsdf_values_layer_1 The first layer of values
 * @param tsdf_values_layer_2 The second layer of values
 * @param dim_x,y,z dimemsion of TSDF
 * @param origin_x,y,z origin coordinate of TSDF
 * @param voxel_size size of voxel
 * @param vz The index of the plane being considered
 * @param vertices An array of 12 vertices per cube
 * @param triangels An array of 5 triangles per cube
 */
__global__
void mc_kernel( const float * tsdf_values_layer_1,
				const float * tsdf_values_layer_2,
				const float3 * centers_layer_1,
				const float3 * centers_layer_2,
                dim3 size,
				float3 origin,
                int   vz,

                // Output variables
				float3 *vertices,
                int3 *triangles ) {


	// Extract the voxel X and Y coordinates which describe the position in the layers
	// We use layer1 = z0, layer2 = z1
	int vx = threadIdx.x + (blockIdx.x * blockDim.x);
	int vy = threadIdx.y + (blockIdx.y * blockDim.y);

	// If this thread is in range (we only go up to penultimate X and Y values)
	if ( ( vx < size.x - 1 ) && ( vy <size.y - 1 ) ) {

		// Compute index of the voxel to address (used to index voxel data)
		int voxel_index =  (size.x * vy) + vx;

		// Compute cube index (ised to index output tris and verts)
		int cube_index =      ((size.x - 1) * vy) + vx;
		int vertex_index   =  cube_index * 12;
		int triangle_index =  cube_index *  5;

		// Load voxel values for the cube
		float voxel_values[8] = {
			tsdf_values_layer_1[voxel_index],							//	vx,   vy,   vz
			tsdf_values_layer_1[voxel_index + 1],						//	vx + 1,   vy,   vz
			tsdf_values_layer_2[voxel_index + 1],		//	vx + 1, vy, vz+1
			tsdf_values_layer_2[voxel_index],		//	vx, vy , vz+1
			tsdf_values_layer_1[voxel_index + size.x],						//	vx, vy+1,	vz
			tsdf_values_layer_1[voxel_index + size.x+ 1],						//	vx+1, vy+1, 	vz
			tsdf_values_layer_2[voxel_index + size.x+ 1],	//	vx+1, vy+1, vz+1
			tsdf_values_layer_2[voxel_index + size.x]	//	vx, vy+1, vz + 1
		};

		// Compute the cube type
		uint8_t cube_type = cube_type_for_values( voxel_values );

		// If it's a non-trivial cube_type, process it
		if ( ( cube_type != 0 ) && ( cube_type != 0xFF ) ) {

			// Compuyte the coordinates of the vertices of the cube
			float3 cube_vertices[8];
			compute_cube_vertices(vx,vy,size.x, centers_layer_1, centers_layer_2, cube_vertices);

			// Compute intersects (up to 12 per cube)
			float3 	intersects[12];
			compute_edge_intersects( cube_type, voxel_values, cube_vertices, intersects);

			// Copy these back into the return vaue array at the appropriate point for this thread
			for ( int i = 0; i < 12; i++ ) {
				vertices[vertex_index + i] = intersects[i];
			}

			// These intersects form triangles in line with the MC triangle table
			// We compute all five triangles because we're writing to a fixed size array
			// and we need to ensure that every thread knows where to write.
			int i = 0;
			for ( int t = 0; t < 5; t++ ) {
				triangles[triangle_index + t].x = TRIANGLE_TABLE[cube_type][i++];
				triangles[triangle_index + t].y = TRIANGLE_TABLE[cube_type][i++];
				triangles[triangle_index + t].z = TRIANGLE_TABLE[cube_type][i++];
			}
		} else {
			// Set all triangle to have -ve indices
			for ( int t = 0; t < 5; t++ ) {
				triangles[triangle_index + t].x = -1;
				triangles[triangle_index + t].y = -1;
				triangles[triangle_index + t].z = -1;
			}
		}
	}
}

__host__
void process_kernel_output( dim3 size,
							const float3          * h_vertices,
                            const int3            * h_triangles,
                            vector<float3>&    vertices,
                            vector<int3>&      triangles) {
	

	// For all but last row of voxels
	int cube_index = 0;
	for ( int y = 0; y < size.y - 1; y++ ) {

		// For all but last column of voxels
		for ( int x = 0; x < size.x - 1; x++ ) {

			// get pointers to vertices and triangles for this voxel
			const float3* verts = h_vertices  + ( cube_index * 12 );
			const int3* tris  = h_triangles + ( cube_index * 5  );

			// Iterate until we have 5 triangles or there are none left
			int tri_index = 0;
			while ( ( tri_index < 5) && ( tris[tri_index].x != -1 ) ) {

				// Get the raw vertex IDs
				int tri_vertices[3];
				tri_vertices[0] = tris[tri_index].x;
				tri_vertices[1] = tris[tri_index].y;
				tri_vertices[2] = tris[tri_index].z;


				// Remap triangle vertex indices to global indices
				int remap[] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
				for ( int tv = 0; tv < 3; tv++ ) {

					int vid = tri_vertices[tv];
					int vertexid = remap[ vid ];
					if ( vertexid == -1 ) {
						// This vertex hasnt been remapped (ie stored) yet
						vertices.push_back( verts[ vid ] );

						// Get the new ID
						vertexid = vertices.size() - 1;

						// And enter in remap table
						remap[ vid ] = vertexid;
					}
					tri_vertices[tv] = vertexid;
				}

				// Store the triangle
				int3 triangle = make_int3(tri_vertices[0], tri_vertices[1], tri_vertices[2]);
				triangles.push_back( triangle );

				tri_index++;
			}
			cube_index++;
		}
	}
	
}

__host__
void extract_surface(TSDFVolume & volume, vector<float3>& vertices, vector<int3>& triangles){
	// Allocate storage on device and locally
	// Fail if not possible
	dim3 size = volume.get_size();
	//int tmp = size.x; size.x = size.z; size.z = tmp;

	size_t num_cubes_per_layer = (size.x - 1) * (size.y - 1);

	// Device vertices
	float3* d_vertices;
	size_t num_vertices =  num_cubes_per_layer * 12;
	cudaError_t err = cudaMalloc( &d_vertices, num_vertices * sizeof( float3 ) );
	if ( err != cudaSuccess ) {
		cout << "Couldn't allocate device memory for vertices" << endl;
		throw std::bad_alloc( );
	}

	// Device triangles
	int3* d_triangles;
	size_t num_triangles  = num_cubes_per_layer * 5;
	err = cudaMalloc( &d_triangles, num_triangles * sizeof(int3) );
	if ( err != cudaSuccess ) {
		cudaFree( d_vertices );
		cout << "Couldn't allocate device memory for triangles" << endl;
		throw std::bad_alloc( );
	}

	// Host vertices
	float3* h_vertices = new float3[ num_vertices ];
	if ( !h_vertices ) {
		cudaFree( d_vertices);
		cudaFree( d_triangles);
		cout << "Couldn't allocate host memory for vertices" << endl;
		throw std::bad_alloc( );
	}

	// Host triangles
	int3 *h_triangles = new int3 [  num_triangles ];
	if ( !h_triangles ) {
		delete [] h_vertices;
		cudaFree( d_vertices);
		cudaFree( d_triangles);
		cout << "Couldn't allocate host memory for triangles" << endl;
		throw std::bad_alloc( );
	}
	
	util::GPUTiktok clock; 
	

	// Now iterate over each slice
	size_t layer_size =  size.x * size.y;
	clock.tik();
	const float* grid = volume.get_grid();
	const float3* centers = volume.get_deform();
	for ( int vz = 0; vz < size.z - 1; vz++ ) {

		// Set up for layer
		const float * layer1_data = &(grid[vz * layer_size] );
		const float * layer2_data = &(grid[(vz + 1) * layer_size] );

		const float3 * layer1_center = &(centers[vz * layer_size]);
		const float3 * layer2_center = &(centers[(vz + 1) * layer_size]);

		// invoke the kernel
		dim3 block( 16, 16, 1 );
		dim3 grid ((size.x + block.x - 1) /block.x, (size.y + block.y - 1)/block.y, 1 );
		
		mc_kernel <<< grid,block >>>(layer1_data,layer2_data,
									layer1_center, layer2_center,
		                               size,
									   volume.get_origin(),
		                               vz, d_vertices, d_triangles );

		err = cudaDeviceSynchronize( );		
		if(err!=cudaSuccess)
			cout << "device synchronize failed" << endl;
		
		
		// Copy the device vertices and triangles back to host
		err = cudaMemcpy( h_vertices, d_vertices, num_vertices * sizeof( float3 ), cudaMemcpyDeviceToHost);
		if(err!=cudaSuccess)
			cout << "Copy of vertex data fom device failed " << endl;

		err = cudaMemcpy( h_triangles, d_triangles, num_triangles * sizeof( int3 ), cudaMemcpyDeviceToHost);
		if(err!=cudaSuccess)
			cout << "Copy of triangle data from device failed " << endl;
		

		// All through all the triangles and vertices and add them to master lists
		process_kernel_output(size, h_vertices, h_triangles, vertices, triangles);
		
	}
	clock.tok();
	std::cout << "Total elapsed time:" << clock.toMilliseconds() << " ms" << std::endl;
	// Free memory and done
	err = cudaFree( d_vertices);
	//check_cuda_error( "extract_vertices: Free device vertex memory failed " , err);
	err = cudaFree( d_triangles);
	//check_cuda_error( "extract_vertices: Free device triangle memory failed " , err);

	delete [] h_vertices;
	delete [] h_triangles;
}
