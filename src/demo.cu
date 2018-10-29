#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include "utils.hpp"
#include "TSDFVolume.h"
#include "MarchingCubes.h"

#include <cstdio>
#include <ctime>

using namespace cv;
using namespace std;


// Loads a binary file with depth data and generates a TSDF voxel volume (5m x 5m x 5m at 1cm resolution)
// Volume is aligned with respect to the camera coordinates of the first frame (a.k.a. base frame)
int main(int argc, char * argv[]) {

	// Location of folder containing RGB-D frames and camera pose files
	//std::string data_path = "data";
	//int base_frame_idx = 0;
	int first_frame_idx = 0;
	float num_frames = 1;
	float cam_K[3 * 3] = {570.342,0,320,  0,570.342,240, 0,0,1};
	float world2cam[4 * 4] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	int im_width = 640;
	int im_height = 480;
	float* depth_im = new float[480 * 640];

	// Voxel grid parameters (change these to change voxel grid resolution, etc.)
	int dim_x = 500; int dim_y = 500; int dim_z = 100; float voxel_size = 0.012f; //0.006f
	if (argc > 1) dim_x = atoi(argv[1]);
	if (argc > 2) dim_y = atoi(argv[2]);    
	if (argc > 3) dim_z = atoi(argv[3]);
	std::cout << "dim_x: " << dim_x << " dim_y: " << dim_y << " dim_z: " << dim_z << std::endl;

	std::clock_t start;
	double duration;
	start = std::clock();
	TSDFVolume volume(dim_x, dim_y, dim_z, make_float3(-1.5f,-1.5f,0.5f), voxel_size);
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout<<"TSDF Initialization: "<< duration <<'\n';
	// Read camera intrinsics
	//std::vector<float> cam_K_vec = LoadMatrixFromFile(cam_K_file, 3, 3);
	//std::copy(cam_K_vec.begin(), cam_K_vec.end(), cam_K);


	// Invert base frame camera pose to get world-to-base frame transform 
	//float base2world_inv[16] = {0};
	//invert_matrix(base2world, base2world_inv);


	// Loop through each depth frame and integrate TSDF voxel grid
	for (int frame_idx = first_frame_idx; frame_idx < first_frame_idx + (int)num_frames; ++frame_idx) {

		std::ostringstream curr_frame_prefix;
		curr_frame_prefix << std::setw(6) << std::setfill('0') << frame_idx;

		// // Read current frame depth
		std::string depth_im_file = "data/frame-" + curr_frame_prefix.str() + ".depth.png";
		std::cout << "data/frame-" + curr_frame_prefix.str() + ".depth.png" << std::endl;
		ReadDepth(depth_im_file, im_height, im_width, depth_im);

		// Compute relative camera pose (camera-to-base frame)
		// multiply_matrix(base2world_inv, cam2world, cam2base);


		std::cout << "Fusing: " << depth_im_file << std::endl;

		start = std::clock();
		volume.Integrate(depth_im,cam_K,world2cam);

		duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
		std::cout<<"TSDF Integration: "<< duration <<'\n';
	}

	// Compute surface points from TSDF voxel grid and save to point cloud .ply file
	std::cout << "Saving surface point cloud (tsdf.ply)..." << std::endl;
	std::vector<float3> vertices ;
	std::vector<int3> triangles;
	extract_surface(volume, vertices, triangles);
	write_to_ply("upperbody.ply",vertices,triangles);


	delete depth_im;
	return 0;
}
