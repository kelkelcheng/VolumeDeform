#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"

#include "utils.hpp"
#include "TSDFVolume.h"
#include "MarchingCubes.h"

#include <sstream>
#include <iomanip>

#include <ctime>

std::string read_file(std::string targetSourceDirectory, int i, int dm_num)
{
	std::stringstream ss;
	std::string filename;
	if (dm_num <= 0) {
		ss << std::setw(6) << std::setfill('0') << i;
		filename = targetSourceDirectory+"/frame-"+ss.str()+".depth.png";
	} else {
		ss << std::setw(4) << std::setfill('0') << i;
		filename = targetSourceDirectory+"/"+ss.str()+"_depth.png";
		//ss << std::setw(6) << std::setfill('0') << i;
		//filename = targetSourceDirectory+"/frame-"+ss.str()+".depth.png";
	}
	ss.str("");
	return filename;
}

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)> SOURCES"
              << "Options:\n"
              << "\t-h,--help\t\tShow this help message\n"
              << "\t-i,--numIter x\t\tnumber of iterations (ICP)\n"
              << "\t-ni,--nonLinearIter x\t\tnumber of nonlinear iterations\n"
              << "\t-li,--linearIter x\t\tnumber of linearIter iterations\n"
              << "\t-n,--numFrames n\t\tnumber of frames\n"
              << "\t-d,--dataPath PATH\t\tpath to the data set\n"
              << "\t-g,--grid\t\toutput the deformed grid\n"
              << "\t-s,--startFrame s\t\twhich frame to start\n"
			  << "\t-dm,--depthMap s\t\tswitch to depth map generation mode\n"
              << std::endl;
}

int main(int argc, const char * argv[])
{   
	// default parameters for the solvers
	std::cout << "Begins: " << std::endl;
    CombinedSolverParameters params;
    params.numIter = 2; 
    params.nonLinearIter = 1;
    params.linearIter = 32;
    params.useOpt = true;
    params.useOptLM = false;
    
    // number of frames, default to be 30
	int num_frames = 30;
	
	// dataset path
	//you can choose: upper_body_depth || hoodie_depth || minion_depth || sunflower_depth
	std::string targetSourceDirectory = "../data/upper_body_depth";
	
	// set to 1 to output the deformed grid
	bool grid_flag = 0;

	// number of generated depth map
	int dm_num = 5;
	// number of files in the directory
	int dm_total = 190;
	
	// start frame, default to be 0
	int start_frame = 0;
	std::stringstream ss;

	// floor height for depth map refinement
	float floor_height = 0;

	// readin parameters
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(std::string(argv[0]));
            return 0;
        } else if ((arg == "-i") || (arg == "--numIter")) { // number of iterations (ICP)
            if (i + 1 < argc) { 
                params.numIter = atoi(argv[++i]); 
            } else { 
                std::cerr << "--numIter option requires one argument." << std::endl;
                return 1;
            }  
        } else if ((arg == "-ni") || (arg == "--nonLinearIter")) { // number of nonlinear iterations
            if (i + 1 < argc) { 
                params.nonLinearIter = atoi(argv[++i]); 
            } else { 
                std::cerr << "--nonLinearIter option requires one argument." << std::endl;
                return 1;
            } 
        } else if ((arg == "-li") || (arg == "--linearIter")) { // number of linear iterations
            if (i + 1 < argc) { 
                params.linearIter = atoi(argv[++i]); 
            } else { 
                std::cerr << "--linearIter option requires one argument." << std::endl;
                return 1;
            } 
        } else if ((arg == "-n") || (arg == "--numFrames")) { // number of frames
            if (i + 1 < argc) { 
                num_frames = atoi(argv[++i]);             
            } else { 
                std::cerr << "--numFrames option requires one argument." << std::endl;
                return 1;
            } 
        } else if ((arg == "-d") || (arg == "--dataPath")) { // set data path
            if (i + 1 < argc) { 
                targetSourceDirectory = "../data/" + std::string(argv[++i]);
            } else { 
                std::cerr << "--dataPath option requires one argument." << std::endl;
                return 1;
            } 
        } else if ((arg == "-s") || (arg == "--startFrame")) { // set data path
            if (i + 1 < argc) { 
                start_frame = atoi(argv[++i]);
            } else { 
                std::cerr << "--startFrame option requires one argument." << std::endl;
                return 1;
            } 
        } else if ((arg == "-g") || (arg == "--grid")) { // if output the deformed grid
            grid_flag = 1;
        } else if ((arg == "-dm") || (arg == "--depthMap")) { // number of generated depth maps
            if (i + 1 < argc) { 
                dm_num = atoi(argv[++i]);             
            } else { 
                std::cerr << "--depthMap option requires one argument." << std::endl;
                return 1;
            } 
        } else if ((arg == "-dmt") || (arg == "--depthMapTotal")) { // number of generated depth maps
            if (i + 1 < argc) { 
                dm_total = atoi(argv[++i]);
				if (dm_total < dm_num) {
					std::cerr << "--depthMapTotal needs to be larger than the number of generated depth maps" << std::endl;
					return 1;
				}             
            } else { 
                std::cerr << "--depthMapTotal option requires one argument." << std::endl;
                return 1;
            } 
        } else if ((arg == "-fh") || (arg == "--floorHeight")) { // number of generated depth maps
            if (i + 1 < argc) { 
                floor_height = atof(argv[++i]);             
            } else { 
                std::cerr << "--floorHeight option requires one argument." << std::endl;
                return 1;
            } 
		}
    }

	// file name for the first frame
	std::string sourceFilename = read_file(targetSourceDirectory, start_frame, dm_num);
	std::cout << "first file: " << sourceFilename << std::endl;
	/*
	if (dm_num > 0) {
		ss << std::setw(6) << std::setfill('0') << start_frame;
		std::string sourceFilename = targetSourceDirectory+"/frame-"+ss.str()+"depth.png";
	} else {
		ss << std::setw(4) << std::setfill('0') << start_frame;
		std::string sourceFilename = targetSourceDirectory+"/"+ss.str()+"_depth.png";		
	}
	ss.str("");
	*/

	// set up TSDF volume, its inputs are:
	// volume(x_length, y_length, z_length, origin, voxel_size, subgrid_scale)
	float3 voxel_size = make_float3(1.0f/361.0f, 1.0f/376.0f, 0.6f/61.0f); 
	//float3 voxel_size = make_float3(2.0f/361.0f, 2.0f/376.0f, 0.6f/61.0f); 
	//float3 voxel_size = make_float3(1.0f/361.0f, 1.0f/376.0f, 0.25f/61.0f);

	int im_width = 640; int im_height = 480;
	//int im_width = 256; int im_height = 256;		
	bool is_perspective = true;

	TSDFVolume volume(361,376,61, make_float3(-0.5611f,-0.4208f, 0.55f), voxel_size, make_int3(15,15,20), im_height, im_width, is_perspective); 
	//TSDFVolume volume(361,376,61, make_float3(0.01f,0.01f, 0.30f), voxel_size, make_int3(15,15,20), im_height, im_width, is_perspective); 
	//volume deform
	//TSDFVolume volume(361,376,61, make_float3(-0.5611f,-0.4208f, 0.6f), voxel_size, make_int3(15,15,20), im_height, im_width, is_perspective); 

	// another set of parameters: a smaller grid
	//float3 voxel_size = make_float3(0.75f/361.0f, 0.7f/376.0f, 0.45f/61.0f);
	//TSDFVolume volume(361,376,61, make_float3(-0.36f,-0.26f, 0.6f), voxel_size, make_int3(15,15,20), im_height, im_width, is_perspective);

	// set up intrinsic and extrinsic matrices
	// for VolumeDeform data
	//float K_mat[9] = {570.342, 0, 320,  0, 570.342, 240,  0, 0, 1};
	//float K_inv_mat[9] = {0.00175333, 0, -0.561066,  0, 0.00175333, -0.4208001,  0, 0, 1};

	// for orthographic projection
	//float K_mat[3 * 3] = {(float)(im_width-1)/2.0f, 0, 0,  0, (float)(im_height-1)/2.0f, 0,  0, 0, 1};
	//float K_inv_mat[3 * 3] = {2.0f/(im_width-1), 0, 0,  0, 2.0f/(float)(im_height-1), 0,  0, 0, 1};

	// for self-acquired Kinect data
	float K_mat[3 * 3] = {1066.01/2.22, 0, (945.0/2.2 - 100.0),  0, 1068.87/2.22, 520/2.22,  0, 0, 1};
	float K_inv_mat[3 * 3] = {0.00208, 0, -0.68629,  0, 0.00208, -0.48650,  0, 0, 1};

	// Sicong data
	//float K_mat[3 * 3] = {532.4718, 0, 329.5454,  0, 532.4718, 234.2342,  0, 0, 1};
	//float K_inv_mat[3 * 3] = {0.00188, 0, -0.61890,  0, 0.00188, -0.43990,  0, 0, 1};	

	float world2cam[16] = {1, 0, 0, 0, 0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1};

	// get the first depth frame and integrate
	float* depth_im = new float[im_height * im_width];
	ReadDepth(sourceFilename, im_height, im_width, depth_im, K_inv_mat, floor_height);
	//write_pt_cloud("../output_mesh/input_pt_cloud.obj", sourceFilename, im_height, im_width);
	write_mesh("../output_mesh/input_dm_to_mesh.obj", sourceFilename, im_height, im_width, K_inv_mat);

	//show mesh after cropping
	depthmap_to_mesh("../output_mesh/input_dm_to_mesh_crop.obj", depth_im,  im_height, im_width, K_inv_mat);

	//TODO test
	//write_mesh("../output_mesh/input_dm_to_mesh2.obj", "../data/kinect/kinect_1_gt/frame-000000.depth.png", 256, 256);
	std::cout << "First image: " << sourceFilename << std::endl;
	volume.Integrate(depth_im, K_mat, world2cam, 1);
	
	// 3D positions of each mesh point
	std::vector<float3> vertices;
	// the 3 indices of mesh points that form a triangle
	std::vector<int3> triangles;
	// unit normals of each mesh point
	std::vector<float3> normals;
	// TSDF cube index of each mesh point 
	std::vector<int3> vol_idx;
	// sparse subgrid trilinear weights of each mesh point (not the full TSDF grid)
	std::vector<float3> rel_coors;
	
	// generate the first mesh using parallelized marching cube
	extract_surface(volume, vertices, triangles, normals, vol_idx, rel_coors);
	//write_to_ply("../output_mesh/before_integration.ply",vertices,triangles);
	write_to_ply("../output_mesh/before_integration.obj",vertices,triangles);

	// store all the depth image names
	std::string target_name;
	std::vector<std::string> targetFiles;
	std::vector<std::string> target_set;
	
	for (int i = start_frame + 1; i < (start_frame + num_frames); i++) {
		/*ss << std::setw(6) << std::setfill('0') << i;
		target_set.push_back(targetSourceDirectory+"/frame-"+ss.str()+".depth.png");
		ss.str("");*/
		target_set.push_back(read_file(targetSourceDirectory, i, dm_num));
	}

	// initialize the solver (Opt)
	CombinedSolver * solver =  new CombinedSolver(	target_set, params, 
													&volume, 
													&vertices, &normals, &triangles, 
													&vol_idx, &rel_coors, 
													K_mat, K_inv_mat,
													im_height, im_width, is_perspective, floor_height);

	// record the processing time
    std::clock_t start;
    double duration;
    start = std::clock();		
    
    // set up depth frame for solver -- using the full depth frame seem to be more robust (depth_im will set depth > 1 to be 0)
    float * full_depth = new float[im_height * im_width];
    solver->set_depth_mat(full_depth);
    
	int dm_range = 1;
	if (dm_num > 0) dm_range = dm_num;
	std::string dm_dir = "../output_mesh/output_depthmap/";
	std::string dm_obj_dir = "../output_mesh/output_depthmap_obj/";

	for (int k=0; k<dm_range; k++) {
		// if in depth map generation mode
		if (dm_num > 0) {
			volume.Reset();
			target_set.clear();
			solver->reset();

			if (start_frame + k <= dm_total - num_frames) {
				for (int i = start_frame + k; i < (start_frame + num_frames + k); i++) {
					/*ss << std::setw(6) << std::setfill('0') << i;
					target_set.push_back(targetSourceDirectory+"/frame-"+ss.str()+".depth.png");
					ss.str("");*/
					target_set.push_back(read_file(targetSourceDirectory, i, dm_num));
				}
			} else {
				for (int i = start_frame + k; i > (start_frame - num_frames + k); i--) {
					/*ss << std::setw(6) << std::setfill('0') << i;
					target_set.push_back(targetSourceDirectory+"/frame-"+ss.str()+".depth.png");
					ss.str("");*/
					target_set.push_back(read_file(targetSourceDirectory, i, dm_num));
				}				
			}

			/*ss << std::setw(6) << std::setfill('0') << start_frame + k;
			std::string start_filename = targetSourceDirectory+"/frame-"+ss.str()+".depth.png";
			ss.str("");*/
			std::string start_filename = read_file(targetSourceDirectory, start_frame+k, dm_num);

			ReadDepth(start_filename, im_height, im_width, depth_im, K_inv_mat, floor_height);
			volume.Integrate(depth_im, K_mat, world2cam, 1);
			extract_surface(volume, vertices, triangles, normals, vol_idx, rel_coors);
		}

		// fusion begins
		for (int i=0; i<target_set.size(); i++) {
			// readin the depth image name
			target_name = target_set[i];
			targetFiles.clear();
			targetFiles.push_back(target_name);
			std::cout << "target mesh: " << targetFiles[0] << std::endl;
			solver->set_targets(targetFiles);
			
			// read the truncated depth
			ReadDepth(target_name, im_height, im_width, depth_im, K_inv_mat, floor_height);
			
			// read the full depth inside the solver
			solver->update_depth();
			
			// update the pointers
			solver->set_vnt(&vertices, &normals, &triangles, &vol_idx);
			
			// run the solver
			solver->solveAll();

			// update the 3d positions of the sparse grid after optimization (from GPU memory to main memory)
			solver->update_grid();
			// upsamling the sparse grid to update the whole grid
			volume.Upsample(*(solver->get_grid()));

			// integrate
			volume.Integrate(depth_im, K_mat, world2cam, 0);
			
			// parallel marching cube
			vertices.clear(); triangles.clear(); normals.clear(); vol_idx.clear(); rel_coors.clear();
			extract_surface(volume, vertices, triangles, normals, vol_idx, rel_coors);
			//write_to_ply("../output_mesh/render/after_integration"+std::to_string(i+1)+".ply",vertices,triangles);
			//write_to_ply("../output_mesh/render/after_integration"+std::to_string(start_frame+1)+".obj",vertices,triangles);
		}

		// depth map generation mode
		if (dm_num > 0) {
			extract_surface(volume, vertices, triangles, normals, vol_idx, rel_coors, false);
			//write_to_ply("../output_mesh/after_integration_ori_grid.obj",vertices,triangles);
			solver->set_vnt(&vertices, &normals, &triangles, &vol_idx);

			ss << std::setw(4) << std::setfill('0') << start_frame + k;
			//std::string start_filename = targetSourceDirectory+"/frame-"+ss.str()+".depth.png";
			solver->create_depthmap(dm_dir + ss.str()+ "_depth.png");
			//TODO test depth map
			write_mesh(dm_obj_dir + "test_"+std::to_string(start_frame + k)+".obj", dm_dir + ss.str()+ "_depth.png", im_height, im_width, K_inv_mat);	
			ss.str("");
		}	
	}

	//TODO temporary
	/*extract_surface(volume, vertices, triangles, normals, vol_idx, rel_coors, false);
	write_to_ply("../output_mesh/after_integration_ori_grid.obj",vertices,triangles);
	solver->set_vnt(&vertices, &normals, &triangles, &vol_idx);
	solver->create_depthmap("../output_mesh/after_integration_depthmap.png");
	//TODO test depth map
	write_mesh("../output_mesh/test.obj", "../output_mesh/after_integration_depthmap.png", im_height, im_width, K_inv_mat);*/

	// output the processing time
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<std::endl<<"total time: "<< duration << " seconds" <<std::endl;
    
    // generate the deformed grid if needed
    if (grid_flag) {solver->saveGraphResults();}
    
    // generate the final output
	//write_to_ply("../output_mesh/after_integration_final.ply",vertices,triangles);
	//if (dm_num <= 0 ) {
		vertices.clear(); triangles.clear(); normals.clear(); vol_idx.clear(); rel_coors.clear();
		extract_surface(volume, vertices, triangles, normals, vol_idx, rel_coors);	
		write_to_ply("../output_mesh/after_integration_final.obj",vertices,triangles);
	//}
	
	// clean up
	delete solver;
	delete depth_im;
	delete full_depth;
	return 0;
}
