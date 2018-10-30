#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"

#include "utils.hpp"
#include "TSDFVolume.h"
#include "MarchingCubes.h"

#include <sstream>
#include <iomanip>

#include <ctime>


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
	
	// start frame, default to be 0
	int start_frame = 0;
	std::stringstream ss;

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
        }
    }

	// file name for the first frame
	ss << std::setw(6) << std::setfill('0') << start_frame;
	std::string sourceFilename = targetSourceDirectory+"/frame-"+ss.str()+".depth.png";
	ss.str("");
	
	// set up TSDF volume, its inputs are:
	// volume(x_length, y_length, z_length, origin, voxel_size, subgrid_scale)
	float3 voxel_size = make_float3(1.0f/361.0f, 1.0f/376.0f, 0.45f/61.0f); 
	int im_width = 640; int im_height = 480;	
	TSDFVolume volume(361,376,61, make_float3(-0.5611f,-0.4208f, 0.6f), voxel_size, make_int3(15,15,20), im_height, im_width); 
	
	// another set of parameters: a smaller grid
	//float3 voxel_size = make_float3(0.6f/361.0f, 0.7f/376.0f, 0.45f/61.0f);
	//TSDFVolume volume(361,376,61, make_float3(-0.36f,-0.26f, 0.6f), voxel_size, make_int3(15,15,20));

	// set up intrinsic and extrinsic matrices
	float K_mat[9] = {570.342, 0, 320,  0, 570.342, 240,  0, 0, 1};
	float K_inv_mat[9] = {0.00175333, 0, -0.561066,  0, 0.00175333, -0.4208001,  0, 0, 1};
	float world2cam[16] = {1, 0, 0, 0, 0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1};

	// get the first depth frame and integrate
	float* depth_im = new float[im_height * im_width];
	ReadDepth(sourceFilename, im_height, im_width, depth_im);
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
	write_to_ply("../output_mesh/before_integration.ply",vertices,triangles);

	// store all the depth image names
	std::string target_name;
	std::vector<std::string> targetFiles;
	std::vector<std::string> target_set;
	
	for (int i = start_frame + 1; i < (start_frame + num_frames); i++) {
		ss << std::setw(6) << std::setfill('0') << i;
		target_set.push_back(targetSourceDirectory+"/frame-"+ss.str()+".depth.png");
		ss.str("");
	}

	// initialize the solver (Opt)
	CombinedSolver * solver =  new CombinedSolver(	target_set, params, 
													&volume, 
													&vertices, &normals, &triangles, 
													&vol_idx, &rel_coors, 
													K_mat, K_inv_mat,
													im_height, im_width);

	// record the processing time
    std::clock_t start;
    double duration;
    start = std::clock();		
    
    // set up depth frame for solver -- using the full depth frame seem to be more robust (depth_im will set depth > 1 to be 0)
    float * full_depth = new float[im_height * im_width];
    solver->set_depth_mat(full_depth);
    
	for (int i=0; i<target_set.size(); i++) {
		// readin the depth image name
		target_name = target_set[i];
		targetFiles.clear();
		targetFiles.push_back(target_name);
		std::cout << "target mesh: " << targetFiles[0] << std::endl;
		solver->set_targets(targetFiles);
		
		// read the truncated depth
		ReadDepth(target_name, im_height, im_width, depth_im);
		
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
		//write_to_ply("../output_mesh/after_integration"+std::to_string(i+1)+".ply",vertices,triangles);
	}
	// output the processing time
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<std::endl<<"total time: "<< duration << " seconds" <<std::endl;
    
    // generate the deformed grid if needed
    if (grid_flag) {solver->saveGraphResults();}
    
    // generate the final output
	write_to_ply("../output_mesh/after_integration_final.ply",vertices,triangles);
	
	// clean up
	delete solver;
	delete depth_im;
	delete full_depth;
	return 0;
}
