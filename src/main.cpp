#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"

//#include "utils.hpp"
#include "TSDFVolume.h"
#include "MarchingCubes.h"

#include <sstream>
#include <iomanip>

#include <ctime>

static SimpleMesh* createMesh(std::string filename) {
    SimpleMesh* mesh = new SimpleMesh();
    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << filename << std::endl;
        exit(1);
    }
    printf("Faces: %d\nVertices: %d\n", (int)mesh->n_faces(), (int)mesh->n_vertices());
    return mesh;
}


int main(int argc, const char * argv[])
{
	std::cout << "Function begins: " << std::endl;
    CombinedSolverParameters params;
    params.numIter = 2; //5;
    params.nonLinearIter = 4;
    params.linearIter = 32;//250;
    params.useOpt = true;
    params.useOptLM = false;

	std::string targetSourceDirectory = "../data/upper_body_depth";

	assert(argc <= 5);
    if (argc > 1) params.numIter = atoi(argv[1]);
    //if (argc > 2) params.nonLinearIter = atoi(argv[2]);  
	int num_frames;  
    if (argc > 2) num_frames = atoi(argv[2]);
	if (argc > 3) params.nonLinearIter = atoi(argv[3]);
    if (argc > 4) params.linearIter = atoi(argv[4]);

    //std::string sourceFilename = "../data/upperbody.ply";  
	std::string sourceFilename = "../data/upper_body_depth/frame-000000.depth.png";

	// volume(x_length, y_length, z_length, origin, voxel_size, subgrid_scale)
	float3 voxel_size = make_float3(1.0f/361.0f, 1.0f/376.0f, 0.4f/61.0f);
	//float3 voxel_size = make_float3(0.6f/361.0f, 0.7f/376.0f, 0.25f/61.0f);
	//TSDFVolume volume(361,376,71, make_float3(-0.5611f,-0.4208f, 0.65f), voxel_size);
	TSDFVolume volume(361,376,61, make_float3(-0.5611f,-0.4208f, 0.65f), voxel_size, make_int3(15,15,20));
	//TSDFVolume volume(361,376,61, make_float3(-0.36f,-0.26f, 0.79f), voxel_size);

	float mat_K[3 * 3] = {570.342, 0, 320,  0, 570.342, 240,  0, 0, 1};
	float world2cam[4 * 4] = {1, 0, 0, 0, 0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1};

	int im_width = 640; int im_height = 480;
	float* depth_im = new float[im_height * im_width];
	ReadDepth(sourceFilename, im_height, im_width, depth_im);
	volume.Integrate(depth_im, mat_K, world2cam, 1);
	std::vector<float3> vertices;
	std::vector<int3> triangles;
	std::vector<float3> normals;
	std::vector<int3> vol_idx;
	std::vector<float3> rel_coors;
	
	extract_surface(volume, vertices, triangles, normals, vol_idx, rel_coors);
	write_to_ply("../output_mesh/after_integration0.ply",vertices,triangles);
	//write_to_ply("../output_mesh/depth30.ply",vertices,triangles);

	CombinedSolver * solver;
	SimpleMesh * sourceMesh;
	std::string target_name;
    //std::vector<std::string> targetFiles = ml::Directory::enumerateFiles(targetSourceDirectory);
	std::vector<std::string> targetFiles;
	std::vector<std::string> target_set;
	std::stringstream ss;
	SimpleMesh* res;

	for (int i=1; i<num_frames; i++) {
		ss << std::setw(6) << std::setfill('0') << i;
		target_set.push_back("../data/upper_body_depth/frame-"+ss.str()+".depth.png");
		ss.str("");
	}
	//target_set.push_back("../data/upper_body_depth/frame-000015.depth.png");
	//target_set.push_back("../data/upper_body_depth/frame-000030.depth.png");
	
	sourceMesh = createMesh("../output_mesh/after_integration"+std::to_string(0)+".ply");
	solver = new CombinedSolver(target_set, params, &volume, &vertices, &normals, &triangles, &vol_idx, &rel_coors);
	//solver->solveAll();

	// record the processing time
    std::clock_t start;
    double duration;
    start = std::clock();		
    
	for (int i=0; i<target_set.size(); i++) {
		//sourceMesh = createMesh("../output_mesh/after_integration"+std::to_string(i)+".ply");
		target_name = target_set[i];
		targetFiles.clear();
		targetFiles.push_back(target_name);
		std::cout << "target mesh: " << targetFiles[0] << std::endl;
		//solver = new CombinedSolver(sourceMesh, targetFiles, params, &volume, &vertices, &normals, &triangles);
		solver->set_targets(targetFiles);
		solver->set_vnt(&vertices, &normals, &triangles, &vol_idx);
		solver->solveAll();
		// test
		//res = solver->result();
		//OpenMesh::IO::write_mesh(*res, "../output_mesh/out"+std::to_string(i)+".ply");

		solver->update_grid();
		volume.Upsample(*(solver->get_grid()), solver->get_grid_dims());
		ReadDepth(target_name, im_height, im_width, depth_im);
		volume.Integrate(depth_im, mat_K, world2cam, 0);
		vertices.clear(); triangles.clear(); normals.clear(); vol_idx.clear(); rel_coors.clear();
		extract_surface(volume, vertices, triangles, normals, vol_idx, rel_coors);
		//write_to_ply("../output_mesh/after_integration"+std::to_string(i+1)+".ply",vertices,triangles);
		//solver->saveGraphResults();
		//delete sourceMesh;
		//delete solver;
	}
	// output the processing time
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout<<std::endl<<"total time: "<< duration << " seconds" <<std::endl;
    
	//solver->saveGraphResults();
	write_to_ply("../output_mesh/after_integration_final.ply",vertices,triangles);
	delete sourceMesh;
	delete solver;
	delete depth_im;
	return 0;
}
