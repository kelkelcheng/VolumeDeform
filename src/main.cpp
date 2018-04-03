#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"

#include "utils.hpp"
#include "TSDFVolumn.h"
#include "MarchingCubes.h"

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

	assert(argc <= 4);
    if (argc > 1) params.numIter = atoi(argv[1]);
    if (argc > 2) params.nonLinearIter = atoi(argv[2]);    
    if (argc > 3) targetSourceDirectory = argv[3];

    
    //std::string sourceFilename = "../data/upperbody.ply";  
	std::string sourceFilename = "../data/upper_body_depth/frame-000000.depth.png";

	//float3 voxel_size = make_float3(0.006f, 0.006f, 0.012f);
	float3 voxel_size = make_float3(1.0f/361.0f, 1.0f/376.0f, 0.4f/61.0f);
  	//TSDFVolumn volumn(500,520,80, voxel_size);
	TSDFVolumn volumn(361,376,61, make_float3(-0.5611f,-0.4208f, 0.65f), voxel_size);

	float mat_K[3 * 3] = {570.342, 0, 320,  0, 570.342, 240,  0, 0, 1};
	float world2cam[4 * 4] = {1, 0, 0, 0, 0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1};

	int im_width = 640; int im_height = 480;
	float* depth_im = new float[im_height * im_width];
	ReadDepth(sourceFilename, im_height, im_width, depth_im);
	volumn.Integrate(depth_im, mat_K, world2cam);
	std::vector<float3> vertices ;
	std::vector<int3> triangles;
	extract_surface(volumn, vertices, triangles);
	write_to_ply("../output_mesh/after_integration0.ply",vertices,triangles);

	CombinedSolver * solver;
	SimpleMesh * sourceMesh;
	std::string target_name;
    //std::vector<std::string> targetFiles = ml::Directory::enumerateFiles(targetSourceDirectory);
	std::vector<std::string> targetFiles;
	std::vector<std::string> target_set;
	target_set.push_back("../data/upper_body_depth/frame-000002.depth.png");
	target_set.push_back("../data/upper_body_depth/frame-000004.depth.png");
	target_set.push_back("../data/upper_body_depth/frame-000006.depth.png");
	target_set.push_back("../data/upper_body_depth/frame-000008.depth.png");
	target_set.push_back("../data/upper_body_depth/frame-000010.depth.png");
	for (int i=0; i<target_set.size(); i++) {
		sourceMesh = createMesh("../output_mesh/after_integration"+std::to_string(i)+".ply");
		target_name = target_set[i];
		targetFiles.clear();
		targetFiles.push_back(target_name);
		solver = new CombinedSolver(sourceMesh, targetFiles, params, &volumn);
		solver->solveAll();
		solver->update_grid();
		volumn.Upsample(*(solver->get_grid()), solver->get_grid_dims());
		ReadDepth(target_name, im_height, im_width, depth_im);
		volumn.Integrate(depth_im, mat_K, world2cam);
		vertices.clear(); triangles.clear();
		extract_surface(volumn, vertices, triangles);
		write_to_ply("../output_mesh/after_integration"+std::to_string(i+1)+".ply",vertices,triangles);
		//solver->saveGraphResults();
		delete sourceMesh;
		delete solver;
	}

	delete depth_im;
	return 0;
}
