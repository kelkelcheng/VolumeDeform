#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"

//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>

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

static std::vector<int4> getSourceTetIndices(std::string filename) {
    // TODO: error handling
    std::ifstream inFile(filename);
    int tetCount = 0;
    int temp;
    inFile >> tetCount >> temp >> temp;
    std::vector<int4> tets(tetCount);
    for (int i = 0; i < tetCount; ++i) {
        inFile >> temp >> tets[i].x >> tets[i].y >> tets[i].z >> tets[i].w;
    }
    int4 f = tets[tets.size() - 1];
    printf("Final tet read: %d %d %d %d\n", f.x, f.y, f.z, f.w);
    return tets;
}

int main(int argc, const char * argv[])
{
	std::cout << "Function begins: " << std::endl;
	//cv::Mat depth_mat = cv::imread("frame-000132.depth.png", CV_LOAD_IMAGE_UNCHANGED);


    std::string targetSourceDirectory = "/home/kel/Optlang/Opt/examples/data/squat_test"; //"../data/squat_target"; 
    std::string sourceFilename = "../data/upperbody.ply";  //"../data/squat_source2.obj";
    std::string tetmeshFilename = "/home/kel/Optlang/Opt/examples/data/squat_tetmesh_test.ele"; //"../data/squat_tetmesh.ele"

    CombinedSolverParameters params;
    params.numIter = 10; //5;
    params.nonLinearIter = 5;
    params.linearIter = 50;//250;
    params.useOpt = true;
    params.useOptLM = false;

    if (argc > 1) {
        assert(argc > 3);
        //targetSourceDirectory = argv[1];
        //sourceFilename = argv[2];
        //tetmeshFilename = argv[3];
		params.numIter = atoi(argv[1]);
		params.nonLinearIter = atoi(argv[2]);
		params.linearIter = atoi(argv[3]);
    }

    //std::vector<std::string> targetFiles = ml::Directory::enumerateFiles(targetSourceDirectory);
    std::vector<std::string> targetFiles;
	targetFiles.push_back("mesh_test.obj");
	//targetFiles.push_back("mesh_0048.obj");



    SimpleMesh* sourceMesh = createMesh(sourceFilename);

    std::vector<SimpleMesh*> targetMeshes;
	int targetCount = 0;
    for (auto target : targetFiles) {
        std::cout << targetSourceDirectory + "/" + target << std::endl;
        targetMeshes.push_back(createMesh(targetSourceDirectory + "/" + target));
		++targetCount;
    }
    std::cout << "All meshes now in memory: " << targetCount << std::endl;

    /*CombinedSolverParameters params;
    params.numIter = 10; //5;
    params.nonLinearIter = 5;
    params.linearIter = 50;//250;
    params.useOpt = true;
    params.useOptLM = false;*/

    CombinedSolver solver(sourceMesh, targetMeshes, params);
	std::cout << "Begin to solve" << std::endl;
    solver.solveAll();
    SimpleMesh* res = solver.result();
    //solver.saveGraphResults();

	if (!OpenMesh::IO::write_mesh(*res, "../output_mesh/out.ply"))
	{
	    std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}
    
    for (SimpleMesh* mesh : targetMeshes) {
        delete mesh;
    }
    delete sourceMesh;

	return 0;
}
