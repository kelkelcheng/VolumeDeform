#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"


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

    std::string targetSourceDirectory = "../data/upper_body_depth";
    std::string sourceFilename = "../data/upperbody.ply";  

    CombinedSolverParameters params;
    params.numIter = 5; //5;
    params.nonLinearIter = 5;
    params.linearIter = 50;//250;
    params.useOpt = true;
    params.useOptLM = false;

	assert(argc <= 5);
    if (argc > 1) params.numIter = atoi(argv[1]);
    if (argc > 2) params.nonLinearIter = atoi(argv[2]);    
    if (argc > 3) targetSourceDirectory = argv[3];
	if (argc > 4) sourceFilename = argv[4];
    
    std::vector<std::string> targetFiles = ml::Directory::enumerateFiles(targetSourceDirectory);
	for (int i=0; i<targetFiles.size(); i++) {
		targetFiles[i] = targetSourceDirectory + "/" + targetFiles[i];
		std::cout<<targetFiles[i]<<std::endl;
	}

    SimpleMesh* sourceMesh = createMesh(sourceFilename);


    CombinedSolver solver(sourceMesh, targetFiles, params);
	std::cout << "Begin to solve" << std::endl;
    solver.solveAll();
    //SimpleMesh* res = solver.result();
    //solver.saveGraphResults();

    delete sourceMesh;

	return 0;
}
