#ifndef CombinedSolver_h
#define CombinedSolver_h

#include "mLibInclude.h"

#include <cuda_runtime.h>

#include "CombinedSolverParameters.h"
#include "CombinedSolverBase.h"
#include "SolverIteration.h"

#include "OpenMesh.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <math.h>
#include "TSDFVolume.h"
//#include "utils.hpp"
#include "MarchingCubes.h"

#include <assert.h> 
//#include "cudaUtil.h"
// just for testing purpose
//extern "C" void initAngle(void* ptr_angles, unsigned int m_nNodes);

static float clamp(float v, float mn, float mx) {
    return std::max(mn,std::min(v, mx));
}

class CombinedSolver : public CombinedSolverBase
{

	public:
        CombinedSolver(const std::vector<std::string> targetFiles, CombinedSolverParameters params, TSDFVolume* volume, std::vector<float3>* vertices, std::vector<float3>* normals, std::vector<int3>* triangles, std::vector<int3>* vol_idx, std::vector<float3>* rel_coors)
		{
            m_combinedSolverParameters = params;
			m_targets = targetFiles;

			m_volume = volume;
			m_vertices = vertices;
			m_normals = normals;
			m_triangles = triangles;
			m_vol_idx = vol_idx;
			m_rel_coors = rel_coors;

			dim3 vol_size = m_volume->get_size();
			
			m_scale = m_volume->get_scale();
			
			std::cout << "vol_size.x "<<vol_size.x<<" m_scale.x "<<m_scale.x<<std::endl;
			assert(((vol_size.x - 1) % m_scale.x) == 0);
			assert(((vol_size.y - 1) % m_scale.y) == 0);
			assert(((vol_size.z - 1) % m_scale.z) == 0);
			
			// sparse subgrid dimension
			// should equal to (25, 26, 4) in this case
			m_gridDims.x = (vol_size.x - 1) / m_scale.x + 1;
			m_gridDims.y = (vol_size.y - 1) / m_scale.y + 1;
			m_gridDims.z = (vol_size.z - 1) / m_scale.z + 1;		
            //m_gridDims = make_int3(25, 26, 4);

            m_dims      = m_gridDims;
			m_dims.x--; m_dims.y--; m_dims.z--;
			m_nNodes    = (m_dims.x + 1)*(m_dims.y + 1)*(m_dims.z + 1);

            m_M = (*m_vertices).size();
			m_vertexToVoxels.resize(m_M);
            m_relativeCoords.resize(m_M);
                     
			std::cout << "dim.x: " << m_dims.x <<" dim.y: "<<m_dims.y<<" dim.z: " <<m_dims.z << std::endl;

            std::vector<unsigned int> dims = { m_M };
			std::vector<unsigned int> dims_grid = { (uint)m_nNodes };

			// initialize Opt optimizer variables
            m_vertexPosTargetFloat3     = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
			m_vertexNormalTargetFloat3  = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
            m_robustWeights             = createEmptyOptImage(dims, OptImage::Type::FLOAT, 1, OptImage::Location::GPU, true);
            m_gridPosFloat3             = createEmptyOptImage(dims_grid, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
            m_gridPosFloat3Urshape      = createEmptyOptImage(dims_grid, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
            m_anglesFloat3              = createEmptyOptImage(dims_grid, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
			m_triWeights              	= createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);

			// initialize graph variables in Opt
			initializeWarpGrid();		
			//grid_state.resize(vol_size.x * vol_size.y * vol_size.z);

			addOptSolvers(dims, "opt_rotation.t", m_combinedSolverParameters.optDoublePrecision);
            addOptSolvers(dims, "opt_position.t", m_combinedSolverParameters.optDoublePrecision);
		} 

        virtual void combinedSolveInit() override {
            m_weightFit = 10.0f; 
            m_weightRegMax = 16.0f; //16 64
            
            m_weightRegMin = 10.0f; //10 32
            m_weightRegFactor = 0.9f; //0.95f

            m_weightReg = m_weightRegMax;

            m_functionTolerance = 0.0000001f;

            m_fitSqrt = sqrt(m_weightFit); //1.0f
            m_regSqrt = sqrt(m_weightReg); //3.0f

            m_problemParams.set("w_fitSqrt", &m_fitSqrt);//Sqrt
            m_problemParams.set("w_regSqrt", &m_regSqrt);//Sqrt

            m_problemParams.set("Offset", m_gridPosFloat3); 
            m_problemParams.set("Angle", m_anglesFloat3);
            m_problemParams.set("RobustWeights", m_robustWeights);
            m_problemParams.set("UrShape", m_gridPosFloat3Urshape); 
            m_problemParams.set("Constraints", m_vertexPosTargetFloat3);
			m_problemParams.set("ConstraintNormals", m_vertexNormalTargetFloat3); 
			m_problemParams.set("TriWeights", m_triWeights);
			m_problemParams.set("G", m_graph);
			m_problemParams.set("RegGrid", m_regGrid);

            m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
            m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
            m_solverParams.set("function_tolerance", &m_functionTolerance);
        }

// function that will be called outside
// after initialization, it mainly calls singleSolve
        virtual void solveAll() override {
            resetGPUMemory();
            combinedSolveInit();

			m_problemParams.set("G", m_graph); // check why it is required to set it again
			m_problemParams.set("RegGrid", m_regGrid);

            for (m_targetIndex = 0; m_targetIndex < m_targets.size(); ++m_targetIndex) {
                if (m_targetIndex!=0) resetGPUMemory();
				singleSolve(m_solverInfo[0], m_solverInfo[1]); 
            }

            combinedSolveFinalize();
        }

/** inside singleSolve
preSingleSolve();
	for i:numIter
		preNonlinearSolve(i);
		s0.solver->solve // rotation
		s1.solver->solve // position
		postNonlinearSolve(i);
**/ 			
		virtual void singleSolve(SolverInfo s0, SolverInfo s1) {
		    preSingleSolve();
		    if (m_combinedSolverParameters.numIter == 1) {
		        preNonlinearSolve(0);
				std::cout << "------------ Position ------------" << std::endl;
				s1.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve, s1.iterationInfo);
		        postNonlinearSolve(0);
		        
		        preNonlinearSolve(0);
		        std::cout << "------------ Rotation ------------" << std::endl;
		        s0.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve, s0.iterationInfo);
				//postNonlinearSolve(0);
				//std::cout << "------------ Position ------------" << std::endl;
				s1.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve, s1.iterationInfo);
		        postNonlinearSolve(0);
		    }

		    else {
				std::cout << "------------ Position ------------" << std::endl;
	            preNonlinearSolve(0);
				s1.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve, s1.iterationInfo);
	            postNonlinearSolve(0);

		        for (int i = 0; i < (int)m_combinedSolverParameters.numIter; ++i) {
		            std::cout << "//////////// ITERATION" << i << "  (" << s0.name << ") ///////////////" << std::endl;
		            //s.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve, s.iterationInfo);
				    std::cout << "------------ Rotation ------------" << std::endl;
		            preNonlinearSolve(i);
				    s0.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve, s0.iterationInfo);

					std::cout << "------------ Position ------------" << std::endl;
					s1.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve, s1.iterationInfo);
		            postNonlinearSolve(i);

		            if (m_combinedSolverParameters.earlyOut || m_endSolveEarly) {
		                m_endSolveEarly = false;
		                break;
		            }
		        }
		    }
		    postSingleSolve();
		}

        virtual void preSingleSolve() override {
            unsigned int M = (unsigned int)(*m_vertices).size();//m_initial.n_vertices();
            m_previousConstraints.resize(M);
            for (int i = 0; i < (int)M; ++i) {
                m_previousConstraints[i] = make_float3(0, 0, -90901283092183);
            }
            m_weightReg = m_weightRegMax;
        }
        
        virtual void postSingleSolve() override { 
            //char buff[100];
            //sprintf(buff, "../output_mesh/out_%04d.ply", m_targetIndex);
            //saveCurrentMesh(buff);        
        }

        virtual void preNonlinearSolve(int) override {
            m_timer.start();
            int newConstraintCount = setConstraints(0.022, 0.2, 0.8); //0.03, 0.2, 0.8
            m_timer.stop();
            double setConstraintsTime = m_timer.getElapsedTime();
            std::cout << "-- Set Constraints: " << setConstraintsTime << "s";
            std::cout << " -------- New constraints: " << newConstraintCount << std::endl;
            if (newConstraintCount <= 5) {
                std::cout << " -------- Few New Constraints" << std::endl;
                if (m_weightReg != m_weightRegMin) {
                    std::cout << " -------- Skipping to min reg weight" << std::endl;
                    m_weightReg = m_weightRegMin;
                }
                m_endSolveEarly = true;
            }
            m_regSqrt = sqrtf(m_weightReg);
        }
        virtual void postNonlinearSolve(int) override {
            m_timer.start();
            copyResultToCPUFromFloat3();
            m_timer.stop();
            double copyTime = m_timer.getElapsedTime();
            std::cout << "--Copy to CPU : " << copyTime << "s " << std::endl;
            m_weightReg = fmaxf(m_weightRegMin, m_weightReg*m_weightRegFactor);
        }

        virtual void combinedSolveFinalize() override {
            //reportFinalCosts("Robust Mesh Deformation", m_combinedSolverParameters, getCost("Opt(GN)"), getCost("Opt(LM)"), nan(""));
        }

		void update_depth() {
			int H = 480; 
			int W = 640;
			cv::Mat depth_mat = cv::imread(m_targets[0], CV_LOAD_IMAGE_UNCHANGED); //m_targetIndex
			std::cout << "read depth: " << m_targets[0] << std::endl; //m_targetIndex
			for (int r = 0; r < H; ++r)
				for (int c = 0; c < W; ++c) {
					m_depth_mat[r * W + c] = (float)(depth_mat.at<unsigned short>(r, c)) / 1000.0f;
			}
		}

		int getIndex1D(int3 idx)
        {
			return idx.z * (m_gridDims.y * m_gridDims.x) + idx.y * m_gridDims.x + idx.x;
		}

		/*void vecTrans(float3 &v, float K[4][4])
		{
			float x = K[0][0] * v.x + K[0][1] * v.y + K[0][2] * v.z + K[0][3];
			float y = K[1][0] * v.x + K[1][1] * v.y + K[1][2] * v.z + K[1][3];
			float z = K[2][0] * v.x + K[2][1] * v.y + K[2][2] * v.z + K[2][3];
			v.x = x; v.y = y; v.z = z;
		}*/
		
        int setConstraints(float positionThreshold, float cosNormalThreshold, float viewThreshold); //0.03 0.2 0.8

		/*void computeBoundingBox()
		{
			m_min = make_float3(+std::numeric_limits<float>::max(), +std::numeric_limits<float>::max(), +std::numeric_limits<float>::max());
			m_max = make_float3(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
			for (int i=0; i<m_nNodes; i++){
				float3 pt = h_gridVertexPosFloat3[i];
				m_min.x = fmin(m_min.x, pt.x); m_min.y = fmin(m_min.y, pt.y); m_min.z = fmin(m_min.z, pt.z);
				m_max.x = fmax(m_max.x, pt.x); m_max.y = fmax(m_max.y, pt.y); m_max.z = fmax(m_max.z, pt.z);
			}
			m_delta = (m_max - m_min); m_delta.x /= (m_dims.x); m_delta.y /= (m_dims.y); m_delta.z /= (m_dims.z);
			
			std::cout<<"m_min.x: "<<m_min.x<<" m_min.y: "<<m_min.y<<" m_min.z: "<<m_min.z <<std::endl;
			std::cout<<"m_max.x: "<<m_max.x<<" m_max.y: "<<m_max.y<<" m_max.z: "<<m_max.z <<std::endl;
			std::cout<<"m_delta.x: "<<m_delta.x<<" m_delta.y: "<<m_delta.y<<" m_delta.z: "<<m_delta.z <<std::endl;
		}*/
		

		// only called once in the constructor
		// setup the the regularization graph
		void initializeWarpGrid()
		{
            std::vector<int> regGrid_v0;
            std::vector<int> regGrid_v1;

			h_gridVertexPosFloat3.resize(m_nNodes);
			
			// initialize sparse subgrid positions
			m_volume->InitSubGrid(h_gridVertexPosFloat3, m_gridDims);
			
			// setup the graph variable for Opt
			// notice that it is index-based
			for (int i = 0; i <= m_dims.x; i++)
			{
				for (int j = 0; j <= m_dims.y; j++)
				{
					for (int k = 0; k <= m_dims.z; k++)
					{
						int3 gridIdx = make_int3(i, j, k);

						if (k!=m_dims.z) { 
							regGrid_v0.push_back(getIndex1D(gridIdx)); 
							regGrid_v1.push_back(getIndex1D(gridIdx+make_int3( 0, 0, 1)));
						}

						if (j!=m_dims.y) { 
							regGrid_v0.push_back(getIndex1D(gridIdx)); 
							regGrid_v1.push_back(getIndex1D(gridIdx+make_int3( 0, 1, 0)));
						}

						if (i!=m_dims.x) { 
							regGrid_v0.push_back(getIndex1D(gridIdx)); 
							regGrid_v1.push_back(getIndex1D(gridIdx+make_int3( 1, 0, 0)));
						}

						if (k!=0) { 
							regGrid_v0.push_back(getIndex1D(gridIdx)); 
							regGrid_v1.push_back(getIndex1D(gridIdx+make_int3( 0, 0,-1)));
						}

						if (j!=0) { 
							regGrid_v0.push_back(getIndex1D(gridIdx)); 
							regGrid_v1.push_back(getIndex1D(gridIdx+make_int3( 0,-1, 0)));
						}

						if (i!=0) { 
							regGrid_v0.push_back(getIndex1D(gridIdx)); 
							regGrid_v1.push_back(getIndex1D(gridIdx+make_int3(-1, 0, 0)));
						}

					}
				}
			}

            m_regGrid = std::make_shared<OptGraph>(std::vector<std::vector<int> >({ regGrid_v0, regGrid_v1 }));
            m_gridPosFloat3->update(h_gridVertexPosFloat3);
            m_gridPosFloat3Urshape->update(h_gridVertexPosFloat3);
			cudaSafeCall(cudaMemset(m_anglesFloat3->data(), 0, sizeof(float3)*m_nNodes));
		}

		// this function will be called for each new depth frame
		// basically it update the Opt graph variables for each newly extracted mesh point
		void resetGPUMemory()
		{
			//update grid pos and angle for regularization
			//m_gridPosFloat3Urshape->update(h_gridVertexPosFloat3);
			//cudaSafeCall(cudaMemset(m_anglesFloat3->data(), 0, sizeof(float3)*m_nNodes));
			//computeBoundingBox(); //--edited: no need to add this line

	        std::vector<int> w;
			std::vector<int> v1; std::vector<int> v2; std::vector<int> v3; std::vector<int> v4;
			std::vector<int> v5; std::vector<int> v6; std::vector<int> v7; std::vector<int> v8; 

			vector<float3>& vertices = *m_vertices;

			m_M = vertices.size();
			m_vertexToVoxels.resize(m_M);
            m_relativeCoords.resize(m_M);
            
			for (int i=0; i<m_M; i++)
			{
				float3 pp = vertices[i];

				int3 pInt = (*m_vol_idx)[i];
				pInt.x = (int)pInt.x/m_scale.x; 
				pInt.y = (int)pInt.y/m_scale.y;
				pInt.z = (int)pInt.z/m_scale.z;
				
				assert(pInt.x < m_gridDims.x-1);
				assert(pInt.y < m_gridDims.y-1);
				assert(pInt.z < m_gridDims.z-1);
				
				m_vertexToVoxels[i] = pInt;
				m_relativeCoords[i] = (*m_rel_coors)[i];// can replace m_relativeCoords later on 
		
				w.push_back(i); 

				v1.push_back( getIndex1D(pInt + make_int3(0, 0, 0)) );
				v2.push_back( getIndex1D(pInt + make_int3(0, 0, 1)) );
				v3.push_back( getIndex1D(pInt + make_int3(0, 1, 0)) );
				v4.push_back( getIndex1D(pInt + make_int3(0, 1, 1)) );
				v5.push_back( getIndex1D(pInt + make_int3(1, 0, 0)) );
				v6.push_back( getIndex1D(pInt + make_int3(1, 0, 1)) );
				v7.push_back( getIndex1D(pInt + make_int3(1, 1, 0)) );
				v8.push_back( getIndex1D(pInt + make_int3(1, 1, 1)) );

			}

			m_triWeights->update(m_relativeCoords);
			m_graph = std::make_shared<OptGraph>(std::vector<std::vector<int> >({ w, v1, v2, v3, v4, v5, v6, v7, v8, w, w, w}));
		}

		std::vector<float3>* get_grid()
		{
			return &h_gridVertexPosFloat3;
		}

		int3 get_grid_dims()
		{
			return m_gridDims;
		}

		void update_grid()
		{
			m_gridPosFloat3->copyTo(h_gridVertexPosFloat3);
		}

		void set_targets(std::vector<std::string> targets)
		{
			m_targets = targets;
		}

		void set_depth_mat(float* depth_mat) {
			m_depth_mat = depth_mat;
		}
		
		void set_vnt(std::vector<float3>* vertices, std::vector<float3>* normals, std::vector<int3>* triangles, std::vector<int3>* vol_idx) {
			m_vertices = vertices;
			m_normals = normals;
			m_triangles = triangles;
			m_vol_idx = vol_idx;
		}

		void copyResultToCPUFromFloat3();


//The following three functions are used to generate the deformed grid
		void GraphAddSphere(SimpleMesh& out, SimpleMesh::Point offset, float scale, SimpleMesh meshSphere, bool constrained)
		{
			unsigned int currentN = (unsigned int)out.n_vertices();
			for (unsigned int i = 0; i < meshSphere.n_vertices(); i++)
			{
				SimpleMesh::Point p = meshSphere.point(VertexHandle(i))*scale + offset;
				VertexHandle vh = out.add_vertex(p);
				out.set_color(vh, SimpleMesh::Color(200, 0, 0));
			}

			for (unsigned int k = 0; k < meshSphere.n_faces(); k++)
			{
				std::vector<VertexHandle> vhs;
				for (SimpleMesh::FaceVertexIter v_it = meshSphere.fv_begin(FaceHandle(k)); v_it != meshSphere.fv_end(FaceHandle(k)); ++v_it)
				{
					vhs.push_back(VertexHandle(v_it->idx() + currentN));
				}
				out.add_face(vhs);
			}
		}

		void GraphAddCone(SimpleMesh& out, SimpleMesh::Point offset, float scale, float3 direction, SimpleMesh meshCone)
		{
			unsigned int currentN = (unsigned int)out.n_vertices();
			for (unsigned int i = 0; i < meshCone.n_vertices(); i++)
			{
				SimpleMesh::Point pO = meshCone.point(VertexHandle(i));

				vec3f o(0.0f, 0.5f, 0.0f);
				mat3f s = ml::mat3f::diag(scale, length(direction), scale);
				vec3f p(pO[0], pO[1], pO[2]);

				p = s*(p + o);

				vec3f f(direction.x, direction.y, direction.z); f = ml::vec3f::normalize(f);
				vec3f up(0.0f, 1.0f, 0.0f);
				vec3f axis  = ml::vec3f::cross(up, f);
				float angle = acos(ml::vec3f::dot(up, f))*180.0f/(float)PI;
				mat3f R = ml::mat3f::rotation(axis, angle); if (axis.length() < 0.00001f) R = ml::mat3f::identity();
				
				p = R*p;

				VertexHandle vh = out.add_vertex(SimpleMesh::Point(p.x, p.y, p.z) + offset);
				out.set_color(vh, SimpleMesh::Color(70, 200, 70));
			}

			for (unsigned int k = 0; k < meshCone.n_faces(); k++)
			{
				std::vector<VertexHandle> vhs;
				for (SimpleMesh::FaceVertexIter v_it = meshCone.fv_begin(FaceHandle(k)); v_it != meshCone.fv_end(FaceHandle(k)); ++v_it)
				{
					vhs.push_back(VertexHandle(v_it->idx() + currentN));
				}
				out.add_face(vhs);
			}
		}

		void saveGraph(const std::string& filename, float3* data, unsigned int N, float scale, SimpleMesh meshSphere, SimpleMesh meshCone)
		{
			SimpleMesh out;

			std::vector<float3> h_gridPosFloat3(m_nNodes);
            m_gridPosFloat3->copyTo(h_gridPosFloat3);
			for (unsigned int i = 0; i < N; i++)
			{
				if (h_gridPosFloat3[i].x != -std::numeric_limits<float>::infinity())
				{
					GraphAddSphere(out, SimpleMesh::Point(data[i].x, data[i].y, data[i].z), scale*2.0f, meshSphere, h_gridPosFloat3[i].x != -std::numeric_limits<float>::infinity());
				}
			}

			for (int i = 0; i <= m_dims.x; i++)
			{
				for (int j = 0; j <= m_dims.y; j++)
				{
					for (int k = 0; k <= m_dims.z; k++)
					{
						float3 pos0 = data[getIndex1D(make_int3(i, j, k))];
						
						if (i + 1 <= m_dims.x) { float3 dir1 = data[getIndex1D(make_int3(i + 1, j, k))] - pos0; GraphAddCone(out, SimpleMesh::Point(pos0.x, pos0.y, pos0.z), scale, dir1, meshCone); }
						if (j + 1 <= m_dims.y) { float3 dir2 = data[getIndex1D(make_int3(i, j + 1, k))] - pos0; GraphAddCone(out, SimpleMesh::Point(pos0.x, pos0.y, pos0.z), scale, dir2, meshCone); }
						if (k + 1 <= m_dims.z) { float3 dir3 = data[getIndex1D(make_int3(i, j, k + 1))] - pos0; GraphAddCone(out, SimpleMesh::Point(pos0.x, pos0.y, pos0.z), scale, dir3, meshCone); }
					}
				}
			}

			OpenMesh::IO::write_mesh(out, filename, IO::Options::VertexColor);
		}

		void saveGraphResults()
		{
			SimpleMesh meshSphere;
			if (!OpenMesh::IO::read_mesh(meshSphere, "../data/sphere.ply"))
			{
				std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
				exit(1);
			}

			SimpleMesh meshCone;
			if (!OpenMesh::IO::read_mesh(meshCone, "../data/cone.ply"))
			{
				std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
				exit(1);
			}

			std::vector<float3> h_gridPosUrshapeFloat3(m_nNodes);
            m_gridPosFloat3Urshape->copyTo(h_gridPosUrshapeFloat3);   saveGraph("../output_mesh/grid.ply", h_gridPosUrshapeFloat3.data(), m_nNodes, 0.001f, meshSphere, meshCone);

            std::vector<float3> h_gridPosFloat3(m_nNodes);
            m_gridPosFloat3->copyTo(h_gridPosFloat3);
			saveGraph("../output_mesh/gridOut.ply", h_gridPosFloat3.data(), m_nNodes, 0.001f, meshSphere, meshCone);
		}

	private:
		unsigned int m_nNodes; //number of grid points
        unsigned int m_M; //number of mesh points
		TSDFVolume* m_volume;
		std::vector<float3>* m_vertices;
		std::vector<float3>* m_normals;
		std::vector<int3>* m_triangles;
		std::vector<int3>* m_vol_idx;
		std::vector<float3>* m_rel_coors;
		//std::vector<unsigned int> grid_state; // only used in CPU version
	
		int3 m_scale;
			
		//float3 m_min;
		//float3 m_max;

		int3   m_dims;
		int3   m_gridDims;
		//float3 m_delta;

		std::vector<int3>   m_vertexToVoxels; // 3d index in sparse subgrid
        std::vector<float3> m_relativeCoords; // trilinear weights in sparse subgrid

        ml::Timer m_timer;
		std::vector<std::string> m_targets;
        std::vector<float3> m_previousConstraints;

        // Current index in solve
        uint m_targetIndex;

		std::shared_ptr<OptImage> m_anglesFloat3;
		std::shared_ptr<OptImage> m_vertexPosTargetFloat3;
		std::shared_ptr<OptImage> m_vertexNormalTargetFloat3;
		std::shared_ptr<OptImage> m_gridPosFloat3; 
		std::shared_ptr<OptImage> m_gridPosFloat3Urshape;
        std::shared_ptr<OptImage> m_robustWeights;
		std::shared_ptr<OptImage> m_triWeights;
		std::shared_ptr<OptGraph> m_graph;
		std::shared_ptr<OptGraph> m_regGrid;	
		
        float m_weightFit;
        float m_weightRegMax;
        float m_weightRegMin;
        float m_weightRegFactor;
        float m_weightReg;
        float m_functionTolerance;
        float m_fitSqrt;
        float m_regSqrt;

		int max_threads = 512;

		float * m_depth_mat;

		/*float mat_K[4][4] =
		{
			{570.342, 0, 320, 0},
			{0, 570.342, 240, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1}
		};

		float mat_K_inv[4][4] =
		{
			{0.00175333, 0, -0.561066, 0}, //
			{0, 0.00175333, -0.4208001, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1}
		};*/
		float K_mat[3 * 3] = {570.342, 0, 320,  0, 570.342, 240,  0, 0, 1};
		float K_inv_mat[3 * 3] = {0.00175333, 0, -0.561066,  0, 0.00175333, -0.4208001,  0, 0, 1};
		std::vector<float3> h_gridVertexPosFloat3;
};
#endif
