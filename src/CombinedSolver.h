#ifndef CombinedSolver_h
#define CombinedSolver_h

#include <nanoflann/include/nanoflann.hpp>
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

#include "cudaUtil.h"
// just for testing purpose
//extern "C" void initAngle(void* ptr_angles, unsigned int m_nNodes);

using namespace nanoflann;
// For nanoflann computation
struct PointCloud_nanoflann {
    std::vector<float3>  pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline float kdtree_distance(const float *p1, const size_t idx_p2, size_t /*size*/) const
    {
        const float d0 = p1[0] - pts[idx_p2].x;
        const float d1 = p1[1] - pts[idx_p2].y;
        const float d2 = p1[2] - pts[idx_p2].z;
        return d0*d0 + d1*d1 + d2*d2;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, int dim) const
    {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};


typedef KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<float, PointCloud_nanoflann>,
    PointCloud_nanoflann,
    3 /* dim */
> NanoKDTree;
#define MAX_K 20

static float clamp(float v, float mn, float mx) {
    return std::max(mn,std::min(v, mx));
}

class CombinedSolver : public CombinedSolverBase
{

	public:
        CombinedSolver(	const std::vector<std::string> targetFiles, 
						CombinedSolverParameters params, 
						TSDFVolume* volume, 
						std::vector<float3>* vertices, 
						std::vector<float3>* normals, 
						std::vector<int3>* triangles, 
						std::vector<int3>* vol_idx, 
						std::vector<float3>* rel_coors, 
						float K[9], float K_inv[9],
						int H, int W,
						bool is_perspective=true
					)
		{
            m_combinedSolverParameters = params;
			m_targets = targetFiles;

            memcpy(K_mat, K, 9*sizeof(float));
            memcpy(K_inv_mat, K_inv, 9*sizeof(float));     

			im_h = H;
			im_w = W;
			m_is_perspective = is_perspective;

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
			m_gridPosFloat3             = createEmptyOptImage(dims_grid, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
            m_anglesFloat3              = createEmptyOptImage(dims_grid, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
			m_robustWeights             = createEmptyOptImage(dims, OptImage::Type::FLOAT, 1, OptImage::Location::GPU, true);
			m_gridPosFloat3Urshape      = createEmptyOptImage(dims_grid, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
			m_vertexPosTargetFloat3     = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
			m_vertexNormalTargetFloat3  = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);          
			m_triWeights              	= createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);

			// initialize graph variables in Opt
			initializeWarpGrid();		
			grid_state.resize(vol_size.x * vol_size.y * vol_size.z);

			addOptSolvers(dims, "opt_formulation.t", m_combinedSolverParameters.optDoublePrecision);
			//addOptSolvers(dims, "opt_rotation.t", m_combinedSolverParameters.optDoublePrecision);
            //addOptSolvers(dims, "opt_position.t", m_combinedSolverParameters.optDoublePrecision);
		} 

        virtual void combinedSolveInit() override {
            m_weightFit = 10.0f; 
            m_weightRegMax = 32.0f; //16 64
            
            m_weightRegMin = 16.0f; //10 32
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
				//singleSolve(m_solverInfo[0], m_solverInfo[1]); 
				singleSolve(m_solverInfo[0]); 
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
		virtual void singleSolve(SolverInfo s0) {
		    preSingleSolve();
		
			for (int i = 0; i < (int)m_combinedSolverParameters.numIter; ++i) {
				std::cout << "//////////// ITERATION" << i << "  (" << s0.name << ") ///////////////" << std::endl;
				std::cout << "------------ Optimization ------------" << std::endl;
				preNonlinearSolve(i);
				s0.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve, s0.iterationInfo);
				postNonlinearSolve(i);

				if (m_combinedSolverParameters.earlyOut || m_endSolveEarly) {
					m_endSolveEarly = false;
					break;
				}
			}
		
		    postSingleSolve();
		}

        virtual void preSingleSolve() override {
            unsigned int M = (unsigned int)(*m_vertices).size();//m_initial.n_vertices();
			m_targetAccelerationStructure = generateAccelerationStructure(&m_vertices_2);

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
			//int H = 480; 
			//int W = 640;
			cv::Mat depth_mat = cv::imread(m_targets[0], CV_LOAD_IMAGE_UNCHANGED); //m_targetIndex
			std::cout << "read depth: " << m_targets[0] << std::endl; //m_targetIndex
			m_vertices_2.clear();
			
			int count = 0; //TODO remove later
			for (int r = 0; r < im_h; ++r) {
				for (int c = 0; c < im_w; ++c) {
					float z = (float)(depth_mat.at<unsigned short>(r, c)) / 2500.0f;// - 1.7;
					//float z = (float)(depth_mat.at<unsigned short>(r, c)) / 10000.0f;
					//float z = (float)(depth_mat.at<unsigned short>(r, c)) / 1000.0f * 1.1f - 11.0f;

					//if (count <= 50) {std::cout << "z: " << z << std::endl; count++;} //TODO remove later

					//if (z!=0) z += 0.45; //0.3  0.45
					m_depth_mat[r * im_w + c] = z;

					if (z >= 0.0 && z < 1.2) { //1.2
						float3 v;
						if (m_is_perspective) {
							v = make_float3(c*z, r*z, z);
						}
						else {
							v = make_float3(c, r, z);
						}
						vecTrans(v, K_inv_mat);
						m_vertices_2.push_back(v);
					}  
				}	
			}
		}


		float edgeFunction(const float3 &a, const float3 &b, const float3 &c) { 
			return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x); 
		} 

		void create_depthmap() {

			vector<float3>& vertices = *m_vertices;
			//float* zbuffer = new float[im_h * im_w];
			std::vector<float> zbuffer(im_h * im_w, 0);
			for (int i = 0; i < im_h * im_w; ++i) {
				zbuffer[i] = 0.0;
			}

			vector<int3>& triangles = *m_triangles;
			for (int i = 0; i < triangles.size(); i++) {
				int3 tri_idx = triangles[i];
				float3 v0 = vertices[tri_idx.x];
				float3 v1 = vertices[tri_idx.y];
				float3 v2 = vertices[tri_idx.z];

				vecTrans(v0, K_mat);
				vecTrans(v1, K_mat);
				vecTrans(v2, K_mat);

				float x0 = v0.x / v0.z;
				float x1 = v1.x / v1.z;
				float x2 = v2.x / v2.z;

				float y0 = v0.y / v0.z;
				float y1 = v1.y / v1.z;
				float y2 = v2.y / v2.z;

				int x_min = (int) std::floor(std::min({x0, x1, x2}));
				int x_max = (int) std::ceil(std::max({x0, x1, x2}));

				int y_min = (int) std::floor(std::min({y0, y1, y2}));
				int y_max = (int) std::ceil(std::max({y0, y1, y2}));

				x_min = std::max(0, std::min(im_w - 1, x_min));
				y_min = std::max(0, std::min(im_h - 1, y_min));
				x_max = std::max(0, std::min(im_w - 1, x_max));
				y_max = std::max(0, std::min(im_h - 1, y_max));

				float3 proj_v0 = make_float3(x0, y0, 1);
				float3 proj_v1 = make_float3(x1, y1, 1);
				float3 proj_v2 = make_float3(x2, y2, 1);

				float area = edgeFunction(proj_v0, proj_v1, proj_v2);

				for (int r = y_min; r <= y_max; ++r) {
					for (int c = x_min; c <= x_max; ++c) {
						//float z; 
						int idx = r*im_w+c;

						float3 proj_v = make_float3(c, r, 1);
						float w0 = edgeFunction(proj_v1, proj_v2, proj_v);
						float w1 = edgeFunction(proj_v2, proj_v0, proj_v);
						float w2 = edgeFunction(proj_v0, proj_v1, proj_v);
						if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
							w0 /= area;
							w1 /= area;
							w2 /= area;
							float z = 1.0 / (w0 * (1.0 / v0.z) + w1 * (1.0 / v1.z) + w2 * (1.0 / v2.z));
							if (z < zbuffer[idx] || zbuffer[idx]==0) {
								zbuffer[idx] = z;							
							}
						}
					}
				}
			}
			
			//int count = 0; //TODO remove later
			/*
			for (int r = 0; r < im_h; ++r) {
				for (int c = 0; c < im_w; ++c) {
					temp_mat[r*im_w+c] = make_float3(0.0, 0.0, 0.0);
					float shortest = 100.0;

					for (int i=0; i<vertices.size(); i++){
						float3 v = vertices[i];
						vecTrans(v, K_mat);

						float x = v.x / v.z;
						float y = v.y / v.z;

						//int x = (int) round(v.x / v.z);
						//int y = (int) round(v.y / v.z);

						if (std::abs(x-c)<1.001 && std::abs(y-r)<1.001) {
							float dis = length(make_float2((float)c, (float)r) - make_float2(x, y));
							if (dis < shortest) {
								shortest = dis;
								temp_mat[r*im_w+c] = v; // vertices[i];
							}
						}
						//if (y<im_h && x<im_w) {temp_mat[y*im_w+x] = vertices[i];}				
					}
				}  
			}
			*/
			/*
			for (int r = 0; r < im_h; ++r) {
				for (int c = 0; c < im_w; ++c) {
					temp_mat[r*im_w+c] = make_float3(0.0, 0.0, 0.0);
				}  
			}

			for (int i=0; i<vertices.size(); i++){
				float3 v = vertices[i];
				vecTrans(v, K_mat);
				int x = (int) round(v.x / v.z);
				int y = (int) round(v.y / v.z);
				if (y<im_h && x<im_w) {temp_mat[y*im_w+x] = vertices[i];}				
			}*/

			int H = im_h;
			int W = im_w;
			std::ofstream f ;
			f.open("../output_mesh/after_integration_depthmap.obj");
			int count = 0;
			float threshold = 0.1;
			if( f.is_open() ) {
				for (int r = 0; r < H; ++r) {
					for (int c = 0; c < W; ++c) {
						/*float3 vv = temp_mat[r*W+c];
						vv.x = (float)c * vv.z;
						vv.y = (float)r * vv.z;*/
						float v_z = zbuffer[r*W+c];
						float3 vv = make_float3((float)c * v_z, (float)r * v_z, v_z);
						vecTrans(vv, K_inv_mat);
						f << "v " << vv.y << " " << vv.x << " " << vv.z << std::endl;
					}
				}
				
				/*for (int i = 0; i < H-1; ++i) {
					for (int j = 0; j < W-1; ++j) {
						if (temp_mat[i*W+j].z > 0.0f) {
							f << "f " << i*W+j+1 << " " << i*W+j+2 << " " << (i+1)*W+j+1 << std::endl;
							f << "f " << (i+1)*W+(j+1)+1 << " " << (i+1)*W+j+1 << " " << i*W+j+1+1 << std::endl;
						}
					}
				}*/

				
				for (int r = 0; r < H-2; ++r) {
					for (int c = 0; c < W-2; ++c) {
						if( r<2 || c<2) continue;

						float z0 = zbuffer[(r-1)*W+c-1];
						float z1 = zbuffer[(r-1)*W+c];
						float z2 = zbuffer[(r-1)*W+c+1];
						float z3 = zbuffer[(r)*W+c-1];
						float z4 = zbuffer[(r)*W+c];
						float z5 = zbuffer[(r)*W+c+1];
						float z6 = zbuffer[(r+1)*W+c-1];
						float z7 = zbuffer[(r+1)*W+c];
						float z8 = zbuffer[(r+1)*W+c+1];

						//if (count<100 && z4>0) {std::cout << z0 <<" "<<z1<<" "<<z2<<" "<<z3<<" "<<z4<<std::endl;count++;}

						float dy_u_max = std::max({fabs(z0 - z3), fabs(z1 - z4), fabs(z2 - z5)});
						float dy_d_max = std::max({fabs(z0 - z6), fabs(z1 - z7), fabs(z2 - z8)});
						float dx_l_max = std::max({fabs(z0 - z1), fabs(z3 - z4), fabs(z6 - z7)});
						float dx_r_max = std::max({fabs(z0 - z2), fabs(z3 - z5), fabs(z6 - z8)});
					
						//if (count<100 && z4>0) {std::cout << z0 - z3 <<" "<<dy_d_max<<" "<<dx_l_max<<" "<<dx_r_max<<std::endl;count++;}

						//f << "f " << r*H+c+1 << " " << r*H+c+1+1 << " " << (r+1)*H+c+1 << std::endl;

						
						if (((dy_u_max<threshold) && (dy_d_max<threshold)) && ((dx_l_max<threshold) && (dx_r_max<threshold))) {
							//if (z0>0 && z1>0 && z2>0 && z3>0 && z4>0 && z5>0 && z6>0 && z7>0 && z8>0){
							if (z4>0 && z5>0 && z7>0){
								f << "f " << r*W+c+1 << " " << r*W+c+2 << " " << (r+1)*W+c+1 << std::endl;
								//f << "f " << (r+1)*W+(c+1)+1 << " " << (r+1)*W+c+1 << " " << r*W+c+1+1 << std::endl;
							}
							if (z5>0 && z7>0 && z8>0){
								//f << "f " << r*W+c+1 << " " << r*W+c+2 << " " << (r+1)*W+c+1 << std::endl;
								f << "f " << (r+1)*W+(c+1)+1 << " " << (r+1)*W+c+1 << " " << r*W+c+1+1 << std::endl;
							}										
						}
						
					}
				}

				// create depth map as png
				/*std::vector<float> depth_map_vec(im_h * im_w, 0);
				for (int i = 0; i < im_h * im_w; ++i) {
					depth_map_vec[i] = zbuffer[i] * 2500.0;
				}*/
				//cv::Mat depth_map = cv::Mat(depth_map_vec).reshape(0, im_h);
				cv::Mat depth_map = cv::Mat(zbuffer).reshape(0, im_h);
				depth_map *= 2500.0;
				//memcpy(depth_map.data, zbuffer);
				depth_map.convertTo( depth_map, CV_16UC1 );
				cv::imwrite( "../output_mesh/after_integration_depthmap.png", depth_map );
				//delete zbuffer;
			}
			f.close();	
		}

		int getIndex1D(int3 idx)
        {
			return idx.z * (m_gridDims.y * m_gridDims.x) + idx.y * m_gridDims.x + idx.x;
		}

		void vecTrans(float3 &v, float K[9])
		{
			float x = K[0] * v.x + K[1] * v.y + K[2] * v.z;
			float y = K[3] * v.x + K[4] * v.y + K[5] * v.z;
			float z = K[6] * v.x + K[7] * v.y + K[8] * v.z;
			v.x = x; v.y = y; v.z = z;
		}
		/*void vecTrans(float3 &v, float K[4][4])
		{
			float x = K[0][0] * v.x + K[0][1] * v.y + K[0][2] * v.z + K[0][3];
			float y = K[1][0] * v.x + K[1][1] * v.y + K[1][2] * v.z + K[1][3];
			float z = K[2][0] * v.x + K[2][1] * v.y + K[2][2] * v.z + K[2][3];
			v.x = x; v.y = y; v.z = z;
		}*/
		
		std::tuple<float3, bool> findNormal(float3 p, float* im, bool isPerspective, bool isWorldCoord);
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
			m_graph = std::make_shared<OptGraph>(std::vector<std::vector<int> >({ w, v1, v2, v3, v4, v5, v6, v7, v8}));
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
        std::unique_ptr<NanoKDTree> generateAccelerationStructure(std::vector<float3>* vertices) {
			int M = (*vertices).size();
			
            //assert(m_spuriousIndices.size() == m_noisyOffsets.size());
            m_pointCloud.pts.resize(M);
            for (unsigned int i = 0; i < M; i++)
            {   
                float3 p = (*vertices)[i];
                m_pointCloud.pts[i] = {p.x, p.y, p.z};
            }
            std::unique_ptr<NanoKDTree> tree = std::unique_ptr<NanoKDTree>(new NanoKDTree(3 /*dim*/, m_pointCloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */)));
            tree->buildIndex();
            return tree;
        }

		PointCloud_nanoflann m_pointCloud;
        std::unique_ptr<NanoKDTree> m_targetAccelerationStructure;

		unsigned int m_nNodes; //number of grid points
        unsigned int m_M; //number of mesh points
		TSDFVolume* m_volume;
		std::vector<float3>* m_vertices;
		std::vector<float3> m_vertices_2;
		std::vector<float3>* m_normals;
		std::vector<int3>* m_triangles;
		std::vector<int3>* m_vol_idx;
		std::vector<float3>* m_rel_coors;
		std::vector<unsigned int> grid_state; // only used in CPU version
		int im_h;
		int im_w;
		bool m_is_perspective;

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

        float m_fitSqrt;
        float m_regSqrt;

		std::shared_ptr<OptImage> m_gridPosFloat3; 
		std::shared_ptr<OptImage> m_anglesFloat3;
		std::shared_ptr<OptImage> m_robustWeights;
		std::shared_ptr<OptImage> m_gridPosFloat3Urshape;
		std::shared_ptr<OptImage> m_vertexPosTargetFloat3;
		std::shared_ptr<OptImage> m_vertexNormalTargetFloat3;		        
		std::shared_ptr<OptImage> m_triWeights;
		std::shared_ptr<OptGraph> m_graph;
		std::shared_ptr<OptGraph> m_regGrid;	
		
        float m_weightFit;
        float m_weightRegMax;
        float m_weightRegMin;
        float m_weightRegFactor;
        float m_weightReg;
        float m_functionTolerance;


		int max_threads = 512;

		float * m_depth_mat;

		float K_mat[9];  //[3 * 3] = {570.342, 0, 320,  0, 570.342, 240,  0, 0, 1};
		float K_inv_mat[9];  //[3 * 3] = {0.00175333, 0, -0.561066,  0, 0.00175333, -0.4208001,  0, 0, 1};
		std::vector<float3> h_gridVertexPosFloat3;
};
#endif
