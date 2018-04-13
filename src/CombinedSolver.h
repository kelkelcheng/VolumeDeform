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
// just for testing purpose
//extern "C" void initAngle(void* ptr_angles, unsigned int m_nNodes);


static bool operator==(const float3& v0, const float3& v1) {
    return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
static bool operator!=(const float3& v0, const float3& v1) {
    return !(v0 == v1);
}
#define MAX_K 20

static float clamp(float v, float mn, float mx) {
    return std::max(mn,std::min(v, mx));
}

class CombinedSolver : public CombinedSolverBase
{

	public:
        CombinedSolver(const SimpleMesh* sourceMesh, const std::vector<std::string> targetFiles, CombinedSolverParameters params, TSDFVolume* volume)
		{
            m_combinedSolverParameters = params;
            m_result = *sourceMesh;
			m_initial = m_result;
			m_targets = targetFiles;
            // !!! New
			m_volume = volume;
            m_gridDims = make_int3(25, 26, 4);
			//m_gridDims = make_int3(25, 26, 8);
			//m_gridDims = make_int3(5, 5, 3);

            m_dims      = m_gridDims;
			m_dims.x--; m_dims.y--; m_dims.z--;
			m_nNodes    = (m_dims.x + 1)*(m_dims.y + 1)*(m_dims.z + 1);

            uint M = (uint)sourceMesh->n_vertices();
            uint E = (uint)sourceMesh->n_edges();

            // !!! New
            m_M = M;
			m_vertexToVoxels.resize(M);
            m_relativeCoords.resize(M);
            
            double sumEdgeLength = 0.0f;
            for (auto edgeHandle : m_initial.edges()) {
                auto edge = m_initial.edge(edgeHandle);
                sumEdgeLength += m_initial.calc_edge_length(edgeHandle);
            }
            m_averageEdgeLength = sumEdgeLength / E;
            std::cout << "Average Edge Length: " << m_averageEdgeLength << std::endl;
			std::cout << "dim.x: " << m_dims.x <<" dim.y: "<<m_dims.y<<" dim.z: " <<m_dims.z << std::endl;

            std::vector<unsigned int> dims = { M };
			std::vector<unsigned int> dims_grid = { (uint)m_nNodes };

            m_vertexPosTargetFloat3     = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
			m_vertexNormalTargetFloat3  = createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
            m_robustWeights             = createEmptyOptImage(dims, OptImage::Type::FLOAT, 1, OptImage::Location::GPU, true);
            m_gridPosFloat3             = createEmptyOptImage(dims_grid, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
            m_gridPosFloat3Urshape      = createEmptyOptImage(dims_grid, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
            m_anglesFloat3              = createEmptyOptImage(dims_grid, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);
			m_triWeights              	= createEmptyOptImage(dims, OptImage::Type::FLOAT, 3, OptImage::Location::GPU, true);

			resetGPUMemory();   

            if (!m_result.has_vertex_colors()) {
                m_result.request_vertex_colors();
            }

            for (unsigned int i = 0; i < M; i++)
            {
                uchar w = 255;
                m_result.set_color(VertexHandle(i), Vec3uc(w, w, w));
            }

			addOptSolvers(dims, "opt_rotation.t", m_combinedSolverParameters.optDoublePrecision);
            addOptSolvers(dims, "opt_position.t", m_combinedSolverParameters.optDoublePrecision);

			//addOptSolvers(dims, "opt_both.t", m_combinedSolverParameters.optDoublePrecision);
            //addOptSolvers(dims, "opt_both.t", m_combinedSolverParameters.optDoublePrecision);		
		} 

        virtual void solveAll() override {
            combinedSolveInit();
            //for (auto s : m_solverInfo) {
                //if (s.enabled) {
                    m_result = m_initial; //? 
                    //resetGPUMemory();
					m_problemParams.set("G", m_graph);
					m_problemParams.set("RegGrid", m_regGrid);
                    for (m_targetIndex = 0; m_targetIndex < m_targets.size(); ++m_targetIndex) {
                        //singleSolve(s); //Edit!!
						singleSolve(m_solverInfo[0], m_solverInfo[1]); // New!!!
                    }
                //}
            //}
            combinedSolveFinalize();
        }

        virtual void combinedSolveInit() override {
            m_weightFit = 10.0f; 
            m_weightRegMax = 64.0f;
            
            m_weightRegMin = 32.0f;
            m_weightRegFactor = 0.9f;

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
			
		// New!!!
		virtual void singleSolve(SolverInfo s0, SolverInfo s1) {
		    preSingleSolve();
		    if (m_combinedSolverParameters.numIter == 1) {
		        preNonlinearSolve(0);
		        std::cout << "//////////// Solver 0 (" << s0.name << ") ///////////////" << std::endl;
		        s0.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve, s0.iterationInfo);
				postNonlinearSolve(0);

		        preNonlinearSolve(0);
				std::cout << "//////////// Solver 1 (" << s1.name << ") ///////////////" << std::endl;
				s1.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve, s1.iterationInfo);
		        postNonlinearSolve(0);
		    }

		    else {
				std::cout << "//////////// Solver 1 (" << s1.name << ") ///////////////" << std::endl;
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
            unsigned int M = (unsigned int)m_initial.n_vertices();
            m_previousConstraints.resize(M);
            for (int i = 0; i < (int)M; ++i) {
                m_previousConstraints[i] = make_float3(0, 0, -90901283092183);
            }
            m_weightReg = m_weightRegMax;
        }
        virtual void postSingleSolve() override { 
            char buff[100];
            sprintf(buff, "../output_mesh/out_%04d.ply", m_targetIndex);
            saveCurrentMesh(buff);
        }

        virtual void preNonlinearSolve(int) override {
            m_timer.start();
            int newConstraintCount = setConstraints((float)m_averageEdgeLength*10.0f); //5.0f (float)m_averageEdgeLength*10.0f
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
            reportFinalCosts("Robust Mesh Deformation", m_combinedSolverParameters, getCost("Opt(GN)"), getCost("Opt(LM)"), nan(""));
        }


        inline vec3f convertDepthToRGB(float depth, float depthMin = 0.0f, float depthMax = 1.0f) {
            float depthZeroOne = (depth - depthMin) / (depthMax - depthMin);
            float x = 1.0f - depthZeroOne;
            if (x < 0.0f) x = 0.0f;
            if (x > 1.0f) x = 1.0f;
            return BaseImageHelper::convertHSVtoRGB(vec3f(240.0f*x, 1.0f, 0.5f));
        }

        void saveCurrentMesh(std::string filename) {
            { // Save intermediate mesh
                unsigned int M = (unsigned int)m_result.n_vertices();
                std::vector<float> h_vertexWeightFloat(M);
                m_robustWeights->copyTo(h_vertexWeightFloat);
                for (unsigned int i = 0; i < M; i++)
                {
                    vec3f color = convertDepthToRGB(1.0f-clamp(h_vertexWeightFloat[i], 0.0f, 1.0f));

                    m_result.set_color(VertexHandle(i), Vec3uc((uchar)(color.r * 255), (uchar)(color.g * 255), (uchar)(color.b * 255)));

                    /*if (h_vertexWeightFloat[i] < 0.9f || h_vertexWeightFloat[i] > 1.0f) {
                        printf("Interesting robustWeight[%d]: %f\n", i, h_vertexWeightFloat[i]);
                    }*/
                }
                
                OpenMesh::IO::Options options = OpenMesh::IO::Options::VertexColor;
                int failure = OpenMesh::IO::write_mesh(m_result, filename, options);
                assert(failure);
            }
        }

        // !!!!!!!!!!!! new !!!!!!!!!!!!
		/*int getIndex1D(int3 idx)
        {
			return idx.x*((m_dims.y + 1)*(m_dims.z + 1)) + idx.y*(m_dims.z + 1) + idx.z;
		}*/
		int getIndex1D(int3 idx)
        {
			return idx.z * (m_gridDims.y * m_gridDims.x) + idx.y * m_gridDims.x + idx.x;
		}

		void vecTrans(float3 &v, float K[4][4])
		{
			float x = K[0][0] * v.x + K[0][1] * v.y + K[0][2] * v.z + K[0][3];
			float y = K[1][0] * v.x + K[1][1] * v.y + K[1][2] * v.z + K[1][3];
			float z = K[2][0] * v.x + K[2][1] * v.y + K[2][2] * v.z + K[2][3];
			v.x = x; v.y = y; v.z = z;
		}
        // !!!!!!!!!!!! new !!!!!!!!!!!!

		
        int setConstraints(float positionThreshold = 0.03f, float cosNormalThreshold = 0.2f, float viewThreshold = 0.8f) //std::numeric_limits<float>::infinity() 0.03 0.2
		{
			unsigned int M = (unsigned int)m_result.n_vertices();
			std::vector<float3> h_vertexPosTargetFloat3(M);
            std::vector<float3> h_vertexNormalTargetFloat3(M);

            if (!(m_result.has_face_normals() && m_result.has_vertex_normals())) {
                m_result.request_face_normals();
                m_result.request_vertex_normals();
            }
            m_result.update_normals();

            uint thrownOutCorrespondenceCount = 0;
            float3 invalidPt = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());

			//std::vector<float>  h_robustWeights(M);
            //m_robustWeights->copyTo(h_robustWeights);
            //int constraintsUpdated = 0;

			cv::Mat depth_mat = cv::imread(m_targets[m_targetIndex], CV_LOAD_IMAGE_UNCHANGED); 

			int testCount = 0;

            for (int i = 0; i < (int)M; i++) {
                std::vector<size_t> neighbors(MAX_K);
                std::vector<float> out_dist_sqr(MAX_K);
                auto currentPt = m_result.point(VertexHandle(i));
                auto sNormal = m_result.normal(VertexHandle(i));
                auto sourceNormal = make_float3(sNormal[0], sNormal[1], sNormal[2]);
                
				auto sourcePt = make_float3(currentPt[0], currentPt[1], currentPt[2]);
				vecTrans(sourcePt, mat_K);

				bool validTargetFound = false;

				float p_x = sourcePt.x / sourcePt.z; // edit -- sourcePt.x
				float p_y = sourcePt.y / sourcePt.z; //
				int p_x1 = floor(p_x); //floor(sourcePt.x / sourcePt.z); 
				int p_x2 = p_x1 + 1;//ceil(sourcePt.x / sourcePt.z); 
				int p_y1 = floor(p_y);//floor(sourcePt.y / sourcePt.z);
				int p_y2 = p_y1 + 1;//ceil(sourcePt.y / sourcePt.z);

				if (p_x1>=0 && p_x2<640-1 && p_y1>=0 && p_y2<480-1)
				{
					// find the depth of the 4 surrounding neighbours
					float p_z11 = (float)(depth_mat.at<unsigned short>(p_y1, p_x1)) / 1000.0f;
					float p_z12 = (float)(depth_mat.at<unsigned short>(p_y2, p_x1)) / 1000.0f;
					float p_z21 = (float)(depth_mat.at<unsigned short>(p_y1, p_x2)) / 1000.0f;
					float p_z22 = (float)(depth_mat.at<unsigned short>(p_y2, p_x2)) / 1000.0f;

					if (p_z11>0.0f && p_z12 >0.0f && p_z21 >0.0f && p_z22 >0.0f) //p_z11 <=1.0f && p_z12 <=1.0f && p_z21 <=1.0f && p_z22 <=1.0f
					{
						// use bilinear interpolation to get the depth
						/*float p_x11 = p_x1 * p_z11; float p_y11 = p_y1 * p_z11; 
						float p_x12 = p_x1 * p_z12; float p_y12 = p_y2 * p_z12; 
						float p_x21 = p_x2 * p_z21; float p_y21 = p_y1 * p_z21; 
						float p_x22 = p_x2 * p_z22; float p_y22 = p_y2 * p_z22;				 

						float p_z1 = (p_x2-p_x)/(p_x2-p_x1)*p_z11 + (p_x-p_x1)/(p_x2-p_x1)*p_z21;
						float p_z2 = (p_x2-p_x)/(p_x2-p_x1)*p_z12 + (p_x-p_x1)/(p_x2-p_x1)*p_z22;
						float p_z = (p_y2-p_y)/(p_y2-p_y1)*p_z1 + (p_y-p_y1)/(p_y2-p_y1)*p_z2;*/
					
						// project back to world coordinate
						//float3 target = make_float3(p_x, p_y, p_z); 

						//p_x = (int)round(p_x); p_y = (int)round(p_y);
						//float p_z = (float)(depth_mat.at<unsigned short>(p_y, p_x));
						//float3 target = make_float3(p_x * p_z, p_y * p_z, p_z);
						float3 target = make_float3(p_x1*p_z11, p_y1*p_z11, p_z11);
						vecTrans(target, mat_K_inv);

						// construct normal
						const Vec3f targetPt = Vec3f(target.x, target.y, target.z);

						// update depth value if the point falls on axis
						/*if (p_x1 == p_x2) {p_x2 = p_x1 + 1;}
						if (p_y1 == p_y2) {p_y2 = p_y1 + 1;}
						p_z12 = (float)(depth_mat.at<unsigned short>(p_y2, p_x1)) / 1000.0f;
						p_z21 = (float)(depth_mat.at<unsigned short>(p_y1, p_x2)) / 1000.0f;
						p_z22 = (float)(depth_mat.at<unsigned short>(p_y2, p_x2)) / 1000.0f;		
						// determine which triangle the point belongs to and get the u,v vectors for calculating normal
						float3 uVec; float3 vVec;
						if (p_y <= (-1*p_x+1)) {
							uVec.x = p_x2*p_z21-p_x1*p_z11; uVec.y = p_y1*p_z21-p_y1*p_z11; uVec.z = p_z21-p_z11;
							vVec.x = p_x1*p_z12-p_x1*p_z11; vVec.y = p_y2*p_z12-p_y1*p_z11; vVec.z = p_z12-p_z11;							
						}
						else {
							uVec.x = p_x2*p_z21-p_x2*p_z22; uVec.y = p_y1*p_z21-p_y2*p_z21; uVec.z = p_z21-p_z22;
							vVec.x = p_x1*p_z12-p_x2*p_z22; vVec.y = p_y2*p_z12-p_y2*p_z22; vVec.z = p_z12-p_z12;							
						}*/

						/*p_x1 = (int)floor(p_x1);  p_x2 = p_x1 + 1;
						p_y1 = (int)floor(p_y1);  p_y2 = p_y1 + 1;
						p_z11 = (float)(depth_mat.at<unsigned short>(p_y1, p_x1)) / 1000.0f;
						p_z12 = (float)(depth_mat.at<unsigned short>(p_y2, p_x1)) / 1000.0f;
						p_z21 = (float)(depth_mat.at<unsigned short>(p_y1, p_x2)) / 1000.0f;*/
						float3 uVec; float3 vVec;	
						uVec.x = p_x2*p_z21-p_x1*p_z11; uVec.y = p_y1*p_z21-p_y1*p_z11; uVec.z = p_z21-p_z11;
						vVec.x = p_x1*p_z12-p_x1*p_z11; vVec.y = p_y2*p_z12-p_y1*p_z11; vVec.z = p_z12-p_z11;
				
						vecTrans(uVec, mat_K_inv); vecTrans(vVec, mat_K_inv);
						// calculate unit normal
						//float3 normVec = make_float3(uVec[1]*vVec[2]-uVec[2]*vVec[1], uVec[2]*vVec[0]-uVec[0]*vVec[2], uVec[0]*vVec[1]-uVec[1]*vVec[0]);
						float3 normVec = make_float3(uVec.y*vVec.z-uVec.z*vVec.y, uVec.z*vVec.x-uVec.x*vVec.z, uVec.x*vVec.y-uVec.y*vVec.x);
						//vecTrans(normVec, mat_K_inv);
						float mag = sqrt(normVec.x*normVec.x + normVec.y*normVec.y + normVec.z*normVec.z);
						normVec.x /= mag; normVec.y /= mag; normVec.z /= mag;

						float dist = (targetPt - currentPt).length();

						// New
						if (dot(normVec, sourceNormal) < 0) {
							normVec.x = -normVec.x; normVec.y = -normVec.y; normVec.z = -normVec.z; 
						}

		                if (dist < positionThreshold) {
							/*if (testCount < 10) {
								std::cout << "i: " << i << std::endl;
								std::cout << "dist: "<<dist<<" p_x1: "<<p_x1<<" p_y1: "<<p_y1<<" p_z11: "<<p_z11<<std::endl;
								std::cout << "currentPt=>x: "<<currentPt[0]<<" y: "<<currentPt[1]<<" z: "<<currentPt[2]<<std::endl;
								std::cout << "targetPt=>x: "<<targetPt[0]<<" y: "<<targetPt[1]<<" z: "<<targetPt[2]<<std::endl;
								std::cout << "normVec=>x: "<<normVec.x<<" y: "<<normVec.y<<" z: "<<normVec.z<<std::endl;
								std::cout << "sourceNormal=>x: "<<sourceNormal.x<<" y: "<<sourceNormal.y<<" z: "<<sourceNormal.z<<std::endl;
								std::cout << "dot(normVec, sourceNormal): "<<dot(normVec, sourceNormal)<<std::endl;
								testCount++;
							}*/
				            if (dot(normVec, sourceNormal) > cosNormalThreshold) {
								h_vertexPosTargetFloat3[i] = target;
								h_vertexNormalTargetFloat3[i] = normVec;
				                validTargetFound = true;
				            }

						}
					}
				}

                if (!validTargetFound) {
                    ++thrownOutCorrespondenceCount;
                    h_vertexPosTargetFloat3[i] = invalidPt;
					//h_vertexNormalTargetFloat3[i] = make_float3(0.0f, 0.0f, 0.0f);
					//h_robustWeights[i] = 0.0f;
                }

                /*if (m_previousConstraints[i] != h_vertexPosTargetFloat3[i]) {
                    m_previousConstraints[i] = h_vertexPosTargetFloat3[i];
                    ++constraintsUpdated;
                    
                    auto currentPt = m_result.point(VertexHandle(i));
                    auto v = h_vertexPosTargetFloat3[i];
                    const Vec3f target = Vec3f(v.x, v.y, v.z);
                    float dist = (target - currentPt).length();

					float weight_d = 1.0f - dist / positionThreshold;
					float weight_n = 1.0f - (1.0f - dot(h_vertexNormalTargetFloat3[i], sourceNormal)) / cosNormalThreshold;

					float3 camera_view = make_float3(-0.4594f, -0.3445f, 0.8187f);
					float weight_v = 1.0f - (1.0f - dot(h_vertexNormalTargetFloat3[i], camera_view)) / viewThreshold;

					if (weight_d >= 0) { //&& weight_n >= 0 && weight_v >= 0
						//float weight = (weight_d + weight_n + weight_v) / 3;
						float weight = weight_d;
						h_robustWeights[i] = fmaxf(0.1f, weight*0.9f+0.05f); //weight*0.9f+0.05f
					} else {
						h_robustWeights[i] = 0.1f;
					}

                    float weight = (positionThreshold - dist) / positionThreshold;
                    h_robustWeights[i] = fmaxf(0.1f, weight*0.9f+0.05f); //weight*weight
                }*/
			}


            m_vertexPosTargetFloat3->update(h_vertexPosTargetFloat3);
            m_vertexNormalTargetFloat3->update(h_vertexNormalTargetFloat3);

            std::vector<float>  h_robustWeights(M);
            m_robustWeights->copyTo(h_robustWeights);
            int constraintsUpdated = 0;
            for (int i = 0; i < (int)M; ++i) {
                if (m_previousConstraints[i] != h_vertexPosTargetFloat3[i]) {
                    m_previousConstraints[i] = h_vertexPosTargetFloat3[i];
                    ++constraintsUpdated;
                    {
                        auto currentPt = m_result.point(VertexHandle(i));
                        auto v = h_vertexPosTargetFloat3[i];
                        const Vec3f target = Vec3f(v.x, v.y, v.z);
                        float dist = (target - currentPt).length();
                        float weight = (positionThreshold - dist) / positionThreshold;
						
                        //h_robustWeights[i] = fmaxf(0.1f, weight*0.9f+0.05f); //weight*weight
						h_robustWeights[i] = fmaxf(0.01f, weight*0.99f+0.005f);
                        //h_robustWeights[i] = 1.0f;
                    }
                }
            }


            m_robustWeights->update(h_robustWeights);

            std::cout << "*******Thrown out correspondence count: " << thrownOutCorrespondenceCount << std::endl;

            return constraintsUpdated;
		}

		/*void computeBoundingBox()
		{
			m_min = make_float3(+std::numeric_limits<float>::max(), +std::numeric_limits<float>::max(), +std::numeric_limits<float>::max());
			m_max = make_float3(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
			for (SimpleMesh::VertexIter v_it = m_initial.vertices_begin(); v_it != m_initial.vertices_end(); ++v_it)
			{
				SimpleMesh::Point p = m_initial.point(VertexHandle(*v_it));
				m_min.x = fmin(m_min.x, p[0]); m_min.y = fmin(m_min.y, p[1]); m_min.z = fmin(m_min.z, p[2]);
				m_max.x = fmax(m_max.x, p[0]); m_max.y = fmax(m_max.y, p[1]); m_max.z = fmax(m_max.z, p[2]);
			}
		}*/

		void computeBoundingBox()
		{
			m_min = make_float3(+std::numeric_limits<float>::max(), +std::numeric_limits<float>::max(), +std::numeric_limits<float>::max());
			m_max = make_float3(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
			for (int i=0; i<m_nNodes; i++){
				float3 pt = h_gridVertexPosFloat3[i];
				m_min.x = fmin(m_min.x, pt.x); m_min.y = fmin(m_min.y, pt.y); m_min.z = fmin(m_min.z, pt.z);
				m_max.x = fmax(m_max.x, pt.x); m_max.y = fmax(m_max.y, pt.y); m_max.z = fmax(m_max.z, pt.z);
			}
		}

		void initializeWarpGrid()
		{
            std::vector<int> regGrid_v0;
            std::vector<int> regGrid_v1;

			//std::vector<float3> h_gridVertexPosFloat3(m_nNodes);//
			h_gridVertexPosFloat3.resize(m_nNodes);
			m_volume->InitSubGrid(h_gridVertexPosFloat3, m_gridDims);
			for (int i = 0; i <= m_dims.x; i++)
			{
				for (int j = 0; j <= m_dims.y; j++)
				{
					for (int k = 0; k <= m_dims.z; k++)
					{
						//float3 fac = make_float3((float)i, (float)j, (float)k);//
						//float3 v = m_min + fac*m_delta;//
						int3 gridIdx = make_int3(i, j, k);
						//h_gridVertexPosFloat3[getIndex1D(gridIdx)] = v;//
						

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

		void resetGPUMemory()
		{
			//computeBoundingBox();
			//m_delta = (m_max - m_min); m_delta.x /= (m_dims.x); m_delta.y /= (m_dims.y); m_delta.z /= (m_dims.z);
			initializeWarpGrid();

			/*m_min = make_float3(+std::numeric_limits<float>::max(), +std::numeric_limits<float>::max(), +std::numeric_limits<float>::max());
			m_max = make_float3(-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max());
			for (SimpleMesh::VertexIter v_it = m_initial.vertices_begin(); v_it != m_initial.vertices_end(); ++v_it)
			{
				SimpleMesh::Point p = m_initial.point(VertexHandle(*v_it));
				m_min.x = fmin(m_min.x, p[0]); m_min.y = fmin(m_min.y, p[1]); m_min.z = fmin(m_min.z, p[2]);
				m_max.x = fmax(m_max.x, p[0]); m_max.y = fmax(m_max.y, p[1]); m_max.z = fmax(m_max.z, p[2]);
			}

			m_delta = (m_max - m_min); m_delta.x /= (m_dims.x); m_delta.y /= (m_dims.y); m_delta.z /= (m_dims.z);
			std::cout<<"m_min.x: "<<m_min.x<<" m_min.y: "<<m_min.y<<" m_min.z: "<<m_min.z <<std::endl;
			std::cout<<"m_max.x: "<<m_max.x<<" m_max.y: "<<m_max.y<<" m_max.z: "<<m_max.z <<std::endl;
			std::cout<<"m_delta.x: "<<m_delta.x<<" m_delta.y: "<<m_delta.y<<" m_delta.z: "<<m_delta.z <<std::endl;*/
            
            computeBoundingBox();

			//saveGraphResults();
			//float EPS = 0.000001f;
			//m_min -= make_float3(EPS, EPS, EPS);
			//m_max += make_float3(EPS, EPS, EPS);
			//m_min = make_float3(-1.5f,-1.5f,0.5f);
			m_delta = (m_max - m_min); m_delta.x /= (m_dims.x); m_delta.y /= (m_dims.y); m_delta.z /= (m_dims.z);
			std::cout<<"m_min.x: "<<m_min.x<<" m_min.y: "<<m_min.y<<" m_min.z: "<<m_min.z <<std::endl;
			std::cout<<"m_max.x: "<<m_max.x<<" m_max.y: "<<m_max.y<<" m_max.z: "<<m_max.z <<std::endl;
			std::cout<<"m_delta.x: "<<m_delta.x<<" m_delta.y: "<<m_delta.y<<" m_delta.z: "<<m_delta.z <<std::endl;
            

	        std::vector<int> w;
			std::vector<int> v1; std::vector<int> v2; std::vector<int> v3; std::vector<int> v4;
			std::vector<int> v5; std::vector<int> v6; std::vector<int> v7; std::vector<int> v8; 

			for (SimpleMesh::VertexIter v_it = m_initial.vertices_begin(); v_it != m_initial.vertices_end(); ++v_it)
			{
			    VertexHandle c_vh(*v_it);
				SimpleMesh::Point p = m_initial.point(c_vh);
				float3 pp = make_float3(p[0], p[1], p[2]);

				pp = (pp - m_min);
				pp.x /= m_delta.x;
				pp.y /= m_delta.y;
				pp.z /= m_delta.z;

				int3 pInt = make_int3((int)pp.x, (int)pp.y, (int)pp.z);
				m_vertexToVoxels[c_vh.idx()] = pInt;
				m_relativeCoords[c_vh.idx()] = pp - make_float3((float)pInt.x, (float)pInt.y, (float)pInt.z);

				w.push_back(c_vh.idx()); 

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

			//initAngle(m_anglesFloat3->data(), m_nNodes);
			//copyResultToCPUFromFloat3();
			//OpenMesh::IO::write_mesh(m_result, "../output_mesh/test_subgrid.ply");
		}

        SimpleMesh* result()
        {
            return &m_result;
        }

		// New!
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

		void set_mesh(SimpleMesh* targetMesh)
		{
			m_result = *targetMesh;
		}

		void copyResultToCPUFromFloat3()
		{
            std::vector<float3> h_gridPosFloat3(m_nNodes);
            m_gridPosFloat3->copyTo(h_gridPosFloat3);

			for (SimpleMesh::VertexIter v_it = m_result.vertices_begin(); v_it != m_result.vertices_end(); ++v_it)
			{
				VertexHandle vh(*v_it);

				int3   voxelId = m_vertexToVoxels[vh.idx()];
				float3 relativeCoords = m_relativeCoords[vh.idx()];

				float3 p000 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(0, 0, 0))];
				float3 p001 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(0, 0, 1))];
				float3 p010 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(0, 1, 0))];
				float3 p011 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(0, 1, 1))];
				float3 p100 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(1, 0, 0))];
				float3 p101 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(1, 0, 1))];
				float3 p110 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(1, 1, 0))];
				float3 p111 = h_gridPosFloat3[getIndex1D(voxelId + make_int3(1, 1, 1))];

				float3 px00 = (1.0f - relativeCoords.x)*p000 + relativeCoords.x*p100;
				float3 px01 = (1.0f - relativeCoords.x)*p001 + relativeCoords.x*p101;
				float3 px10 = (1.0f - relativeCoords.x)*p010 + relativeCoords.x*p110;
				float3 px11 = (1.0f - relativeCoords.x)*p011 + relativeCoords.x*p111;

				float3 pxx0 = (1.0f - relativeCoords.y)*px00 + relativeCoords.y*px10;
				float3 pxx1 = (1.0f - relativeCoords.y)*px01 + relativeCoords.y*px11;

				float3 p = (1.0f - relativeCoords.z)*pxx0 + relativeCoords.z*pxx1;

				m_result.set_point(vh, SimpleMesh::Point(p.x, p.y, p.z));
			}
		}

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

		float3 m_min;
		float3 m_max;

		int3   m_dims;
		int3   m_gridDims;
		float3 m_delta;

		std::vector<int3>   m_vertexToVoxels;
        std::vector<float3> m_relativeCoords;

        ml::Timer m_timer;

		SimpleMesh m_result;
		SimpleMesh m_initial;

		std::vector<std::string> m_targets;
        std::vector<float3> m_previousConstraints;
        std::vector<int4> m_sourceTetIndices;

        double m_averageEdgeLength;

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

		float mat_K[4][4] =
		{
			{570.342, 0, 320, 0},
			{0, 570.342, 240, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1}
		};

		/*float mat_K_inv[4][4] =
		{
			{0.0018, 0, -0.5611, 0}, //0.00175333
			{0, 0.0018, -0.4208, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1}
		};*/

		float mat_K_inv[4][4] =
		{
			{0.00175333, 0, -0.561066, 0}, //
			{0, 0.00175333, -0.4208001, 0},
			{0, 0, 1, 0},
			{0, 0, 0, 1}
		};
		std::vector<float3> h_gridVertexPosFloat3;
};
