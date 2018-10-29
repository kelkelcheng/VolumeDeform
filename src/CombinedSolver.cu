#include "CombinedSolver.h"
#include "cudaUtil.h"
#include <math.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>

static bool __host__ __device__ operator==(const float3& v0, const float3& v1) {
    return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
static bool __host__ __device__ operator!=(const float3& v0, const float3& v1) {
    return !(v0 == v1);
}

// from 3d index to 1d index
__device__
int index_3to1(int3 idx, int3 grid_size)
{
	return idx.z * (grid_size.y * grid_size.x) + idx.y * grid_size.x + idx.x;
}

// trilinear intepolation
__device__
void update_pos(int i, float3 * vertices, float3 * gridPosFloat3, int3 * voxel_idx, float3 * rel_coors, int3 grid_size) {
	int3   voxelId = voxel_idx[i]; 
	float3 relativeCoords = rel_coors[i]; 

	float3 p000 = gridPosFloat3[index_3to1(voxelId + make_int3(0, 0, 0), grid_size)];
	float3 p001 = gridPosFloat3[index_3to1(voxelId + make_int3(0, 0, 1), grid_size)];
	float3 p010 = gridPosFloat3[index_3to1(voxelId + make_int3(0, 1, 0), grid_size)];
	float3 p011 = gridPosFloat3[index_3to1(voxelId + make_int3(0, 1, 1), grid_size)];
	float3 p100 = gridPosFloat3[index_3to1(voxelId + make_int3(1, 0, 0), grid_size)];
	float3 p101 = gridPosFloat3[index_3to1(voxelId + make_int3(1, 0, 1), grid_size)];
	float3 p110 = gridPosFloat3[index_3to1(voxelId + make_int3(1, 1, 0), grid_size)];
	float3 p111 = gridPosFloat3[index_3to1(voxelId + make_int3(1, 1, 1), grid_size)];

	float3 px00 = (1.0f - relativeCoords.x)*p000 + relativeCoords.x*p100;
	float3 px01 = (1.0f - relativeCoords.x)*p001 + relativeCoords.x*p101;
	float3 px10 = (1.0f - relativeCoords.x)*p010 + relativeCoords.x*p110;
	float3 px11 = (1.0f - relativeCoords.x)*p011 + relativeCoords.x*p111;

	float3 pxx0 = (1.0f - relativeCoords.y)*px00 + relativeCoords.y*px10;
	float3 pxx1 = (1.0f - relativeCoords.y)*px01 + relativeCoords.y*px11;

	float3 p = (1.0f - relativeCoords.z)*pxx0 + relativeCoords.z*pxx1;

	vertices[i] = p;
}

// update vertices positions and normals
// notice that each index in triangle is unique... maybe we can save up some space but it is easier this way
__global__
void update_pos_normal(float3 * vertices, float3 * normals, int3 * triangles, float3 * gridPosFloat3, int3 * voxel_idx, float3 * rel_coors, int size, int3 grid_size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		update_pos(triangles[i].x, vertices, gridPosFloat3, voxel_idx, rel_coors, grid_size);
		update_pos(triangles[i].y, vertices, gridPosFloat3, voxel_idx, rel_coors, grid_size);
		update_pos(triangles[i].z, vertices, gridPosFloat3, voxel_idx, rel_coors, grid_size);
		
		float3 uVec = vertices[triangles[i].y] - vertices[triangles[i].x];
		float3 vVec = vertices[triangles[i].z] - vertices[triangles[i].x];
		float3 v_norm = make_float3(uVec.y*vVec.z-uVec.z*vVec.y, uVec.z*vVec.x-uVec.x*vVec.z, uVec.x*vVec.y-uVec.y*vVec.x);
		if (v_norm.z > 0) {
			v_norm.x = -v_norm.x; v_norm.y = -v_norm.y; v_norm.z = -v_norm.z;
		}
		v_norm = normalize(v_norm);
		normals[triangles[i].x] = v_norm;
		normals[triangles[i].y] = v_norm;
		normals[triangles[i].z] = v_norm;			
	}
}

// update vertices positions and normals after each optimization
__host__
void CombinedSolver::copyResultToCPUFromFloat3() {
	std::vector<float3>& vertices = *m_vertices;
	std::vector<float3>& normals = *m_normals;
	std::vector<int3>& triangles = *m_triangles;

	// get the number of vertices and the number of blocks for GPU
	int M = vertices.size();
	int blocknum = ceil(M/(float)max_threads);
	
	// intialize and copy memory to GPU	
	float3 * d_vertices;
	float3 * d_rel_coors; // trilinear weights in the sparse subgrid
	int3 * d_voxel_idx; // cube index in the full grid
	cudaSafeCall( cudaMalloc(&d_vertices, M * sizeof(float3)) );
	cudaSafeCall( cudaMalloc(&d_voxel_idx, M * sizeof(int3)) );
	cudaSafeCall( cudaMalloc(&d_rel_coors, M * sizeof(float3)) );
	
	cudaSafeCall( cudaMemcpy(d_voxel_idx, m_vertexToVoxels.data(), M * sizeof(int3), cudaMemcpyHostToDevice) );
	cudaSafeCall( cudaMemcpy(d_rel_coors, m_relativeCoords.data(), M * sizeof(float3), cudaMemcpyHostToDevice) );

	// normals	
	float3 * d_normals;
	int3 * d_triangles;
	cudaSafeCall( cudaMalloc(&d_normals, M * sizeof(float3)) );
	
	// since each index in triangles is unique (each vertice belong to only one triangle), we only need one thread per triangle
	M = triangles.size();
	blocknum = ceil(M/(float)max_threads);
	cudaSafeCall( cudaMalloc(&d_triangles, M * sizeof(int3)) );
	cudaSafeCall( cudaMemcpy(d_triangles, triangles.data(), M * sizeof(int3), cudaMemcpyHostToDevice) );
	
	update_pos_normal<<< blocknum, max_threads >>>(d_vertices, d_normals, d_triangles, (float3*)m_gridPosFloat3->data(), d_voxel_idx, d_rel_coors, M, m_gridDims);
	cudaSafeCall( cudaDeviceSynchronize() );
	
	cudaSafeCall( cudaMemcpy(vertices.data(), d_vertices, vertices.size() * sizeof(float3), cudaMemcpyDeviceToHost) );
	cudaSafeCall( cudaMemcpy(normals.data(), d_normals, vertices.size() * sizeof(float3), cudaMemcpyDeviceToHost) );
	
	cudaSafeCall( cudaFree(d_vertices) );
	cudaSafeCall( cudaFree(d_normals) );
	cudaSafeCall( cudaFree(d_triangles) );
	cudaSafeCall( cudaFree(d_voxel_idx) );
	cudaSafeCall( cudaFree(d_rel_coors) );
}

// K * v
__device__
void vecTrans(float3 &v, float K[9])
{
	float x = K[0] * v.x + K[1] * v.y + K[2] * v.z;
	float y = K[3] * v.x + K[4] * v.y + K[5] * v.z;
	float z = K[6] * v.x + K[7] * v.y + K[8] * v.z;
	v.x = x; v.y = y; v.z = z;
}

// kernel for updating target point positions, normals, and robust weights		
__global__
void update_constraint(	float3 * d_vertices, float3 * d_normals, int3 * d_vol_idx, dim3 vol_size,
						float * K_mat, float * K_inv_mat, int size, float * d_depth_mat, float3 * d_previousConstraints,
						float3 * d_vertexPosTargetFloat3, float3 * d_vertexNormalTargetFloat3, float * d_robustWeights, unsigned int * grid_state,
						float positionThreshold, float cosNormalThreshold, float3 invalidPt, int * num_update) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		// get the associated point and normal
		float3 currentPt = d_vertices[i];
		float3 sourcePt = d_vertices[i];
		float3 sourceNormal = d_normals[i];

		// K * p: to image coordinates
		vecTrans(sourcePt, K_mat);

		bool validTargetFound = 0;

		// get the surrounding x and y
		int p_x1 = floor(sourcePt.x / sourcePt.z); 
		int p_x2 = p_x1 + 1; 
		int p_y1 = floor(sourcePt.y / sourcePt.z);
		int p_y2 = p_y1 + 1;

		if (p_x1>=0 && p_x2<640-1 && p_y1>=0 && p_y2<480-1)
		{
			// find the depth of the 4 surrounding neighbours
			float p_z11 = d_depth_mat[p_y1 * 640 + p_x1];
			float p_z12 = d_depth_mat[p_y2 * 640 + p_x1];
			float p_z21 = d_depth_mat[p_y1 * 640 + p_x2];
			float p_z22 = d_depth_mat[p_y2 * 640 + p_x2];

			if (p_z11>0.0f && p_z12 >0.0f && p_z21 >0.0f && p_z22 >0.0f) // safe guard
			{
				// get target point position
				float3 target = make_float3(p_x1*p_z11, p_y1*p_z11, p_z11);
				vecTrans(target, K_inv_mat);

				// get tangent and bitanget as uVec and vVec
				float3 uVec; float3 vVec;	
				uVec.x = p_x2*p_z21-p_x1*p_z11; uVec.y = p_y1*p_z21-p_y1*p_z11; uVec.z = p_z21-p_z11;
				vVec.x = p_x1*p_z12-p_x1*p_z11; vVec.y = p_y2*p_z12-p_y1*p_z11; vVec.z = p_z12-p_z11;
		
				// transform tangent and bitanget
				vecTrans(uVec, K_inv_mat); vecTrans(vVec, K_inv_mat);
				
				// calculate unit normal
				float3 normVec = make_float3(uVec.y*vVec.z-uVec.z*vVec.y, uVec.z*vVec.x-uVec.x*vVec.z, uVec.x*vVec.y-uVec.y*vVec.x);
				normVec = normalize(normVec);

				// get the distance between the source point and the target point
				float3 distVec = target - currentPt;
				float dist = length(distVec);

				// get the correct normal direction
				if (normVec.z > 0) {
					normVec.x = -normVec.x; normVec.y = -normVec.y; normVec.z = -normVec.z; 
				}
				
				// update if it is under threshold
				// get corresponding volume index (inside TSDFVolume)        
                int3 idx3 = d_vol_idx[i];
                int idx1 = idx3.z * (vol_size.y * vol_size.x) + idx3.y * vol_size.x + idx3.x;
				bool check_updated = 0;
				
                if (dist < positionThreshold) {
		            if (dot(normVec, sourceNormal) > cosNormalThreshold) {
						d_vertexPosTargetFloat3[i] = target;
						d_vertexNormalTargetFloat3[i] = normVec;
		                validTargetFound = 1;	               
		            }
		            check_updated = 1;
		            grid_state[idx1] = 2; // grid_state controls the if neighbours can be integrated
				}
				// disable the neighbors of the grid if the cube is not well deformed
				if ((!check_updated) && (grid_state[idx1]==2)) {grid_state[idx1] = 0;}
			}
		}
		
		// set the target to negative infinity if unbounded
        if (!validTargetFound) {
            //++thrownOutCorrespondenceCount;
            d_vertexPosTargetFloat3[i] = invalidPt;
        }

		// update the weights
        if (d_previousConstraints[i] != d_vertexPosTargetFloat3[i]) { //d_previousConstraints[i] != d_vertexPosTargetFloat3[i]
            d_previousConstraints[i] = d_vertexPosTargetFloat3[i];
            //++constraintsUpdated;
            num_update[i] = 1;
            
            float3 currentPt = d_vertices[i];
            float3 v = d_vertexPosTargetFloat3[i];
            float3 distVec = currentPt - v;
            float dist = length(distVec);

			// weight_d, weight_n, weight_v correspond to position, normal, camera view deviation
			float weight_d = 1.0f - dist / positionThreshold;
			float weight_n = 1.0f - (1.0f - dot(d_vertexNormalTargetFloat3[i], sourceNormal)) / cosNormalThreshold;

			// use camera view direction will make the edges unstable
			//float3 camera_view = make_float3(-0.4594f, -0.3445f, 0.8187f);
			//float weight_v = 1.0f - (1.0f - dot(h_vertexNormalTargetFloat3[i], camera_view)) / viewThreshold;

			if (weight_d >= 0 && weight_n >= 0) { //&& weight_n >= 0 && weight_v >= 0
				//float weight = (weight_d + weight_n + weight_v) / 3;
				float weight = (weight_d + weight_n) / 2;
				//float weight = weight_d;
				d_robustWeights[i] = fmaxf(0.1f, weight*0.9f+0.05f); //weight*0.9f+0.05f
			} else {
				d_robustWeights[i] = 0.0f;
			}
        }
	}
}

// setting target point positions, normals, and robust weights	before optimization
// will be called for each ICP iteration
// many of the inputs here can be passed directly from MarchingCubes, make it a class later...
__host__
int CombinedSolver::setConstraints(float positionThreshold, float cosNormalThreshold, float viewThreshold) //0.03 0.2 0.8
{
	dim3 vol_size = m_volume->get_size(); // used to locate grid_state
	int M = (*m_vertices).size();

    int constraintsUpdated = 0; // this information might not be very useful... 

	float3 invalidPt = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());

	int blocknum = ceil(M/(float)max_threads);
	float3 * d_vertices; float3 * d_normals; float3 * d_previousConstraints;
	int3 * d_voxel_idx;
	float * d_K; float * d_K_inv; float * d_depth_mat;
	int * num_update; 
	cudaSafeCall( cudaMalloc(&d_vertices, M * sizeof(float3)) );
	cudaSafeCall( cudaMalloc(&d_normals, M * sizeof(float3)) );
	cudaSafeCall( cudaMalloc(&d_voxel_idx, M * sizeof(int3)) ); // cube index for each vertex
	
	cudaSafeCall( cudaMalloc(&d_K, 9 * sizeof(float)) ); // intrinsic matrix K
	cudaSafeCall( cudaMalloc(&d_K_inv, 9 * sizeof(float)) ); // inverse of K
	cudaSafeCall( cudaMalloc(&d_depth_mat, (640*480) * sizeof(float)) ); // depth image
	cudaSafeCall( cudaMalloc(&d_previousConstraints, M * sizeof(float3)) ); // previous target positions
	
	cudaSafeCall( cudaMalloc(&num_update, M * sizeof(int)) );	
	cudaSafeCall( cudaMemset(num_update, 0, M * sizeof(int)) );
	
	cudaSafeCall( cudaMemcpy(d_voxel_idx, m_vol_idx->data(), M * sizeof(int3), cudaMemcpyHostToDevice) ); //do not use m_vertexToVoxels.data()...  terrible mistake
	cudaSafeCall( cudaMemcpy(d_vertices, m_vertices->data(), M * sizeof(float3), cudaMemcpyHostToDevice) );	
	cudaSafeCall( cudaMemcpy(d_normals, m_normals->data(), M * sizeof(float3), cudaMemcpyHostToDevice) );	
	cudaSafeCall( cudaMemcpy(d_K, K_mat, 9 * sizeof(float), cudaMemcpyHostToDevice) );
	cudaSafeCall( cudaMemcpy(d_K_inv, K_inv_mat, 9 * sizeof(float), cudaMemcpyHostToDevice) );
	cudaSafeCall( cudaMemcpy(d_depth_mat, m_depth_mat, (640*480) * sizeof(float), cudaMemcpyHostToDevice) );
	cudaSafeCall( cudaMemcpy(d_previousConstraints, m_previousConstraints.data(), M * sizeof(float3), cudaMemcpyHostToDevice) );	
	
	update_constraint<<< blocknum, max_threads >>>(	d_vertices, d_normals, d_voxel_idx, vol_size,
						d_K, d_K_inv, M, d_depth_mat, d_previousConstraints,
						(float3*) m_vertexPosTargetFloat3->data(), (float3*) m_vertexNormalTargetFloat3->data(), 
						(float*) m_robustWeights->data(), m_volume->get_state(), positionThreshold, cosNormalThreshold, invalidPt, num_update); 
	cudaSafeCall( cudaDeviceSynchronize() );

	constraintsUpdated = thrust::count(thrust::device_ptr<int>(num_update), thrust::device_ptr<int>(num_update + M), 1);
	
	cudaSafeCall( cudaMemcpy(m_previousConstraints.data(), d_previousConstraints, M * sizeof(float3), cudaMemcpyDeviceToHost) );

	cudaSafeCall( cudaFree(d_vertices) );
	cudaSafeCall( cudaFree(d_normals) );
	cudaSafeCall( cudaFree(d_voxel_idx) );
	cudaSafeCall( cudaFree(d_K) );
	cudaSafeCall( cudaFree(d_K_inv) );
	cudaSafeCall( cudaFree(d_depth_mat) );
	cudaSafeCall( cudaFree(d_previousConstraints) );
	cudaSafeCall( cudaFree(num_update) );
	
    return constraintsUpdated;
}

// CPU version of setConstraints
/*__host__
int CombinedSolver::setConstraints(float positionThreshold = 0.03f, float cosNormalThreshold = 0.2f, float viewThreshold = 0.8f) //std::numeric_limits<float>::infinity() 0.03 0.2
{
	//test
	dim3 vol_size = m_volume->get_size();
	cudaMemcpy(grid_state.data(), m_volume->get_state(), vol_size.x * vol_size.y * vol_size.z * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	//unsigned int M = (unsigned int)m_result.n_vertices();
	unsigned int M = (unsigned int)(*m_vertices).size();
	std::vector<float3> h_vertexPosTargetFloat3(M);
    std::vector<float3> h_vertexNormalTargetFloat3(M);

    uint thrownOutCorrespondenceCount = 0;
    float3 invalidPt = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());

	std::vector<float>  h_robustWeights(M);
    m_robustWeights->copyTo(h_robustWeights);
    int constraintsUpdated = 0;

	//cv::Mat depth_mat = cv::imread(m_targets[m_targetIndex], CV_LOAD_IMAGE_UNCHANGED); 

	//int testCount = 0;

    for (int i = 0; i < (int)M; i++) {
		float3 currentPt = (*m_vertices)[i];
		float3 sourcePt = (*m_vertices)[i];
		float3 sourceNormal = (*m_normals)[i];

		vecTrans(sourcePt, mat_K);

		bool validTargetFound = false;

		float p_x = sourcePt.x / sourcePt.z; // edit -- sourcePt.x
		float p_y = sourcePt.y / sourcePt.z; //
		int p_x1 = floor(p_x); 
		int p_x2 = p_x1 + 1; 
		int p_y1 = floor(p_y);
		int p_y2 = p_y1 + 1;

		if (p_x1>=0 && p_x2<640-1 && p_y1>=0 && p_y2<480-1)
		{
			// find the depth of the 4 surrounding neighbours
			//float p_z11 = (float)(depth_mat.at<unsigned short>(p_y1, p_x1)) / 1000.0f;
			//float p_z12 = (float)(depth_mat.at<unsigned short>(p_y2, p_x1)) / 1000.0f;
			//float p_z21 = (float)(depth_mat.at<unsigned short>(p_y1, p_x2)) / 1000.0f;
			//float p_z22 = (float)(depth_mat.at<unsigned short>(p_y2, p_x2)) / 1000.0f;

			float p_z11 = m_depth_mat[p_y1 * 640 + p_x1];
			float p_z12 = m_depth_mat[p_y2 * 640 + p_x1];
			float p_z21 = m_depth_mat[p_y1 * 640 + p_x2];
			float p_z22 = m_depth_mat[p_y2 * 640 + p_x2];

			if (p_z11>0.0f && p_z12 >0.0f && p_z21 >0.0f && p_z22 >0.0f) //p_z11 <=1.0f && p_z12 <=1.0f && p_z21 <=1.0f && p_z22 <=1.0f
			{
				// get target point position
				float3 target = make_float3(p_x1*p_z11, p_y1*p_z11, p_z11);
				vecTrans(target, mat_K_inv);

				// get tangent and bitanget as uVec and vVec
				float3 uVec; float3 vVec;	
				uVec.x = p_x2*p_z21-p_x1*p_z11; uVec.y = p_y1*p_z21-p_y1*p_z11; uVec.z = p_z21-p_z11;
				vVec.x = p_x1*p_z12-p_x1*p_z11; vVec.y = p_y2*p_z12-p_y1*p_z11; vVec.z = p_z12-p_z11;
		
				vecTrans(uVec, mat_K_inv); vecTrans(vVec, mat_K_inv);
				
				// calculate unit normal
				float3 normVec = make_float3(uVec.y*vVec.z-uVec.z*vVec.y, uVec.z*vVec.x-uVec.x*vVec.z, uVec.x*vVec.y-uVec.y*vVec.x);
				normVec = normalize(normVec);

				// get the distance between the source point and the target point
				float3 distVec = target - currentPt;
				float dist = length(distVec);

				// get the correct normal direction
				if (normVec.z > 0) {
					normVec.x = -normVec.x; normVec.y = -normVec.y; normVec.z = -normVec.z; 
				}
				// update if it is under threshold
				// get corresponding volume index (inside TSDFVolume)
				// m_volume->get_origin() is wrong!	           
                int3 idx3 = (*m_vol_idx)[i];
                int idx1 = idx3.z * (vol_size.y * vol_size.x) + idx3.y * vol_size.x + idx3.x;
				bool check_updated = 0;
				
                if (dist < positionThreshold) {
		            if (dot(normVec, sourceNormal) > cosNormalThreshold) {
						h_vertexPosTargetFloat3[i] = target;
						h_vertexNormalTargetFloat3[i] = normVec;
		                validTargetFound = true;
		                
		                // set grid to be integratable
		                //grid_state[idx1] = 2;
		                //check_updated = 1;
		            }
		            check_updated = 1;
		            grid_state[idx1] = 2; // maybe using distance is good enough
				}
				// disable the neighbors of the grid
				if (!check_updated && grid_state[idx1]==2) {grid_state[idx1] = 0;}
			}
		}
		
		// set the target to negative infinity if unbounded
        if (!validTargetFound) {
            ++thrownOutCorrespondenceCount;
            h_vertexPosTargetFloat3[i] = invalidPt;
        }

		// update the weights
        if (m_previousConstraints[i] != h_vertexPosTargetFloat3[i]) {
            m_previousConstraints[i] = h_vertexPosTargetFloat3[i];
            ++constraintsUpdated;
            
            float3 currentPt = (*m_vertices)[i];
            float3 v = h_vertexPosTargetFloat3[i];
            float3 distVec = currentPt - v;
            float dist = length(distVec);

			// weight_d, weight_n, weight_v correspond to position, normal, camera view deviation
			float weight_d = 1.0f - dist / positionThreshold;
			float weight_n = 1.0f - (1.0f - dot(h_vertexNormalTargetFloat3[i], sourceNormal)) / cosNormalThreshold;

			// use camera view direction will make the edges unstable
			//float3 camera_view = make_float3(-0.4594f, -0.3445f, 0.8187f);
			//float weight_v = 1.0f - (1.0f - dot(h_vertexNormalTargetFloat3[i], camera_view)) / viewThreshold;

			if (weight_d >= 0 && weight_n >= 0) { //&& weight_n >= 0 && weight_v >= 0
				//float weight = (weight_d + weight_n + weight_v) / 3;
				float weight = (weight_d + weight_n) / 2;
				//float weight = weight_d;
				h_robustWeights[i] = fmaxf(0.1f, weight*0.9f+0.05f); //weight*0.9f+0.05f
			} else {
				h_robustWeights[i] = 0.0f;
			}
        }
	}

	//test
	cudaMemcpy(m_volume->get_state(), grid_state.data(), vol_size.x * vol_size.y * vol_size.z * sizeof(unsigned int), cudaMemcpyHostToDevice);

    m_vertexPosTargetFloat3->update(h_vertexPosTargetFloat3);
    m_vertexNormalTargetFloat3->update(h_vertexNormalTargetFloat3);
    m_robustWeights->update(h_robustWeights);

    std::cout << "*******Thrown out correspondence count: " << thrownOutCorrespondenceCount << std::endl;

    return constraintsUpdated;
}*/
