// ---------------------------------------------------------
// Author: Andy Zeng, Princeton University, 2016
// ---------------------------------------------------------

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
//#include <opencv2/opencv.hpp>
//#include <opencv2/gpu/gpu.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <math.h>

void write_pt_cloud( const std::string& file_name_out, const std::string& file_name, int H, int W) {
  cv::Mat depth_mat = cv::imread(file_name, CV_LOAD_IMAGE_UNCHANGED);
  if (depth_mat.empty()) {
    std::cout << "Error: depth image file not read!" << std::endl;
    cv::waitKey(0);
  }  
  float z_max = 0;
  float z_min = 3;
	std::ofstream f ;
	f.open(file_name_out);
	if( f.is_open() ) {
    for (int r = 0; r < H; ++r) {
      for (int c = 0; c < W; ++c) {
        float z = (float)(depth_mat.at<unsigned short>(r, c)) / 1000.0f - 10.0f;
        if (z>0.0f && z<1.5f) {f << "v " << c/float(W)*2 << " " << r/float(H)*2 << " " << z << std::endl;}
        z_max = std::max(z_max, z);
        if (z>0.0) z_min = std::min(z_min, z);
        /*//if (z>0.0) std::cout << "depth value: " << z << std::endl;
        if (depth[r * W + c] > 1.5f) { // Only consider depth < 1m
          depth[r * W + c] = 0;
          count++;
        }*/
      }
    }
    std::cout << "point cloud: max: " << z_max << " min: " << z_min << std::endl;    
	} else {
		std::cout << "Problem opening file for write " << file_name_out << std::endl;
	}
	f.close();
}

void write_to_ply( const std::string& file_name, const std::vector<float3>& vertices, const std::vector<int3>& triangles ) {
	std::ofstream f ;
	f.open(file_name);
	if( f.is_open() ) {
    /*
		f << "ply" << std::endl;
		f << "format ascii 1.0" << std::endl;

		f << "element vertex " << vertices.size() << std::endl;
		f << "property float x" << std::endl;
		f << "property float y" << std::endl;
		f << "property float z" << std::endl;

		f << "element face " << triangles.size() << std::endl;
		f << "property list uchar int vertex_indices" << std::endl;
		f << "end_header" << std::endl;*/

		std::vector<bool> v_include;
		for ( int v = 0; v < vertices.size(); v++ ) {
			if (vertices[v].x > 2.0f || vertices[v].x < -2.0f || vertices[v].y > 2.0f || vertices[v].y < -2.0f || vertices[v].z > 2.0f || vertices[v].z < -2.0f) {
				//f << 0.0f << " " << 0.0f << " " << 0.75f << std::endl;
        f << "v " << 0.0f << " " << 0.0f << " " << 0.75f << std::endl;
				v_include.push_back(0);
			} else {
				//f << vertices[v].x << " " << vertices[v].y << " " << vertices[v].z << std::endl;
        f << "v " << vertices[v].x << " " << vertices[v].y << " " << vertices[v].z << std::endl;	
				v_include.push_back(1);
			}
		}
		for ( int t = 0; t < triangles.size(); t++ ) {
			if (v_include[t]) {
				//f << "3 " << triangles[t].x << " " << triangles[t].y << " " << triangles[t].z << std::endl;
        f << "f " << triangles[t].x+1 << " " << triangles[t].y+1 << " " << triangles[t].z+1 << std::endl;
			}
		}
	} else {
		std::cout << "Problem opening file for write " << file_name << std::endl;
	}
	f.close();
}

float3 mat_mul_vec(float3 v, float K[9])
{
  float x = K[0] * v.x + K[1] * v.y + K[2] * v.z;
  float y = K[3] * v.x + K[4] * v.y + K[5] * v.z;
  float z = K[6] * v.x + K[7] * v.y + K[8] * v.z;
  return make_float3(x, y ,z);
}

void write_mesh( const std::string& file_name_out, const std::string& file_name, int H, int W, float K_inv_mat[9]) {
  cv::Mat depth_mat = cv::imread(file_name, CV_LOAD_IMAGE_UNCHANGED);
  if (depth_mat.empty()) {
    std::cout << "Error: depth image file not read!" << std::endl;
    cv::waitKey(0);
  }

  float z_max = 0;
  float z_min = 10;
  int count = 0;
  float threshold = 0.1;

	std::ofstream f;
	f.open(file_name_out);

  float d_scale = 2500.0;

	if( f.is_open() ) {
    for (int r = 0; r < H; ++r) {
      for (int c = 0; c < W; ++c) {
        float z = (float)(depth_mat.at<unsigned short>(r, c)) / d_scale;
        //if (z==0.0) z=3.0;

        //f << "v " << (r)/float(H)*2.0f << " " << (c)/float(W)*2.0f << " " << z << std::endl;
        float3 v = make_float3((float)c*z, (float)r*z, z);
        v = mat_mul_vec(v, K_inv_mat);
        f << "v " << v.x << " " << v.y << " " << v.z << std::endl; 

        //if (count<700) {std::cout << "r*H+c: " << r*H+c <<std::endl;count++;}
        z_max = std::max(z_max, v.z);
        if (v.z>0.0) z_min = std::min(z_min, v.z);
      }
    }
    
    for (int r = 0; r < H-2; ++r) {
      for (int c = 0; c < W-2; ++c) {
        if( r<2 || c<2) continue;

        float z0 = (float)(depth_mat.at<unsigned short>(r-1, c-1)) / d_scale;
        float z1 = (float)(depth_mat.at<unsigned short>(r-1, c  )) / d_scale;
        float z2 = (float)(depth_mat.at<unsigned short>(r-1, c+1)) / d_scale;
        float z3 = (float)(depth_mat.at<unsigned short>(r  , c-1)) / d_scale;
        float z4 = (float)(depth_mat.at<unsigned short>(r  , c  )) / d_scale;
        float z5 = (float)(depth_mat.at<unsigned short>(r  , c+1)) / d_scale;
        float z6 = (float)(depth_mat.at<unsigned short>(r+1, c-1)) / d_scale;
        float z7 = (float)(depth_mat.at<unsigned short>(r+1, c  )) / d_scale;
        float z8 = (float)(depth_mat.at<unsigned short>(r+1, c+1)) / d_scale;

        float dy_u_max = std::max({fabs(z0 - z3), fabs(z1 - z4), fabs(z2 - z5)});
        float dy_d_max = std::max({fabs(z0 - z6), fabs(z1 - z7), fabs(z2 - z8)});
        float dx_l_max = std::max({fabs(z0 - z1), fabs(z3 - z4), fabs(z6 - z7)});
        float dx_r_max = std::max({fabs(z0 - z2), fabs(z3 - z5), fabs(z6 - z8)});

        if (((dy_u_max<threshold) && (dy_d_max<threshold)) && ((dx_l_max<threshold) && (dx_r_max<threshold))) {
          if (z0>0 && z1>0 && z2>0 && z3>0 && z4>0 && z5>0 && z6>0 && z7>0 && z8>0){
              f << "f " << r*W+c+1 << " " << r*W+c+2 << " " << (r+1)*W+c+1 << std::endl;
              f << "f " << (r+1)*W+c+2 << " " << (r+1)*W+c+1 << " " << r*W+c+2 << std::endl;
          }
        }
      } 
    }
    std::cout << "mesh: max: " << z_max << " min: " << z_min << std::endl;    
	} else {
		std::cout << "Problem opening file for write " << file_name_out << std::endl;
	}
	f.close();
}

// Load an M x N matrix from a text file (numbers delimited by spaces/tabs)
// Return the matrix as a float vector of the matrix in row-major order
std::vector<float> LoadMatrixFromFile(std::string filename, int M, int N) {
  std::vector<float> matrix;
  FILE *fp = fopen(filename.c_str(), "r");
  for (int i = 0; i < M * N; i++) {
    float tmp;
    int iret = fscanf(fp, "%f", &tmp);
	//std::cout << tmp << std::endl;
    matrix.push_back(tmp);
  }
  fclose(fp);
  return matrix;
}

// Read a depth image with size H x W and save the depth values (in meters) into a float array (in row-major order)
// The depth image file is assumed to be in 16-bit PNG format, depth in millimeters
void ReadDepth(std::string filename, int H, int W, float * depth, float K_inv_mat[9], float floor_height) {
  cv::Mat depth_mat = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
  if (depth_mat.empty()) {
    std::cout << "Error: depth image file not read!" << std::endl;
    cv::waitKey(0);
  }
  int count = 0;
  float z_max = 0;
  float z_min = 3;
  float bound = 1.2;
  float d_scale = 2500.0;
  
  for (int r = 0; r < H; ++r) {
    for (int c = 0; c < W; ++c) {
      float z = (float)(depth_mat.at<unsigned short>(r, c)) / d_scale;// - 1.7;
      //float z = (float)(depth_mat.at<unsigned short>(r, c)) / 10000.0f;
      //float z = (float)(depth_mat.at<unsigned short>(r, c)) / 1000.0f * 1.1f - 11.0f;
      //if (z!=0) z += 0.45; //0.3 0.45

      depth[r * W + c] = z;
      z_max = std::max(z_max, z);
      if (z>0.0) z_min = std::min(z_min, z);
      //if (z>0.0) std::cout << "depth value: " << z << std::endl;

      if (depth[r * W + c] > bound) { // Only consider depth < 1m
        depth[r * W + c] = 0;
        count++;
        continue;
      }

      //check if floor
      float3 v = make_float3((float)c*z, (float)r*z, z);
      v = mat_mul_vec(v, K_inv_mat);
      if (v.y > floor_height) {
        depth[r * W + c] = 0;
        count++;  
        continue;    
      }

      // check if is slihouette
      /*bool slihouette_flag = 0;
      if (r > 0 && r < H-1 && c > 0 && c < W-1) {
        float z0 = (float)(depth_mat.at<unsigned short>(r+1, c+1)) / d_scale;
        float z1 = (float)(depth_mat.at<unsigned short>(r+1, c)) / d_scale;
        float z2 = (float)(depth_mat.at<unsigned short>(r+1, c-1)) / d_scale;
        float z3 = (float)(depth_mat.at<unsigned short>(r, c+1)) / d_scale;
        float z4 = (float)(depth_mat.at<unsigned short>(r, c)) / d_scale;
        float z5 = (float)(depth_mat.at<unsigned short>(r, c-1)) / d_scale;
        float z6 = (float)(depth_mat.at<unsigned short>(r-1, c+1)) / d_scale;
        float z7 = (float)(depth_mat.at<unsigned short>(r-1, c)) / d_scale;
        float z8 = (float)(depth_mat.at<unsigned short>(r-1, c+1)) / d_scale;

        if (z0==0 || z1==0 || z2==0 || z3==0 || z4==0 || z5==0 || z6==0 || z7==0 || z8==0) {
          depth[r * W + c] = 0;
          count++;    
        } else {
          depth[r * W + c] = z;
        }
      }*/
    }
  }
  std::cout << "max: " << z_max << " min: " << z_min << std::endl;
  std::cout << "number of discard depth values: " << count << std::endl;
}

void depthmap_to_mesh( const std::string& file_name_out, float* depth_map, int H, int W, float K_inv_mat[9]) {

  float z_max = 0;
  float z_min = 10;
  int count = 0;
  float threshold = 0.1;

	std::ofstream f;
	f.open(file_name_out);

	if( f.is_open() ) {
    for (int r = 0; r < H; ++r) {
      for (int c = 0; c < W; ++c) {
        float z = depth_map[r * W + c];
        //if (z==0.0) z=3.0;

        //f << "v " << (r)/float(H)*2.0f << " " << (c)/float(W)*2.0f << " " << z << std::endl;
        float3 v = make_float3((float)c*z, (float)r*z, z);
        v = mat_mul_vec(v, K_inv_mat);
        f << "v " << v.x << " " << v.y << " " << v.z << std::endl; 

        //if (count<700) {std::cout << "r*H+c: " << r*H+c <<std::endl;count++;}
        z_max = std::max(z_max, v.z);
        if (v.z>0.0) z_min = std::min(z_min, v.z);
      }
    }
    
    for (int r = 0; r < H-2; ++r) {
      for (int c = 0; c < W-2; ++c) {
        if( r<2 || c<2) continue;

        float z0 = depth_map[(r-1) * W + (c-1)];
        float z1 = depth_map[(r-1) * W + (c)];
        float z2 = depth_map[(r-1) * W + (c+1)];
        float z3 = depth_map[(r) * W + (c-1)];
        float z4 = depth_map[(r) * W + (c)];
        float z5 = depth_map[(r) * W + (c+1)];
        float z6 = depth_map[(r+1) * W + (c-1)];
        float z7 = depth_map[(r+1) * W + (c)];
        float z8 = depth_map[(r+1) * W + (c+1)];

        float dy_u_max = std::max({fabs(z0 - z3), fabs(z1 - z4), fabs(z2 - z5)});
        float dy_d_max = std::max({fabs(z0 - z6), fabs(z1 - z7), fabs(z2 - z8)});
        float dx_l_max = std::max({fabs(z0 - z1), fabs(z3 - z4), fabs(z6 - z7)});
        float dx_r_max = std::max({fabs(z0 - z2), fabs(z3 - z5), fabs(z6 - z8)});

        if (((dy_u_max<threshold) && (dy_d_max<threshold)) && ((dx_l_max<threshold) && (dx_r_max<threshold))) {
          if (z0>0 && z1>0 && z2>0 && z3>0 && z4>0 && z5>0 && z6>0 && z7>0 && z8>0){
              f << "f " << r*W+c+1 << " " << r*W+c+2 << " " << (r+1)*W+c+1 << std::endl;
              f << "f " << (r+1)*W+c+2 << " " << (r+1)*W+c+1 << " " << r*W+c+2 << std::endl;
          }
        }
      } 
    }
    std::cout << "mesh: max: " << z_max << " min: " << z_min << std::endl;    
	} else {
		std::cout << "Problem opening file for write " << file_name_out << std::endl;
	}
	f.close();
}

// 4x4 matrix multiplication (matrices are stored as float arrays in row-major order)
void multiply_matrix(const float m1[16], const float m2[16], float mOut[16]) {
  mOut[0]  = m1[0] * m2[0]  + m1[1] * m2[4]  + m1[2] * m2[8]   + m1[3] * m2[12];
  mOut[1]  = m1[0] * m2[1]  + m1[1] * m2[5]  + m1[2] * m2[9]   + m1[3] * m2[13];
  mOut[2]  = m1[0] * m2[2]  + m1[1] * m2[6]  + m1[2] * m2[10]  + m1[3] * m2[14];
  mOut[3]  = m1[0] * m2[3]  + m1[1] * m2[7]  + m1[2] * m2[11]  + m1[3] * m2[15];

  mOut[4]  = m1[4] * m2[0]  + m1[5] * m2[4]  + m1[6] * m2[8]   + m1[7] * m2[12];
  mOut[5]  = m1[4] * m2[1]  + m1[5] * m2[5]  + m1[6] * m2[9]   + m1[7] * m2[13];
  mOut[6]  = m1[4] * m2[2]  + m1[5] * m2[6]  + m1[6] * m2[10]  + m1[7] * m2[14];
  mOut[7]  = m1[4] * m2[3]  + m1[5] * m2[7]  + m1[6] * m2[11]  + m1[7] * m2[15];

  mOut[8]  = m1[8] * m2[0]  + m1[9] * m2[4]  + m1[10] * m2[8]  + m1[11] * m2[12];
  mOut[9]  = m1[8] * m2[1]  + m1[9] * m2[5]  + m1[10] * m2[9]  + m1[11] * m2[13];
  mOut[10] = m1[8] * m2[2]  + m1[9] * m2[6]  + m1[10] * m2[10] + m1[11] * m2[14];
  mOut[11] = m1[8] * m2[3]  + m1[9] * m2[7]  + m1[10] * m2[11] + m1[11] * m2[15];

  mOut[12] = m1[12] * m2[0] + m1[13] * m2[4] + m1[14] * m2[8]  + m1[15] * m2[12];
  mOut[13] = m1[12] * m2[1] + m1[13] * m2[5] + m1[14] * m2[9]  + m1[15] * m2[13];
  mOut[14] = m1[12] * m2[2] + m1[13] * m2[6] + m1[14] * m2[10] + m1[15] * m2[14];
  mOut[15] = m1[12] * m2[3] + m1[13] * m2[7] + m1[14] * m2[11] + m1[15] * m2[15];
}

// 4x4 matrix inversion (matrices are stored as float arrays in row-major order)
bool invert_matrix(const float m[16], float invOut[16]) {
  float inv[16], det;
  int i;
  inv[0] = m[5]  * m[10] * m[15] -
           m[5]  * m[11] * m[14] -
           m[9]  * m[6]  * m[15] +
           m[9]  * m[7]  * m[14] +
           m[13] * m[6]  * m[11] -
           m[13] * m[7]  * m[10];

  inv[4] = -m[4]  * m[10] * m[15] +
           m[4]  * m[11] * m[14] +
           m[8]  * m[6]  * m[15] -
           m[8]  * m[7]  * m[14] -
           m[12] * m[6]  * m[11] +
           m[12] * m[7]  * m[10];

  inv[8] = m[4]  * m[9] * m[15] -
           m[4]  * m[11] * m[13] -
           m[8]  * m[5] * m[15] +
           m[8]  * m[7] * m[13] +
           m[12] * m[5] * m[11] -
           m[12] * m[7] * m[9];

  inv[12] = -m[4]  * m[9] * m[14] +
            m[4]  * m[10] * m[13] +
            m[8]  * m[5] * m[14] -
            m[8]  * m[6] * m[13] -
            m[12] * m[5] * m[10] +
            m[12] * m[6] * m[9];

  inv[1] = -m[1]  * m[10] * m[15] +
           m[1]  * m[11] * m[14] +
           m[9]  * m[2] * m[15] -
           m[9]  * m[3] * m[14] -
           m[13] * m[2] * m[11] +
           m[13] * m[3] * m[10];

  inv[5] = m[0]  * m[10] * m[15] -
           m[0]  * m[11] * m[14] -
           m[8]  * m[2] * m[15] +
           m[8]  * m[3] * m[14] +
           m[12] * m[2] * m[11] -
           m[12] * m[3] * m[10];

  inv[9] = -m[0]  * m[9] * m[15] +
           m[0]  * m[11] * m[13] +
           m[8]  * m[1] * m[15] -
           m[8]  * m[3] * m[13] -
           m[12] * m[1] * m[11] +
           m[12] * m[3] * m[9];

  inv[13] = m[0]  * m[9] * m[14] -
            m[0]  * m[10] * m[13] -
            m[8]  * m[1] * m[14] +
            m[8]  * m[2] * m[13] +
            m[12] * m[1] * m[10] -
            m[12] * m[2] * m[9];

  inv[2] = m[1]  * m[6] * m[15] -
           m[1]  * m[7] * m[14] -
           m[5]  * m[2] * m[15] +
           m[5]  * m[3] * m[14] +
           m[13] * m[2] * m[7] -
           m[13] * m[3] * m[6];

  inv[6] = -m[0]  * m[6] * m[15] +
           m[0]  * m[7] * m[14] +
           m[4]  * m[2] * m[15] -
           m[4]  * m[3] * m[14] -
           m[12] * m[2] * m[7] +
           m[12] * m[3] * m[6];

  inv[10] = m[0]  * m[5] * m[15] -
            m[0]  * m[7] * m[13] -
            m[4]  * m[1] * m[15] +
            m[4]  * m[3] * m[13] +
            m[12] * m[1] * m[7] -
            m[12] * m[3] * m[5];

  inv[14] = -m[0]  * m[5] * m[14] +
            m[0]  * m[6] * m[13] +
            m[4]  * m[1] * m[14] -
            m[4]  * m[2] * m[13] -
            m[12] * m[1] * m[6] +
            m[12] * m[2] * m[5];

  inv[3] = -m[1] * m[6] * m[11] +
           m[1] * m[7] * m[10] +
           m[5] * m[2] * m[11] -
           m[5] * m[3] * m[10] -
           m[9] * m[2] * m[7] +
           m[9] * m[3] * m[6];

  inv[7] = m[0] * m[6] * m[11] -
           m[0] * m[7] * m[10] -
           m[4] * m[2] * m[11] +
           m[4] * m[3] * m[10] +
           m[8] * m[2] * m[7] -
           m[8] * m[3] * m[6];

  inv[11] = -m[0] * m[5] * m[11] +
            m[0] * m[7] * m[9] +
            m[4] * m[1] * m[11] -
            m[4] * m[3] * m[9] -
            m[8] * m[1] * m[7] +
            m[8] * m[3] * m[5];

  inv[15] = m[0] * m[5] * m[10] -
            m[0] * m[6] * m[9] -
            m[4] * m[1] * m[10] +
            m[4] * m[2] * m[9] +
            m[8] * m[1] * m[6] -
            m[8] * m[2] * m[5];

  det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

  if (det == 0)
    return false;

  det = 1.0 / det;

  for (i = 0; i < 16; i++)
    invOut[i] = inv[i] * det;

  return true;
}

void FatalError(const int lineNumber = 0) {
  std::cerr << "FatalError";
  if (lineNumber != 0) std::cerr << " at LINE " << lineNumber;
  std::cerr << ". Program Terminated." << std::endl;
  cudaDeviceReset();
  exit(EXIT_FAILURE);
}

void checkCUDA(const int lineNumber, cudaError_t status) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA failure at LINE " << lineNumber << ": " << status << std::endl;
    FatalError();
  }
}
