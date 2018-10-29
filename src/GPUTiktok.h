#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>

namespace util
{

    class GPUTiktok {
		private:

		    cudaEvent_t start_, stop_;

		public:

		GPUTiktok(){
		    cudaEventCreate(&start_);
		    cudaEventCreate(&stop_);
		}


		~GPUTiktok(){
		    cudaEventDestroy(start_);
		    cudaEventDestroy(stop_);
		}

		void tik(){
		    cudaEventRecord(start_);
		}

		void tok(){
		    cudaEventRecord(stop_);
		    cudaEventSynchronize(stop_);
		}

		double toSeconds(){
		    float milliseconds = 0;
		    cudaEventElapsedTime(&milliseconds, start_, stop_);
		    return (double)milliseconds/1000.f;
		}

		double toMilliseconds(){
		    float milliseconds = 0;
		    cudaEventElapsedTime(&milliseconds, start_, stop_);
		    return (double)milliseconds;
		}

    };

}

