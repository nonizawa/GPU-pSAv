#include <cuda_runtime.h>

extern "C"
{
    __global__ void calculate_cut_val(int vertex, int8_t* J_matrix, int* spin_vector, float* cut_val) 
    {
        int idx = blockIdx.y * blockDim.y + threadIdx.y;
        int stride = blockDim.y * gridDim.y;

        // Use shared memory for intermediate sum reduction
        extern __shared__ float shared_cut[]; 
        shared_cut[threadIdx.y] = 0.0f;

        for (int i = idx; i < vertex; i += stride) {
            for (int j = i + 1; j < vertex; j++) {
                shared_cut[threadIdx.y] -= static_cast<int>(J_matrix[i * vertex + j]) * (1.0f - spin_vector[i] * spin_vector[j]);
            }
        }
        __syncthreads();

        // Perform reduction to sum the values in shared memory
        if (threadIdx.y == 0) {
            float block_sum = 0.0f;
            for (int k = 0; k < blockDim.y; k++) {
                block_sum += shared_cut[k]/2;
            }
            atomicAdd(cut_val, block_sum);
        }  
    }

    
    __global__ void annealing_module(int mean_range, int vertex, float mem_I0, int8_t *h_vector, int8_t *J_matrix, int *spin_vector, float *rnd, float *D_res)
    {
        extern __shared__ float shared_D_res[];

        int b, c;
        int i, k;
        float Itanh;
        float avg_D_res;

        i = blockIdx.y * blockDim.y + threadIdx.y;

        if (i < vertex)
        {
            // グローバルメモリから共有メモリにコピー
            for (c = 0; c < mean_range; c++) {
                shared_D_res[threadIdx.y * mean_range + c] = D_res[i * mean_range + c];
            }
            __syncthreads();

            //for(b=0; b<tau; b++){
                shared_D_res[threadIdx.y * mean_range] = h_vector[i];
                avg_D_res = 0;
                Itanh = 0;
                __syncthreads();
                
                for(k=0; k<vertex; k++){
                    shared_D_res[threadIdx.y * mean_range] += static_cast<int>(J_matrix[i * vertex + k]) * spin_vector[k];
                }

                
                    for(c=0; c<mean_range; c++) {
                        avg_D_res += shared_D_res[threadIdx.y * mean_range + c];
                    }
                    avg_D_res /= mean_range;
                
                                                    
                Itanh = tanh(mem_I0 * avg_D_res) + rnd[i];
                spin_vector[i] = (Itanh > 0) ? 1 : -1;
                
                
                    for (c = mean_range - 1; c > 0; c--) {
                        shared_D_res[threadIdx.y * mean_range + c] = shared_D_res[threadIdx.y * mean_range + c - 1];
                    }
                
                __syncthreads();

                // 共有メモリからグローバルメモリにコピー
                for (c = 0; c < mean_range; c++) {
                    D_res[i * mean_range + c] = shared_D_res[threadIdx.y * mean_range + c];
                }

            //}
        }
    }
}
