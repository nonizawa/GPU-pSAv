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

    __global__ void annealing_module(int vertex, float mem_I0, int8_t *h_vector, int8_t *J_matrix, int *spin_vector, float *rnd, float *lambda, float *delta, int *nu, int count_device)
    {

        int i, k;
        float D_res;

        i = blockIdx.y * blockDim.y + threadIdx.y;
        //j = blockIdx.x * blockDim.x + threadIdx.x; 
        
        if (i < vertex)
        {
            if (count_device % nu[i] == 0)
            {
                    D_res = h_vector[i];
                    __syncthreads();
                    for(k=0; k<vertex; k++){
                        D_res += static_cast<int>(J_matrix[i * vertex + k]) * spin_vector[k];
                    }
                
                    float Itanh = tanh(lambda[i] * mem_I0 * (D_res + delta[i]))  + rnd[i];
                    spin_vector[i] = (Itanh > 0) ? 1 : -1;
                    
                    __syncthreads();

            }
            
        }
    }
}