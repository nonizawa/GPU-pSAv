import math
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import random
import time
import statistics
import csv
import os
import sys
import argparse
import cProfile
import pstats
import io

def parse_arguments():
    parser = argparse.ArgumentParser(description="Specify GPU device number")
    parser.add_argument('--gpu', type=int, default=0, help="GPU device number (default: 0)")
    parser.add_argument('--file_path', type=str, required=True, help="File path for data")
    parser.add_argument('--param', type=int, default=1, help="Parameter type (default: 1)")
    parser.add_argument('--cycle', type=int, default=1000, help="Number of cycles (default: 1000)")
    parser.add_argument('--trial', type=int, default=100, help="Number of trials (default: 100)")
    parser.add_argument('--tau', type=int, default=1, help="tau (default: 1)")
    parser.add_argument('--thread', type=int, default=32, help="Number of threads (default: 32)")
    parser.add_argument('--config', type=int, default=1, help="Configuration (default: 1)")
    parser.add_argument('--unique', type=int, default=1, help="Unique noise magnitude (default: 1)")
    parser.add_argument('--mean_range', type=int, default=4, help="Configuration (default: 1)")
    parser.add_argument('--stall_prop', type=float, default=0.5, help='stalled prop値')
    return parser.parse_args()

def initialize_cuda(args):
    cuda.Device(args.gpu).make_context()
    cuda.init()
    device = cuda.Device(args.gpu)
    properties = device.get_attributes()
    max_shared_mem = properties[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK]
    print(f"Maximum shared memory size: {max_shared_mem} bytes")
    return device

def load_gpu_code(config):
    if config == 1:
        cu_file = 'ssa_annealing_kernel.cu'
    elif config == 2:
        cu_file = 'psa_annealing_kernel.cu'
    elif config == 3:
        cu_file = 'tapsa_annealing_kernel.cu'
    elif config == 4:
        cu_file = 'spsa_annealing_kernel.cu'

    with open(cu_file, 'r') as file:
        gpu_code = file.read()
    return gpu_code

def read_file_MAXCUT(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
        third_line = f.readline().strip()
        fourth_line = f.readline().strip()
        lines = f.readlines()
    return first_line, second_line, third_line, fourth_line, lines

# Function to calculate the number of cuts
def cut_calculate(vertex, G_matrix, spin_vector):
    spin_vector_reshaped = np.reshape(spin_vector, (len(spin_vector),)) # Convert spin_vector to 1D array
    upper_triangle = np.triu_indices(len(spin_vector), k=1) # Get the indices of the upper triangular matrix
    cut_val = np.sum(G_matrix[upper_triangle] * (1 - np.outer(spin_vector_reshaped, spin_vector_reshaped)[upper_triangle])) # Calculate cut_val by multiplying the upper triangular elements with the corresponding entries in G_matrix
    return int(cut_val / 2)

# Function to calculate the energy
def energy_calculate(vertex, h_vector, J_matrix, spin_vector):
    h_energy = np.sum(h_vector * spin_vector)
    J_energy = np.sum((np.dot(J_matrix, spin_vector) - h_vector) * spin_vector) / 2
    return -(J_energy + h_energy)

def get_graph_MAXCUT(vertex, lines):
    G_matrix = np.zeros((vertex, vertex), int)
    
    # Count the number of lines (edges)
    line_count = len(lines)
    print('Number of Edges:', line_count)
    
    # Iterate through the lines to construct the adjacency matrix
    for line_text in lines:
        weight_list = list(map(int, line_text.split(' ')))  # Convert space-separated string to list of integers
        i = weight_list[0] - 1
        j = weight_list[1] - 1
        G_matrix[i, j] = weight_list[2]  # Assign weight to the corresponding entry in the matrix
    
    # Adding the matrix to its transpose to make it symmetric
    G_matrix = G_matrix + G_matrix.T
    return G_matrix

def set_annealing_parameters(vertex, args, h_vector, J_matrix):
    mean_each = []
    std_each = []
    for j in range(vertex):
        mean_each.append((vertex-1) * np.mean(J_matrix[j]))
        std_each.append(np.sqrt((vertex-1) * np.var(np.concatenate([J_matrix[j], -J_matrix[j]]))))
    sigma = np.mean(std_each)
    mean = np.mean(mean_each)

    min_cycle = np.int32(args.cycle)
    trial = np.int32(args.trial)
    tau = np.int32(args.tau)
    Mshot = np.int32(1)

    sigma_vector = np.array(std_each, dtype=np.float32).reshape((-1, 1))
    unique = args.unique
    if unique == 0:
        nrnd_vector = np.float32(0.67448975 * np.mean(sigma_vector) * np.ones((vertex, 1)))
    elif unique == 1:
        nrnd_vector = np.float32(0.67448975 * sigma_vector)
    nrnd_max = np.max(nrnd_vector)

    param = args.param
    if param == 1:
        I0_min = np.float32(np.max(sigma_vector) * 0.01 + np.min(np.abs(mean_each)))
        I0_max = np.float32(np.max(sigma_vector) * 2 + np.min(np.abs(mean_each)))
    elif param == 2: 
        I0_min = np.float32(0.1/sigma)
        I0_max = np.float32(10/sigma)

    beta = np.float32((I0_min / I0_max) ** (tau / (min_cycle - 1)))
    max_cycle = math.ceil((math.log10(I0_min / I0_max) / math.log10(beta))) * tau

    threads_per_block = (1, args.thread, 1)
    block_x = math.ceil(1 / threads_per_block[0])
    block_y = math.ceil(vertex / threads_per_block[1])
    blocks_per_grid = (block_x, block_y, 1)

    return min_cycle, trial, tau, Mshot, sigma_vector, nrnd_vector, I0_min, I0_max, beta, max_cycle, threads_per_block, blocks_per_grid

def cleanup_cuda():
    cuda.Context.pop()

def check_cuda_error(message=""):
    try:
        cuda.Context.synchronize()  # 同期を行い、エラーチェックをする
    except pycuda._driver.Error as e:
        raise RuntimeError(f"CUDA error during {message}: {str(e)}")

def run_trials(stall_prop, mean_range, file_base, config, vertex, min_cycle, trial, tau, Mshot, gpu_code, h_vector, G_matrix, J_matrix, sigma_vector, nrnd_vector, I0_min, I0_max, beta, max_cycle, threads_per_block, blocks_per_grid):
    # Add compiler flags
    nvcc_flags = ["-std=c++14", "--compiler-options", "-fno-strict-aliasing"]
    # Load the module with the modified compilation command
    mod = SourceModule(gpu_code, options=nvcc_flags, no_extern_c=True)
    annealing_kernel = mod.get_function("annealing_module")
    cut_calculate_kernel = mod.get_function("calculate_cut_val")

    print('Number of trials:', trial)
    print("Min Cycles:", min_cycle)
    print('beta:', beta)
    print('I0_min:', I0_min)
    print('I0_max:', I0_max)
    print('tau:', tau)
    #print('nrnd vector', nrnd_vector)

    if config == 1:
        nrnd_vector_gpu = cuda.mem_alloc(nrnd_vector.nbytes)
        cuda.memcpy_htod(nrnd_vector_gpu, nrnd_vector)
        #check_cuda_error("memcpy_htod for nrnd_vector_gpu")

    h_vector_int8 = h_vector.astype(np.int8)
    h_vector_gpu = cuda.mem_alloc(h_vector_int8.nbytes)
    cuda.memcpy_htod(h_vector_gpu, h_vector_int8)
    #check_cuda_error("memcpy_htod for h_vector_gpu")

    J_matrix_int8 = J_matrix.astype(np.int8)
    J_matrix_gpu = cuda.mem_alloc(J_matrix_int8.nbytes)
    cuda.memcpy_htod(J_matrix_gpu, J_matrix_int8)
    #check_cuda_error("memcpy_htod for J_matrix_gpu")

    time_list = [] 
    cut_list = []
    energy_list = []
    for k in range(trial + 1):
        print("######## Trial", k + 1, " ###########")

        spin_vector = (np.random.randint(0, 2, (vertex, 1)) * 2 - 1).astype(np.int32)
        spin_vector_gpu = cuda.mem_alloc(spin_vector.nbytes)
        cuda.memcpy_htod(spin_vector_gpu, spin_vector)
        #check_cuda_error("memcpy_htod for spin_vector_gpu")

        if config == 1:
            rnd_ini = (np.random.randint(0, 2, (vertex, 1)) * 2 - 1).astype(np.float32)
        elif config == 2 or config == 3 or config == 4:
            rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)

        rnd_ini_gpu = cuda.mem_alloc(rnd_ini.nbytes)
        #check_cuda_error("memcpy_htod for rnd_ini_gpu")

        if config == 1:
            Itanh_ini = np.zeros((vertex, 1), dtype=np.float32)
            Itanh_ini_gpu = cuda.mem_alloc(Itanh_ini.nbytes)
            cuda.memcpy_htod(Itanh_ini_gpu, Itanh_ini)
            #check_cuda_error("memcpy_htod for Itanh_ini_gpu")
        elif config == 3:
            #count = 0
            D_res = np.zeros((vertex*mean_range, 1), dtype=np.float32)
            D_res_gpu = cuda.mem_alloc(D_res.nbytes)
            cuda.memcpy_htod(D_res_gpu, D_res)
            #check_cuda_error("memcpy_htod for D_res_gpu")
            shared_mem_size = 4 * threads_per_block[1] * mean_range


        cut_val = np.zeros(1, dtype=np.float32)
        cut_val_gpu = cuda.mem_alloc(cut_val.nbytes)
        cuda.memcpy_htod(cut_val_gpu, cut_val)
        #check_cuda_error("memcpy_htod for cut_val_gpu")

        time_start_gpu = cuda.Event()
        time_end_gpu = cuda.Event()
        time_start_gpu.record()
        I0 = I0_min

        while I0 <= I0_max:
            for i in range(tau):
                if config == 1:
                    rnd_ini = (np.random.randint(0, 2, (vertex, 1)) * 2 - 1).astype(np.float32)
                    cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                    #("memcpy_htod for rnd_ini_gpu")
                    annealing_kernel(np.int32(vertex), I0, h_vector_gpu, J_matrix_gpu, spin_vector_gpu, Itanh_ini_gpu, rnd_ini_gpu, nrnd_vector_gpu,
                                    block=threads_per_block, grid=blocks_per_grid)
                    #check_cuda_error("annealing_kernel execution for config 1")
                elif config == 2:
                    rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)
                    cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                    #check_cuda_error("memcpy_htod for rnd_ini_gpu")
                    annealing_kernel(np.int32(vertex), I0, h_vector_gpu, J_matrix_gpu, spin_vector_gpu, rnd_ini_gpu,
                                    block=threads_per_block, grid=blocks_per_grid)
                    #check_cuda_error("annealing_kernel execution for config 2")
                elif config == 3:
                    rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)
                    cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                    #heck_cuda_error("memcpy_htod for rnd_ini_gpu")
                    annealing_kernel(np.int32(mean_range), np.int32(vertex), I0,  h_vector_gpu, J_matrix_gpu, spin_vector_gpu, rnd_ini_gpu, D_res_gpu,
                                    block=threads_per_block, grid=blocks_per_grid, shared=shared_mem_size)
                    #check_cuda_error("annealing_kernel execution for config 3")
                    #count = count + 1
                elif config == 4:
                    rnd_ini = (2.0 * np.random.rand(vertex,1) - 1.0).astype(np.float32)
                    cuda.memcpy_htod(rnd_ini_gpu, rnd_ini)
                    #check_cuda_error("memcpy_htod for rnd_ini_gpu")
                    annealing_kernel(np.float32(stall_prop), np.int32(vertex), I0,  h_vector_gpu, J_matrix_gpu, spin_vector_gpu, rnd_ini_gpu,
                                    block=threads_per_block, grid=blocks_per_grid)
                    #check_cuda_error("annealing_kernel execution for config 4")
            I0 /= beta
        time_end_gpu.record()
        time_end_gpu.synchronize()
        annealing_time = time_start_gpu.time_till(time_end_gpu)

        last_spin_vector = np.empty_like(spin_vector)
        cuda.memcpy_dtoh(last_spin_vector, spin_vector_gpu)
        
        #cut_val = cut_calculate(vertex, G_matrix, last_spin_vector)
        cut_calculate_kernel(np.int32(vertex), J_matrix_gpu, spin_vector_gpu, cut_val_gpu, block=threads_per_block, grid=blocks_per_grid)
        cuda.memcpy_dtoh(cut_val, cut_val_gpu)
        
        #time_end_gpu.record()
        #time_end_gpu.synchronize()
        #annealing_time = time_start_gpu.time_till(time_end_gpu)

        min_energy = energy_calculate(vertex, h_vector, J_matrix, last_spin_vector)

        cut_list.append(int(cut_val[0]))
        time_list.append(annealing_time)
        energy_list.append(min_energy)

        print('Graph:', file_base)
        print('Time:', annealing_time)
        print('Cut value:', cut_val)
        print('Ising Energy:', min_energy)

    return cut_list, time_list, energy_list

def save_results(best_known, trial, cut_list, time_list, energy_list, args, file_base, first_line, second_line, third_line, fourth_line,  sigma_vector, I0_min, I0_max, total_time):
    #-----------Statistical processing-----------
    del cut_list[0]
    del time_list[0]
    del energy_list[0]

    cut_average = sum(cut_list) / trial
    cut_max = max(cut_list)
    cut_min = min(cut_list)
    std_cut = statistics.stdev(cut_list)
    time_average = sum(time_list) / trial
    energy_average = sum(energy_list) / trial
    P = sum(1 for x in cut_list if x > best_known * 0.99) / trial
    if P == 1:
        TTS = time_average
    elif P == 0:
        TTS = 'None'
    else:
        TTS = time_average * math.log(1 - 0.99) / math.log(1 - P)

    print('######################## Final result #######################')
    print('Average cut:', cut_average)
    print('Maximum cut:', cut_max)
    print('Minimum cut:', cut_min)
    print('Average annealing time:', time_average, "[ms]")
    print('Average energy:', energy_average)
    print('TTS(0.99):', TTS)
    print('Average reachability [%]:', 100 * cut_average / best_known)
    print('Maximum reachability [%]:', 100 * cut_max / best_known)
    print('Std of cut value:', std_cut)

    if args.config == 1:
        alg = 'SSA'
    elif args.config == 2:
        alg = 'pSA'
    elif args.config == 3:
        alg = 'TApSA'
    elif args.config == 4:
        alg = 'SpSA'

    csv_file_name1 = f'./result/{alg}_result_unique{args.unique}_config{args.config}_cycle{args.cycle}_trial{args.trial}_tau{args.tau}_thread{args.thread}_param{args.param}.csv'
    csv_file_name2 = f'./result/{alg}_cut_unique{args.unique}_config{args.config}_cycle{args.cycle}_trial{args.trial}_tau{args.tau}_thread{args.thread}_param{args.param}.csv'

    data = [
        file_base, first_line, second_line, third_line, fourth_line, cut_average, cut_max, cut_min, std_cut, 0.67448975 * np.mean(sigma_vector), I0_min, I0_max, 100 * cut_average / int(fourth_line), 100 * cut_max / int(fourth_line), time_average, total_time, args.mean_range, args.stall_prop]

    if os.path.isfile(csv_file_name1):
        with open(csv_file_name1, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)
    else:
        with open(csv_file_name1, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Gset', 'number of edges', 'edge value', 'edge type', 'best-known value', 'mean_cut_value', 'max_cut_value', 'min_cut_value', 'std_cut', 'n_rnd', 'I0_min', 'I0_max', 'ratio of mean/best', 'ratio of max/best', '1 annealing_time [ms]', 'Total time [s]', 'mean_range', 'stall_prop'])
            writer.writerow(data)

    with open(csv_file_name2, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([cut_average, cut_max, cut_min, std_cut, time_average])

def main():
    starttime = time.time()
    rs = time.time()
    #verbose = True

    args = parse_arguments()
    device = initialize_cuda(args)
    config = args.config
    gpu_code = load_gpu_code(config)
    
    first_line, second_line, third_line, fourth_line, lines = read_file_MAXCUT(args.file_path)
    vertex = int(first_line)
    G_matrix = get_graph_MAXCUT(vertex, lines)
    J_matrix = (-G_matrix).astype(np.int32)
    h_vector = np.reshape(np.diag(J_matrix), (vertex, 1))
    file_path = args.file_path
    dir_path, file_name = os.path.split(file_path)
    file_base, file_ext = os.path.splitext(file_name)

    re = time.time()
    print('File reading time:', re - rs, '[s]')
    
    best_known = np.int32(fourth_line)

    min_cycle, trial, tau, Mshot, sigma_vector, nrnd_vector, I0_min, I0_max, beta, max_cycle, threads_per_block, blocks_per_grid = set_annealing_parameters(vertex, args, h_vector, J_matrix)

    print(f"blocks_per_grid: {blocks_per_grid}")
    print(f"threads_per_block: {threads_per_block}")
    print(f"vertex: {vertex}, tau: {tau}, beta: {beta}, I0_min: {I0_min}, I0_max: {I0_max}")

    try:
        cut_list, time_list, energy_list = run_trials(args.stall_prop, args.mean_range, file_base, config, vertex, min_cycle, trial, tau, Mshot,  gpu_code, h_vector, G_matrix, J_matrix, sigma_vector, nrnd_vector, I0_min, I0_max, beta, max_cycle, threads_per_block, blocks_per_grid)
    finally:
        cleanup_cuda()

    total_time = time.time() - starttime
    save_results(best_known, trial, cut_list, time_list, energy_list, args, file_base, first_line, second_line, third_line, fourth_line, sigma_vector, I0_min, I0_max, total_time)
    print("Total time:", total_time, '[s]')

if __name__ == "__main__":
    #pr = cProfile.Profile()
    #pr.enable()
    main()
    #pr.disable()
    #s = io.StringIO()
    #sortby = 'cumulative'
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())