# GPU-pSAv (GPU-accelerated p-bit-based simulated annealing with device variability)

[![GitHub license]](https://github.com/nonizawa/GPU-pSAv/blob/main/LICENSE)

Probabilistic computing based on probabilistic bits (p-bits) offers a more efficient approach than conventional CMOS logic and has shown great potential for solving complex problems, such as simulated annealing and machine learning. However, the physical realization of p-bits, often using emerging devices like Magnetic Tunnel Junctions (MTJs), introduces variability in device characteristics. In this article, we present an open-source, GPU-accelerated simulated annealing algorithm, developed using Python and PyCUDA, tailored for large-scale p-bit computing. By incorporating a model that accounts for device variability, our framework enables simulation and evaluation of how different types of variability affect computational performance. Compared to CPU implementations, our GPU-based approach demonstrates a two-order magnitude speedup when applied to the MAX-CUT benchmark, which includes problems ranging from 800 to 20,000 nodes. Our open-source solution aims to provide a scalable and accessible tool for the research community, facilitating advancements in probabilistic computing and optimization across various fields.

## Installation

### Prerequisites

- Python 3.6.8
- PyCUDA 2022.1
- CUDA 12.2

### Clone the Repository

To get started, clone the repository using git:

```sh
git clone https://github.com/nonizawa/GPU-pSAv.git
cd GPU-pSAv
```

## Structure

- `gpu_MAXCUT.py`: This is the Python script that runs the variations of pSAs without device variability for MAX-CUT algorithm.
- `gpu_MAXCUT_var.py`: This is the Python script that runs the variations of pSAs with device variabilityfor MAX-CUT algorithm.
- `./graph/`: This directory contains the MAX-CUT dataset of graphs used for evaluation.
- `./result/`: This directory contains the evaluation results generated using simulation.

## Single run without device variability

To run the pSA algorithms on a single instance without device variability, use the gpu_MAXCUT.py script. For example with G1:

```sh
python3 gpu_MAXCUT.py  --mean_range $mean_range --stall_prop $stall_prop  --param $PARAM --config $CONFIG --cycle $CYCLE  --trial $TRIAL --tau $TAU  --gpu $GPU_DEVICE --file_path ./graph/G1.txt
```

You can find the simulation results in ./result/. The result csv includes simulation retsuls, such as mean cut values and simulation time. There are three pSA algorithsm: pSA, TApSA (time-average pSA), and SpSA (stalled pSA).

Here ia the options.

- `--file_path`: a graph file

- `--cycle`: Number of cycles for 1 trial

- `--trial`: Number of trials to evaluate the performance on average

- `--tau`:  A pseudo inverse temperature is increased every tau cycle

- `--config`:  A configuration for pSA algorithms: 2 for pSA, 3 for TApSA, and 4 for SpSA

- `--param`: Hyperparameters for pSA algorithsms: 2 for pSA, TApSA< and SpSA 

- `--mean_range`: The size of time window for averaing (used for only TApSA)

- `--stall_prop`: The probability of p-bits stalled (used for only SpSA)

## Single run with device variability

To run the pSA algorithms on a single instance with device variability, use the gpu_MAXCUT.py script. For example with G1:

```sh
python3 gpu_MAXCUT_var.py  --res $RES --l_scale $l_scale --d_scale $d_scale --n_scale $n_scale --mean_range $mean_range --stall_prop $stall_prop  --param $PARAM --config $CONFIG --cycle $CYCLE  --trial $TRIAL --tau $TAU  --gpu $GPU_DEVICE --file_path ./graph/G1.txt
```

Here ia the options.

- `--res`: Time resolution of 1 cycle of updaing the states of p-bits

- `--l_scale`: A standard deviation of lambda that represents the input sensitivity of p-bits

- `--d_scale`: A standard deviation of delta that represents the input offset of p-bits

- `--n_scale`: A standard deviation of nu that represents the updated timing of p-bits

## Contact

For any questions, issues, or inquiries, feel free to create an issue in the repository or contact the repository owner [@nonizawa](https://github.com/nonizawa).

## Citation

## License

This project is licensed under the modified MIT License.
