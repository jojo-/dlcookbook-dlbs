# A quick tutorial for DLBS

This tutorial will show you how to benchmark the following deep learning workload:
- Training using `n` GPUs with Pytorch 1.11 and Tensorflow 1.15
- Inference with 1 GPU with  Pytorch 1.11, Tensorflow 1.15 and TensorRT 8
- Models: resnet50, resnet101, resnet200, vgg19, acoustic_model (see https://hewlettpackard.github.io/dlcookbook-dlbs/#/models/models?id=supported-models)
- precision for training: FP16 and FP32
- inference precision: FP16, FP32 and INT8 (for TensorRT only)
- Batch size: 16 per GPU

DLBS will report two common notions of scalability in high performance computing:
- Strong scaling is defined as how the solution time varies with the number of processors for a fixed total problem size.
- Weak scaling is defined as how the solution time varies with the number of processors for a fixed problem size per processor.

## Installation notes

### Requirements

- Ubuntu 20.04 LTS
- python 3.8 and the following modules and their dependencies (using `pip3 install <module>`):
   - matplotlib
   - pandas
- `python-is-python3` package: (`apt-get install python-is-python3`)
- latest NVIDIA drivers
- NVIDIA docker runtime engine (see https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Clone the repository

`git clone https://github.com/jojo-/dlcookbook-dlbs.git dlbs`

### Build the containers containing the Deep Learning frameworks

Use the latest docker containers images for the frameworks to test 

1.  Pull the following NGC container images for Tensorflow and Pytorch

    ```
    docker pull nvcr.io/nvidia/tensorflow:22.01-tf1-py3
    docker pull nvcr.io/nvidia/pytorch:22.01-py3
    ``` 

2.  Build TensorRT docker image
    ```
    cd /path/to/dlbs
    cd docker/tensorrt/21.08
    cp -r ../../../src/tensorrt .
    docker build . --tag trtbench:22.01
    ```

## Running the benchmarks


1. Setting the environment variables

    ```
    cd /path/to/dlbs
    source scripts/environment.sh
    ```

2. Selecting the gpus to benchmark (here, we will first run the benchmark experiment with GPU 0, then with GPUs 0 and 1)
    ```
    export gpus_bench='"0", "0,1"'
    ```

3. Performing the benchmarks experiments

    ```
    #Tensorflow and Pytorch - Training: 40 experiments
    python3 $experimenter run  -Vexp.framework='["pytorch", "tensorflow"]' -Vexp.phase='["training"]' -Vexp.model='["resnet50", "resnet101", "resnet200", "vgg19", "acoustic_model"]' -Vexp.gpus="[$gpus_bench]" -Vexp.dtype='["float32", "float16"]' -Pexp.log_file='"./logs/bench_${exp.framework}_${exp.model}_${exp.phase}_${exp.num_gpus}_${exp.dtype}.log"' -Ppytorch.docker_image='"nvcr.io/nvidia/pytorch:22.01-py3"' -Ptensorflow.docker_image='"nvcr.io/nvidia/tensorflow:22.01-tf1-py3"'

    # Tensorflow and Pytorch - Inference: 20 experiments
    python3 $experimenter run  -Vexp.framework='["pytorch", "tensorflow"]' -Vexp.phase='["inference"]' -Vexp.model='["resnet50", "resnet101", "resnet200", "vgg19", "acoustic_model"]' -Pexp.gpus='["0"]' -Vexp.dtype='["float32", "float16"]' -Pexp.log_file='"./logs/bench_${exp.framework}_${exp.model}_${exp.phase}_${exp.num_gpus}_${exp.dtype}.log"' -Ppytorch.docker_image='"nvcr.io/nvidia/pytorch:22.01-py3"' -Ptensorflow.docker_image='"nvcr.io/nvidia/tensorflow:22.01-tf1-py3"'

    # TensorRT -- Inference only: 15 experiments
    python3 $experimenter run -Pexp.framework='"tensorrt"' -Pexp.phase='"inference"' -Vexp.model='["resnet50", "resnet101", "resnet200", "vgg19", "acoustic_model"]' -Vexp.gpus='["0"]' -Vexp.dtype='["float32", "float16", "int8"]' -Pexp.log_file='"./logs/bench_${exp.framework}_${exp.model}_${exp.phase}_${exp.num_gpus}_${exp.dtype}.log"' -Ptensorrt.docker_image='"trtbench:22.01"'
    ``` 

4. Report of Experiments

    ```
    python3 python/dlbs/bench_data.py --report strong report logs/*.log
    python3 python/dlbs/bench_data.py --report weak   report logs/*.log
    ```

**Note:** 
- You might need to empty the directory containing the logs before running the same benchmark (to prevent overwriting previous experiments).
- See https://hewlettpackard.github.io/dlcookbook-dlbs/#/ for detailed documentation.