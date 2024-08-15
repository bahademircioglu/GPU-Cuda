# Vector Addition

CUDA Kernel Implementation:

A CUDA kernel named convolutionalLayerKernelWithSharedMemory is defined. This kernel performs a 2D convolution operation using block-shared memory to store portions of the input image that are relevant to the current block of threads.
Threads within a block collaborate to load data into shared memory, apply the filter (kernel) to the corresponding region of the input, and then store the result in the output array.

GPU Convolution (CUDA):

The function forward_pass handles the convolution operation on the GPU. It allocates memory on the GPU, transfers data from the host (CPU) to the device (GPU), and then launches the CUDA kernel.
The result is then copied back from the GPU to the host.

CPU Convolution:

The function convolution_cpu performs the same 2D convolution operation but entirely on the CPU. It uses nested loops to apply the filter to each possible position in the input matrix.

Performance Measurement:

The measure_performance function times how long the convolution operation takes on both the CPU and GPU.
It calls both the CPU and GPU implementations and records the time taken by each.

Testing with Different Sizes:

The code runs the performance measurement for several different input and filter sizes. These sizes are predefined in the sizes and filter_sizes lists.
For each combination of input size and filter size, the code prints the time taken by the CPU and GPU to perform the convolution.

Vector Addition on GPU:

The program adds three vectors (A, B, C) on the GPU using CUDA, and stores the result in a fourth vector (D).
A CUDA kernel function vecAdd is defined, which performs the element-wise addition of the three vectors in parallel on the GPU.
Host (CPU) Operation:

The same three vectors (A, B, C) are added on the CPU, and the result is stored in another vector (D_host). This is done to compare the performance of CPU and GPU.

Memory Management:

The program allocates memory on both the host (CPU) and the device (GPU) for the vectors. After the operations are completed, the allocated memory is freed.

Performance Measurement:

The program measures the time taken for vector addition on both the GPU and CPU.
It calculates the speed-up achieved by using the GPU compared to the CPU.

Comparison:

The experiment is repeated for different vector sizes (N = 2^10, N = 2^12, N = 2^14).
The performance results, including the execution time for each vector size and the speed-up, are printed to the console.

# Histogram Computation

Error Checking and Time Measurement:

A utility function checkCUDAError is provided to check for CUDA errors.
Another function, seconds, is defined to measure elapsed time during computations.

GPU-based Histogram Computation:

The kernel histogramKernel computes a histogram on the GPU.
Shared memory (__shared__ int temp[NUM_BINS];) is used to temporarily store histogram data within each block to reduce global memory accesses.
Threads within each block update the shared memory histogram using atomic operations (atomicAdd).
The results from shared memory are then atomically added to the global memory histogram.

CPU-based Histogram Computation:

The function histogramCPU computes the histogram on the CPU by iterating through the input array and incrementing the corresponding bins.
Comparing GPU and CPU Computation Times:

The function runHistogram is responsible for generating random input data, invoking both the GPU and CPU histogram computation functions, and comparing their performance.
It also prints the first 10 histogram bins calculated by both the GPU and CPU for verification.

Main Function:

The main function runs the runHistogram function for different input sizes (256KB, 512KB, 1MB, and 2MB).


# Convolutional Operation

CUDA Kernel Implementation:

A CUDA kernel named convolutionalLayerKernelWithSharedMemory is defined. This kernel performs a 2D convolution operation using block-shared memory to store portions of the input image that are relevant to the current block of threads.
Threads within a block collaborate to load data into shared memory, apply the filter (kernel) to the corresponding region of the input, and then store the result in the output array.
GPU Convolution (CUDA):

The function forward_pass handles the convolution operation on the GPU. It allocates memory on the GPU, transfers data from the host (CPU) to the device (GPU), and then launches the CUDA kernel.
The result is then copied back from the GPU to the host.

CPU Convolution:

The function convolution_cpu performs the same 2D convolution operation but entirely on the CPU. It uses nested loops to apply the filter to each possible position in the input matrix.

Performance Measurement:

The measure_performance function times how long the convolution operation takes on both the CPU and GPU.
It calls both the CPU and GPU implementations and records the time taken by each.

Testing with Different Sizes:

The code runs the performance measurement for several different input and filter sizes. These sizes are predefined in the sizes and filter_sizes lists.
For each combination of input size and filter size, the code prints the time taken by the CPU and GPU to perform the convolution.
