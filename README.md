# Vector Addition 2 Compare CPU and GPU

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
