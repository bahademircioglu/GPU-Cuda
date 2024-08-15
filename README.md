# Vector Addition 2 Compare CPU and GPU


Vector Addition Kernel: The CUDA kernel vecAdd is defined to add three vectors (A, B, and C) element-wise and store the result in vector D. Each thread processes one element.

Random Initialization: The randomInit function initializes the vectors with random values between 0 and 99.

CPU Vector Addition: The vecAddHost function performs the same vector addition operation on the CPU.

Speed Comparison: The printSpeedComparison function compares the speed of the GPU and CPU implementations for different data sizes.

Main Function:

The program tests vector addition for different data sizes (1024, 4096, 16384, and 65536 elements).
For each size, memory is allocated on the host (CPU) and the device (GPU).
The vectors are initialized on the host and then copied to the device memory.
The GPU kernel is executed, and the time taken is recorded.
The result is copied back to the host memory and validated.
The same operation is performed on the CPU, and the time is recorded.
The program prints the time taken by both the GPU and CPU and calculates the speed-up.
