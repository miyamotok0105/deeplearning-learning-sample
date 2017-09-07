
gcc -Wall -c cuda_main.c
nvcc -c cuda_kernel.cu
gcc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 cuda_main.o cuda_kernel.o -lcudart
./a.out

