#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define CUDA_SAFE_CALL(func) { \
    cudaError_t err = (func); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "error [%d] : %s\n", err, cudaGetErrorString(err)); \
        exit(err); \
    } \
}

// __global__関数はホストから呼び出せるデバイス側関数
// 戻り値は返せない
__global__ void addKernel(int *c, const int *a, const int *b, const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        //printf(">> blockIdx: %d  threadIdx: %d\n", blockIdx.x, threadIdx.x);
        return;
    }
    c[i] = a[i] + b[i];
}

void add(int *c, const int *a, const int *b, unsigned int n)
{
    int *dev_a;
    int *dev_b;
    int *dev_c;
    cudaSetDevice(0);

    // 3つの配列領域(2入力/1出力)をGPU側に確保
    CUDA_SAFE_CALL( cudaMalloc((void **)&dev_c, sizeof(int) * n) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&dev_a, sizeof(int) * n) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&dev_b, sizeof(int) * n) );
    // 2つの入力データ(配列)をホストからデバイスへ転送
    CUDA_SAFE_CALL( cudaMemcpy(dev_a, a, n * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(dev_b, b, n * sizeof(int), cudaMemcpyHostToDevice) );

    // カーネル呼び出し
    // Grid > Block > Thread
    // func_gpu << Dg, Db [ , Ns, S ] >>> (a, b, c);
    // Dg : グリッドサイズ (グリッド内のブロック数)
    // Db : ブロックサイズ (ブロック内のスレッド数)
    // Ns : シェアードメモリのサイズ. 省略時は 0
    // S  : ストリーム番号
    int th = 1024;
    int bl = (n / 1024) + 1;
    dim3 blocks(bl, 1, 1);
    dim3 threads(th, 1, 1);
    addKernel<<<blocks, threads>>>(dev_c, dev_a, dev_b, n);

    // カーネルの終了を待つ
    cudaDeviceSynchronize();
    // 結果(配列)をデバイスからホストへ転送
    cudaMemcpy(c, dev_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    // デバイス側メモリ開放
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return;
}

#ifdef __cplusplus
extern "C" {
#endif
void cuda_kernel_exec(int n)
{
    int i;
    int *a = (int *)malloc(sizeof(int) * n);
    int *b = (int *)malloc(sizeof(int) * n);
    int *c = (int *)malloc(sizeof(int) * n);

    for (i=0; i<n; i++) {
      a[i] = i+1;
      b[i] = i-1;
      c[i] = 0;
    }
    // Add vectors in parallel.
    add(c, a, b, n);
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();
    free(a);
    free(b);
    free(c);
    return;
}
#ifdef __cplusplus
};
#endif
