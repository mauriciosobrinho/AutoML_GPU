/*
 * This is a CUDA code that performs an iterative reverse edge 
 * detection algorithm.
 *
 * Training material developed by James Perry and Alan Gray
 * Copyright EPCC, The University of Edinburgh, 2013 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <sys/types.h>
#include <sys/time.h>

#include "reconstruct.h"

/* Data buffer to read edge data into */
float edge[N][N];

/* Data buffer for the resulting image */
float img[N][N];

/* Work buffers, with halos */
float host_input[N+2][N+2];
float gpu_output[N+2][N+2];
float host_output[N+2][N+2];


int main(int argc, char *argv[])
{
  int x, y;
  int i;
  int errors;

  double start_time_inc_data, end_time_inc_data;
  double cpu_start_time, cpu_end_time;

  float *d_input, *d_output, *d_edge;

  size_t memSize = (N+2) * (N+2) * sizeof(float);

  printf("Image size: %dx%d\n", N, N);
  printf("ITERATIONS: %d\n", ITERATIONS);
  printf("THREADSPERBLOCK: %d\n", THREADSPERBLOCK);

if ( N%THREADSPERBLOCK != 0 ){
    printf("Error: THREADSPERBLOCK must exactly divide N\n");
    exit(1);
   }

  /* allocate memory on device */
  cudaMalloc(&d_input, memSize);
  cudaMalloc(&d_output, memSize);
  cudaMalloc(&d_edge, memSize);

  /* read in edge data */
  datread("edge2048x2048.dat", (void *)edge, N, N);

  /* zero buffer so that halo is zeroed */
  for (y = 0; y < N+2; y++) {
    for (x = 0; x < N+2; x++) {
      host_input[y][x] = 0.0;
    }
  }

  /* copy input to buffer with halo */
  for (y = 0; y < N; y++) {
    for (x = 0; x < N; x++) {
       host_input[y+1][x+1] = edge[y][x];
    }
  }

  /*
   * copy to all the GPU arrays. d_output doesn't need to have this data but
   * this will zero its halo
   */
  start_time_inc_data = get_current_time();
  cudaMemcpy( d_input, host_input, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy( d_output, host_input, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy( d_edge, host_input, memSize, cudaMemcpyHostToDevice);

  /* run on GPU */
  for (i = 0; i < ITERATIONS; i++) {
    /* run the kernel */

    dim3 blocksPerGrid(N/THREADSPERBLOCK,1,1);
    dim3 threadsPerBlock(THREADSPERBLOCK,1,1);
    inverseEdgeDetect<<< blocksPerGrid, threadsPerBlock >>>(d_output, d_input, d_edge);
 
    cudaThreadSynchronize();

    /* copy the data back from the output buffer on the device */
    cudaMemcpy(gpu_output, d_output, memSize, cudaMemcpyDeviceToHost);

    /* copy the new data to the input buffer on the device */
    cudaMemcpy( d_input, gpu_output, memSize, cudaMemcpyHostToDevice);

  }


  end_time_inc_data = get_current_time();

  checkCUDAError("Main loop");

  /*
   * run on host for comparison
   */
  cpu_start_time = get_current_time();
  for (i = 0; i < ITERATIONS; i++) {

    /* perform stencil operation */
    for (y = 0; y < N; y++) {
      for (x = 0; x < N; x++) {
	host_output[y+1][x+1] = (host_input[y+1][x] + host_input[y+1][x+2] +
				 host_input[y][x+1] + host_input[y+2][x+1] \
				 - edge[y][x]) * 0.25;
      }
    }
    
    /* copy output back to input buffer */
    for (y = 0; y < N; y++) {
      for (x = 0; x < N; x++) {
	host_input[y+1][x+1] = host_output[y+1][x+1];
      }
    }
  }
  cpu_end_time = get_current_time();

/* Maximum difference allowed between host result and GPU result */
#define MAX_DIFF 0.01

  /* check that GPU result matches host result */
  errors = 0;
  for (y = 0; y < N; y++) {
    for (x = 0; x < N; x++) {
      float diff = fabs(gpu_output[y+1][x+1] - host_output[y+1][x+1]);
      if (diff >= MAX_DIFF) {
        errors++;
        //printf("Error at %d,%d (CPU=%f, GPU=%f)\n", x, y,	\
  	  //     host_output[y+1][x+1],				\
		   //	      gpu_output[y+1][x+1]);
      }
    }
  }

  if (errors == 0) 
    printf("\n\n ***TEST PASSED SUCCESSFULLY*** \n\n\n");
  else
    printf("\n\n ***ERROR: TEST FAILED*** \n\n\n");

  /* copy result to output buffer */
  for (y = 0; y < N; y++) {
    for (x = 0; x < N; x++) {
      img[y][x] = gpu_output[y+1][x+1];
    }
  }

  /* write PGM */
  pgmwrite("output.pgm", (void *)img, N, N);
  system("convert output.pgm'[300x300]' output300.jpg");
  system("convert output.pgm'[1000x1000]' output1000.jpg");

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_edge);

  printf("GPU Time (Including Data Transfer): %fs\n", \
	 end_time_inc_data - start_time_inc_data);
  printf("CPU Time                          : %fs\n", \
	 cpu_end_time - cpu_start_time);

  return 0;
}

