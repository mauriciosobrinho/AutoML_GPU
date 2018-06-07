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

/* The actual CUDA kernel that runs on the GPU - 1D version by column */
__global__ void inverseEdgeDetect(float *d_output, float *d_input, \
					float *d_edge)
{
  int col, row;
  int idx, idx_south, idx_north, idx_west, idx_east;
  int numcols = N + 2;

  /*
   * calculate global row index for this thread  
   * from blockIdx.x, blockDim.x and threadIdx.x
   * remember to add 1 to account for halo    
   */


  row = blockIdx.x*blockDim.x + threadIdx.x + 1;
  /*
   * loop over all columns of the image
   */
  for (col = 1; col <= N; col++) {
      /*
       * calculate linear index from col and row, for the centre
       * and neighbouring points needed below.
       * For the neighbouring points you need to add/subtract 1  
       * to/from the row or col indices.
       */
      
      idx = row * numcols + col;
      idx_south = (row - 1) * numcols + col;
      idx_north = (row + 1) * numcols + col;
      
      idx_west = row * numcols + (col - 1);
      idx_east = row * numcols + (col + 1);
      

      /* perform stencil operation */  
      d_output[idx] = (d_input[idx_south] + d_input[idx_west] \
		       + d_input[idx_north] + d_input[idx_east] - \
		       d_edge[idx]) * 0.25;

    }
}




    




