
__global__ void average_precompiled_Overlap1DKernel_average_kernel_1d_MapOverlapKernel_CU(
	unsigned char* input, unsigned long elemPerPx,  unsigned char* output,
	unsigned char* wrap, size_t n,
	size_t out_offset, size_t out_numelements,
	int poly, unsigned char pad, size_t overlap
)
{
   extern __shared__ unsigned char sdata[];
   // extern __shared__ char _sdata[];
   // unsigned char* sdata = reinterpret_cast<unsigned char*>(_sdata);

   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;

   while(i<(n+overlap-1))
   {
      //Copy data to shared memory
      if(poly == 0) // constant policy
      {
         sdata[overlap+tid] = (i < n) ? input[i] : pad;

         if(tid < overlap)
         {
            sdata[tid] = (i<overlap) ? pad : input[i-overlap];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (i+overlap < n) ? input[i+overlap] : pad;
         }
      }
      else if(poly == 1)
      {
         if(i < n)
         {
            sdata[overlap+tid] = input[i];
         }
         else
         {
            sdata[overlap+tid] = wrap[overlap+(i-n)];
         }

         if(tid < overlap)
         {
            sdata[tid] = (i<overlap) ? wrap[tid] : input[i-overlap];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (i+overlap < n) ? input[i+overlap] : wrap[overlap+(i+overlap-n)];
         }
      }
      else if(poly == 2) // DUPLICATE
      {
         sdata[overlap+tid] = (i < n) ? input[i] : input[n-1];

         if(tid < overlap)
         {
            sdata[tid] = (i<overlap) ? input[0] : input[i-overlap];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (i+overlap < n) ? input[i+overlap] : input[n-1];
         }
      }

      __syncthreads();

      //Compute and store data
      if( (i >= out_offset) && (i < out_offset+out_numelements) )
      	 output[i-out_offset] = skepu_userfunction_skepu_skel_0conv_average_kernel_1d::CU({(int)overlap, 1, &sdata[tid + overlap]} , elemPerPx);

      i += gridSize;

      __syncthreads();
   }
}

__global__ void average_precompiled_Overlap1DKernel_average_kernel_1d_MapOverlapKernel_CU_Matrix_Row(
	unsigned char* input, unsigned long elemPerPx,  unsigned char* output,
	unsigned char* wrap, size_t n, size_t out_offset, size_t out_numelements,
	int poly, unsigned char pad, size_t overlap, size_t blocksPerRow, size_t rowWidth
)
{
   extern __shared__ unsigned char sdata[];
   // extern __shared__ char _sdata[];
   // unsigned char* sdata = reinterpret_cast<unsigned char*>(_sdata);

   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockDim.x + tid;

   size_t wrapIndex= 2 * overlap * (int)(blockIdx.x/blocksPerRow);
   size_t tmp= (blockIdx.x % blocksPerRow);
   size_t tmp2= (blockIdx.x / blocksPerRow);


   //Copy data to shared memory
   if(poly == 0)
   {
      sdata[overlap+tid] = (i < n) ? input[i] : pad;

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? pad : input[i-overlap];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (i+overlap < n) && tmp!=(blocksPerRow-1)) ? input[i+overlap] : pad;
      }
   }
   else if(poly == 1)
   {
      if(i < n)
      {
         sdata[overlap+tid] = input[i];
      }
      else if(i-n < overlap)
      {
         sdata[overlap+tid] = wrap[(overlap+(i-n))+ wrapIndex];
      }
      else
      {
         sdata[overlap+tid] = pad;
      }

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? wrap[tid+wrapIndex] : input[i-overlap];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && i+overlap < n && tmp!=(blocksPerRow-1)) ? input[i+overlap] : wrap[overlap+wrapIndex+(tid+overlap-blockDim.x)];
      }
   }
   else if(poly == 2)
   {
      sdata[overlap+tid] = (i < n) ? input[i] : input[n-1];

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? input[tmp2*rowWidth] : input[i-overlap];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (i+overlap < n) && (tmp!=(blocksPerRow-1))) ? input[i+overlap] : input[(tmp2+1)*rowWidth-1];
      }
   }

   __syncthreads();

   //Compute and store data
   if( (i >= out_offset) && (i < out_offset+out_numelements) )
   	output[i-out_offset] = skepu_userfunction_skepu_skel_0conv_average_kernel_1d::CU({(int)overlap, 1, &sdata[tid + overlap]} , elemPerPx);
}

__global__ void average_precompiled_Overlap1DKernel_average_kernel_1d_MapOverlapKernel_CU_Matrix_Col(
	unsigned char* input, unsigned long elemPerPx,  unsigned char* output,
	unsigned char* wrap, size_t n, size_t out_offset, size_t out_numelements,
	int poly, unsigned char pad, size_t overlap, size_t blocksPerCol, size_t rowWidth, size_t colWidth
)
{
   extern __shared__ unsigned char sdata[];
   // extern __shared__ char _sdata[];
   // unsigned char* sdata = reinterpret_cast<unsigned char*>(_sdata);

   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockDim.x + tid;

   size_t wrapIndex= 2 * overlap * (int)(blockIdx.x/blocksPerCol);
   size_t tmp= (blockIdx.x % blocksPerCol);
   size_t tmp2= (blockIdx.x / blocksPerCol);

   size_t arrInd = (threadIdx.x + tmp*blockDim.x)*rowWidth + ((blockIdx.x)/blocksPerCol);

   //Copy data to shared memory
   if(poly == 0)
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd] : pad;

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? pad : input[(arrInd-(overlap*rowWidth))];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : pad;
      }
   }
   else if(poly == 1)
   {
      if(i < n)
      {
         sdata[overlap+tid] = input[arrInd];
      }
      else if(i-n < overlap)
      {
         sdata[overlap+tid] = wrap[(overlap+(i-n))+ wrapIndex];
      }
      else
      {
         sdata[overlap+tid] = pad;
      }

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? wrap[tid+wrapIndex] : input[(arrInd-(overlap*rowWidth))];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : wrap[overlap+wrapIndex+(tid+overlap-blockDim.x)];
      }
   }
   else if(poly == 2)
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd] : input[n-1];

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? input[tmp2] : input[(arrInd-(overlap*rowWidth))];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : input[tmp2+(colWidth-1)*rowWidth];
      }
   }

   __syncthreads();

   //Compute and store data
   if( (arrInd >= out_offset) && (arrInd < out_offset+out_numelements) )
   {
      output[arrInd-out_offset] = skepu_userfunction_skepu_skel_0conv_average_kernel_1d::CU({(int)overlap, 1, &sdata[tid + overlap]} , elemPerPx);
   }
}

__global__ void average_precompiled_Overlap1DKernel_average_kernel_1d_MapOverlapKernel_CU_Matrix_ColMulti(
	unsigned char* input, unsigned long elemPerPx,  unsigned char* output,
	unsigned char* wrap, size_t n, size_t in_offset, size_t out_numelements,
	int poly, int deviceType, unsigned char pad, size_t overlap, size_t blocksPerCol, size_t rowWidth, size_t colWidth
)
{
   extern __shared__ unsigned char sdata[];
   // extern __shared__ char _sdata[];
   // unsigned char* sdata = reinterpret_cast<unsigned char*>(_sdata);

   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockDim.x + tid;

   size_t tmp= (blockIdx.x % blocksPerCol);
   size_t tmp2= (blockIdx.x / blocksPerCol);

   size_t arrInd = (threadIdx.x + tmp*blockDim.x)*rowWidth + tmp2; //((blockIdx.x)/blocksPerCol);

   if(poly == 0) //IF overlap policy is CONSTANT
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : pad; // in_offset

      if(deviceType == -1) // first device, i.e. in_offset=0
      {
         if(tid < overlap)
         {
            sdata[tid] = (tmp==0) ? pad : input[(arrInd-(overlap*rowWidth))];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 0) // middle device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 1) // last device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+in_offset+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+in_offset+(overlap*rowWidth))] : pad;
         }
      }
   }
   else if(poly == 1) //IF overlap policy is CYCLIC
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : ((i-n < overlap) ? wrap[(i-n)+ (overlap * tmp2)] : pad);

      if(deviceType == -1) // first device, i.e. in_offset=0
      {
         if(tid < overlap)
         {
            sdata[tid] = (tmp==0) ? wrap[tid+(overlap * tmp2)] : input[(arrInd-(overlap*rowWidth))];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 0) // middle device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 1) // last device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+in_offset+(overlap*rowWidth))] : wrap[(overlap * tmp2)+(tid+overlap-blockDim.x)];
         }
      }
   }
   else if(poly == 2) //IF overlap policy is DUPLICATE
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : input[n+in_offset-1];

      if(deviceType == -1) // first device, i.e. in_offset=0
      {
         if(tid < overlap)
         {
            sdata[tid] = (tmp==0) ? input[tmp2] : input[(arrInd-(overlap*rowWidth))];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 0) // middle device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 1) // last device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+in_offset+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+in_offset+(overlap*rowWidth))] : input[tmp2+in_offset+(colWidth-1)*rowWidth];
         }
      }
   }

   __syncthreads();

   //Compute and store data
   if( arrInd < out_numelements )
   {
      output[arrInd] = skepu_userfunction_skepu_skel_0conv_average_kernel_1d::CU({(int)overlap, 1, &sdata[tid + overlap]} , elemPerPx);
   }
}
