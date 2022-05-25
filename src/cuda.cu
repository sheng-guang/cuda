#include "cuda.cuh"

#include <cstring>
#include <device_launch_parameters.h>
#include "helper.h"

///
/// Algorithm storage
///
// Host copy of input image
Image input_image;
// Host copy of image tiles in each dimension
unsigned int tile_x_count, tile_y_count;
// Pointer to device buffer for calculating the sum of each tile mosaic, this must be passed to a kernel to be used on device
unsigned long long* d_sums;

// Pointer to device buffer for storing the output pixels of each tile, this must be passed to a kernel to be used on device
unsigned char* d_mosaic_value;
// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
//unsigned long long* d_global_pixel_sum_origin;
unsigned long long* d_global_sum;


// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;

size_t w_h_c_sizeof_c;
size_t w_h_c;
size_t w_h;
size_t tx_ty_c;
size_t tx_ty;
int channels;
int wide;



//VALIDATION-------------------------------------------------------------------------------------------------------------------
#ifdef VALIDATION
unsigned long long* sums;
unsigned char* cpu_mosaic_value;
Image output_image;
#endif

void cuda_begin(const Image* in) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

    tile_x_count = in->width / TILE_SIZE;
    tile_y_count = in->height / TILE_SIZE;
    tx_ty_c = tile_x_count * tile_y_count * in->channels;
    tx_ty = tile_x_count * tile_y_count;
    // Allocate buffer for calculating the sum of each tile mosaic
    CUDA_CALL(cudaMalloc(&d_sums, tx_ty_c * sizeof(unsigned long long)));
#ifdef VALIDATION
    sums = (unsigned long long*)malloc(tx_ty_c * sizeof(unsigned long long));
#endif

    // Allocate buffer for storing the output pixel value of each tile
    CUDA_CALL(cudaMalloc(&d_mosaic_value, tx_ty_c * sizeof(unsigned char)));
#ifdef VALIDATION
    cpu_mosaic_value = (unsigned char*)malloc(tx_ty_c * sizeof(unsigned char));
#endif


    w_h_c_sizeof_c = in->width * in->height * in->channels * sizeof(unsigned char);
    w_h_c = in->width * in->height * in->channels;
    w_h = in->width * in->height;

    // Allocate copy of input image
    input_image = *in;
    input_image.data = (unsigned char*)malloc(w_h_c_sizeof_c);
    channels = input_image.channels;
    wide = input_image.width;
    memcpy(input_image.data, in->data, w_h_c_sizeof_c);

    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, w_h_c_sizeof_c));
    CUDA_CALL(cudaMemcpy(d_input_image_data, in->data, w_h_c_sizeof_c, cudaMemcpyHostToDevice));

    // Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, w_h_c_sizeof_c));

    //Allocate and zero buffer for calculation global pixel average
    //CUDA_CALL(cudaMalloc(&d_global_pixel_sum_origin, tx_ty_c * sizeof(unsigned long long)));
    CUDA_CALL(cudaMalloc(&d_global_sum, tx_ty_c * sizeof(unsigned long long)));



#ifdef VALIDATION
    output_image = *in;
    output_image.data = (unsigned char*)malloc(in->width * in->height * in->channels * sizeof(unsigned char));
#endif

}



int cfg1(int total, int cfg2) {
    int re = total / cfg2;
    if (total % cfg2 != 0)re++;
    return re;
}






//1-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template <unsigned int blockSize>
__device__ void last32(volatile unsigned long long* s_data, unsigned int tid) {
    //if (blockSize >= 512) s_data[tid] += s_data[tid + 256];
    //if (blockSize >= 256)  s_data[tid] += s_data[tid + 128];
    //if (blockSize >= 128)  s_data[tid] += s_data[tid + 64];

    if (blockSize >= 64) s_data[tid] += s_data[tid + 32];
    if (blockSize >= 32) s_data[tid] += s_data[tid + 16];
    if (blockSize >= 16) s_data[tid] += s_data[tid + 8];
    if (blockSize >= 8) s_data[tid] += s_data[tid + 4];
    if (blockSize >= 4) s_data[tid] += s_data[tid + 2];
    if (blockSize >= 2) s_data[tid] += s_data[tid + 1];
}

template <unsigned int blockSize>
__global__
void sum(unsigned char* d_input_image_data, unsigned long long* d_sums
    , int tile_x_count, int channels, int wide)
{
    int t_x = blockIdx.x;
    int t_y = blockIdx.y;
    unsigned int tile_offset = (t_y * tile_x_count * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * channels;


    int p_x = threadIdx.x;
    int p_y = threadIdx.y;
    //printf("%d,%d,%d,%d,\n", t_x, t_y, p_x, p_y);
    unsigned int pixel_offset = (p_y * wide + p_x) * channels;
    int data_index = tile_offset + pixel_offset;

    int offset_x = TILE_SIZE / 2 * channels;
    int offset_y = TILE_SIZE / 2 * wide * channels;

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    //int blockSize = TILE_PIXELS/4;


    for (int ch = 0; ch < channels; ch++)
    {
        extern __shared__ unsigned long long s_data[blockSize];
        s_data[tid] = d_input_image_data[data_index + ch] +
            d_input_image_data[data_index + offset_x + ch] + d_input_image_data[data_index + offset_y + ch] +
            d_input_image_data[data_index + offset_x + offset_y + ch];
        __syncthreads();


        if (blockSize >= 512) { if (tid < 256) { s_data[tid] += s_data[tid + 256]; } __syncthreads(); }
        if (blockSize >= 256) { if (tid < 128) { s_data[tid] += s_data[tid + 128]; } __syncthreads(); }
        if (blockSize >= 128) { if (tid < 64) { s_data[tid] += s_data[tid + 64]; } __syncthreads(); }

        //if (blockSize >= 64) { if (tid < 32) { s_data[tid] += s_data[tid + 32]; } __syncthreads(); }
        //if (blockSize >= 32) { if (tid < 16) { s_data[tid] += s_data[tid + 16]; } __syncthreads(); }
        //if (blockSize >= 16) { if (tid < 8) { s_data[tid] += s_data[tid + 8]; } __syncthreads(); }
        //if (blockSize >= 8) { if (tid < 4) { s_data[tid] += s_data[tid + 4]; } __syncthreads(); }
        //if (blockSize >= 4) { if (tid < 2) { s_data[tid] += s_data[tid + 2]; } __syncthreads(); }
        //if (blockSize >= 2) { if (tid < 1) { s_data[tid] += s_data[tid + 1]; } __syncthreads(); }
        if (tid < 32) last32<blockSize>(s_data, tid);

        int T_Index = (t_y * tile_x_count + t_x) * channels;
        // write result for this block to global mem
        if (tid == 0) d_sums[T_Index + ch] = s_data[0];
    }
}



void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_tile_sum(&input_image, sums);
    //printf("sums count: %d\n", tx_ty_c);
    //printf("tile count: %d\n", tx_ty_c/channels);
    //printf("channels count: %d\n",channels);

    //int c1 = cfg1(tx_ty_c, 32);
    dim3 blocks;
    blocks.x = tile_x_count;
    blocks.y = tile_y_count;
    blocks.z = 1;

    dim3 threads;
    threads.x = TILE_SIZE / 2;
    threads.y = TILE_SIZE / 2;
    threads.z = 1;

    sum<TILE_PIXELS / 4> << <blocks, threads >> > (d_input_image_data, d_sums
        , tile_x_count, channels, wide);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    cudaDeviceSynchronize();
    CUDA_CALL(cudaMemcpy(sums, d_sums, tx_ty_c * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    validate_tile_sum(&input_image, sums);
#endif
}

//2-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





//__global__
//void average2(int n, unsigned long long* d_sums, unsigned long long* d_global_pixel_sum, unsigned char* d_mosaic_value, int count)
//{
//    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
//    if (index >= n)return;
//    d_mosaic_value[index] = d_sums[index] / count;
//    d_global_pixel_sum[index] = d_mosaic_value[index];
//}
//__global__
//void sun_4_v2(int count, unsigned long long* arr, unsigned long long* sum, int _channel)
//{
//    //#define _channel 3
//    unsigned int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
//    int to = thread_index * _channel;
//    int tid = thread_index * _channel * 4;
//
//    for (int i = 0; i < _channel; i++)
//    {
//        sum[to + i] = arr[i + tid] + arr[i + tid + _channel] + arr[i + tid + _channel * 2] + arr[i + tid + _channel * 3];
//    }
//
//
//}
//
    //gridDim{
    // blockIdx.x;
    // }


    //blockDim
    //threadIdx.x;


template<unsigned int blockSize>
__global__
void sum_4_v3(int count,unsigned long long* d_sums, unsigned char* d_mosaic_value, unsigned long long* d_global_sum
    ,int channels ,int foraverage,int tx_ty)
{
    extern __shared__ unsigned long long s[TILE_PIXELS ];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockSize * 4 + tid;
    for (int ch = 0; ch < channels; ch++)
    {
        int start_Index = (blockIdx.x * blockSize*4+ tid);
        int start_Index_4 =(blockIdx.x * blockSize * 4 + tid)*4;
        int i0 = start_Index_4 * channels + ch;
        int i1 = (start_Index_4 + 1) * channels + ch;
        int i2 = (start_Index_4 + 2) * channels + ch;
        int i3 = (start_Index_4 + 3) * channels + ch;
        d_mosaic_value[i0] = d_sums[i0] / foraverage;
        d_mosaic_value[i1] = d_sums[i1] / foraverage;
        d_mosaic_value[i2] = d_sums[i2] / foraverage;
        d_mosaic_value[i3] = d_sums[i3] / foraverage;
        //printf("%d<-  %d,%d,%d,%d\n", tid, i0, i1, i2, i3);
        s[tid] = d_mosaic_value[i0] + d_mosaic_value[i1] + d_mosaic_value[i2] + d_mosaic_value[i3];
        s[tid] = s[tid] ;
        //printf("to index=%d    tid=%d \n", tid * channels + ch,tid);
        //d_mosaic_value[tid*channels+ch] = d_sums[tid*channels+ch] / foraverage;
        //s[tid] = d_mosaic_value[tid * channels + ch];
        __syncthreads();
        if (blockSize >= 256) { if (tid < 64) { s[tid] += s[tid + 64] + s[tid + 64 * 2] + s[tid + 64 * 3]; } __syncthreads(); }
        if (blockSize >= 64) { if (tid < 16) {
            s[tid] += s[tid + 16] + s[tid + 16 * 2] + s[tid + 16 * 3]; } __syncthreads(); }
        if (blockSize >= 16) { if (tid < 4) {
            s[tid] += s[tid + 4] + s[tid + 4 * 2] + s[tid + 4 * 3]; } __syncthreads(); }
        if (blockSize >= 4) { if (tid < 1) 
        {s[tid] += s[tid + 1] + s[tid + 1 * 2] + s[tid + 1 * 3]; } __syncthreads(); }


//        if (blockSize >= 64) { if (tid < 32) { s[tid] += s[tid + 32]; } __syncthreads(); }
//if (blockSize >= 32) { if (tid < 16) { s[tid] += s[tid + 16]; } __syncthreads(); }
//if (blockSize >= 16) { if (tid < 8) { s[tid] += s[tid + 8]; } __syncthreads(); }
//if (blockSize >= 8) { if (tid < 4) { s[tid] += s[tid + 4]; } __syncthreads(); }
//if (blockSize >= 4) { if (tid < 2) { s[tid] += s[tid + 2]; } __syncthreads(); }
//if (blockSize >= 2) { if (tid < 1) { s[tid] += s[tid + 1]; } __syncthreads(); }

        // write result for this block to global mem
        __syncthreads();
        if (tid == 0) 
        {
            int ountindex = blockIdx.x * channels + ch;
            //printf("out %d\n", ountindex);
            d_global_sum[ountindex] = s[0]/tx_ty; 
        }
    }

}
//0.167ms
void cuda_stage2(unsigned char* output_global_average) {
    //printf("stage2------------------------------------------count= %d   tx_ty= %d    tile_x_count= %d\n",tx_ty_c,tx_ty, tile_x_count);

    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_compact_mosaic(tile_x_count, tile_y_count, sums, cpu_mosaic_value, output_global_average);
    dim3 threads;


    if (tile_x_count ==8) {
        //printf("enter1\n");

        sum_4_v3 <16> << <1, 16 >> > (16, d_sums, d_mosaic_value, d_global_sum
            , channels, TILE_PIXELS, tx_ty);
    }
    else if(tile_x_count==32)
    {
        //printf("enter2\n");
        sum_4_v3 <256> << <1, 256 >> > (256, d_sums, d_mosaic_value, d_global_sum
            , channels, TILE_PIXELS, tx_ty);
    }
    else if(tile_x_count==64)
    {
        sum_4_v3 <1024> << <1, 1024 >> > (1024, d_sums, d_mosaic_value, d_global_sum
            , channels, TILE_PIXELS, tx_ty);
    }
    else
    {
        //printf("enter2\n");

        threads.x = TILE_SIZE / 2;
        threads.y = TILE_SIZE / 2;
        threads.z = 1;
        //32*32/4=256
        int c1 = cfg1(tx_ty, TILE_PIXELS);
        sum_4_v3 < TILE_PIXELS / 4> << <c1, threads >> > (tx_ty / 4, d_sums, d_mosaic_value, d_global_sum
            , channels, TILE_PIXELS, tx_ty);
        printf("c1=%d\n", c1);
    }



    cudaDeviceSynchronize();

    unsigned long long arr4[] = { 0,0,0,0 };
    CUDA_CALL(cudaMemcpy(arr4, d_global_sum, channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));


    for (size_t i = 0; i < channels; i++)
    {
        //arr4[i] /= tx_ty;
        output_global_average[i] = arr4[i];
    }
#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    CUDA_CALL(cudaMemcpy(cpu_mosaic_value, d_mosaic_value, tx_ty_c * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    validate_compact_mosaic(tile_x_count, tile_y_count, sums, cpu_mosaic_value, output_global_average);
#endif    
}



//3-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


//__global__
//void broadcast(int w_h, unsigned char* d_output_image_data, unsigned char* mosaic_value
//    ,  int count_in_tile_line,  int count_in_img_line,  int t_size
//    ,  int tile_x_count, int channels)
//{
//    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
//    if (index >= w_h)return;
//    int t_y = index / count_in_tile_line;
//    int left = index % count_in_img_line;
//    int t_x = left / t_size;
//
//    int index1 = index * channels;
//    int index2 = (t_y * tile_x_count + t_x) * channels;
//
//    for (size_t i = 0; i < channels; i++)
//    {
//        d_output_image_data[index1 +i] = mosaic_value[index2 + i];
//    }
//}


//__global__
//void broadcast2(int mosaic_count, unsigned char* d_output_image_data, unsigned char* d_mosaic_value
//    , int tile_x_count, int channels, int wide)
//{
//
//    unsigned int mosaic_index = blockDim.x * blockIdx.x + threadIdx.x;
//    if (mosaic_index >= mosaic_count)return;
//
//    unsigned int t_y = mosaic_index / (tile_x_count * channels);
//    unsigned int t_x = (mosaic_index - t_y * (tile_x_count * channels)) / channels;
//    unsigned int ch = mosaic_index - t_y * (tile_x_count * channels) - t_x * channels;
//    unsigned int tile_offset = (t_y * tile_x_count * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * channels;
//
//    unsigned int p_y;
//    for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
//        unsigned int p_x;
//        for (p_x = 0; p_x < TILE_SIZE; ++p_x) {
//            // For each colour channel
//            unsigned int pixel_offset = (p_y * wide + p_x) * channels;
//            // Load pixel
//            d_output_image_data[tile_offset + pixel_offset + ch] = d_mosaic_value[mosaic_index];
//        }
//    }
//}
__global__
void broadcast3(unsigned char* d_output_image_data, unsigned char* mosaic_value
    , unsigned int tile_x_count, unsigned int channels, unsigned int wide)
{

    unsigned int t_x = blockIdx.x;
    unsigned int t_y = blockIdx.y;
    unsigned int tile_index = (t_y * tile_x_count + t_x) * channels;
    unsigned int tile_offset = (t_y * tile_x_count * TILE_PIXELS + t_x * TILE_SIZE) * channels;


    unsigned int p_x = threadIdx.x;
    unsigned int p_y = threadIdx.y;
    unsigned int pixel_offset = (p_y * wide + p_x) * channels;
    unsigned int data_index = tile_offset + pixel_offset;



    int offset_x = TILE_SIZE / 2 * channels;
    int offset_y = TILE_SIZE / 2 * wide * channels;
    for (int ch = 0; ch < channels; ch++)
    {
        d_output_image_data[data_index + ch] = mosaic_value[tile_index + ch];
        d_output_image_data[data_index + ch + offset_x] = mosaic_value[tile_index + ch];
        d_output_image_data[data_index + ch + offset_y] = mosaic_value[tile_index + ch];
        d_output_image_data[data_index + ch + offset_x + offset_y] = mosaic_value[tile_index + ch];
    }
}

void cuda_stage3() {
    //{
    //    int c1 = cfg1(w_h, 32);
    //    broadcast << <c1, 32 >> > (w_h, d_output_image_data, d_mosaic_value
    //        , tile_x_count * TILE_PIXELS, wide, TILE_SIZE
    //        , tile_x_count, channels);
    //}

    //{
    //    broadcast2 << <c1, 32 >> > (tx_ty_c, d_output_image_data, d_mosaic_value
    //        , tile_x_count, channels, wide);
    //}

    {
        dim3 blocks_3;
        dim3 threads_3;
        blocks_3.x = tile_x_count;
        blocks_3.y = tile_y_count;
        blocks_3.z = 1;

        threads_3.x = TILE_SIZE / 2;
        threads_3.y = TILE_SIZE / 2;
        threads_3.z = 1;
        broadcast3 << <blocks_3, threads_3 >> > (d_output_image_data, d_mosaic_value,
            tile_x_count, channels, wide);
    }
#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    cudaDeviceSynchronize();
    CUDA_CALL(cudaMemcpy(output_image.data, d_output_image_data, w_h_c_sizeof_c, cudaMemcpyDeviceToHost));
    validate_broadcast(&input_image, cpu_mosaic_value, &output_image);
#endif    
}
void cuda_end(Image* out) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    out->width = input_image.width;
    out->height = input_image.height;
    out->channels = input_image.channels;
    CUDA_CALL(cudaMemcpy(out->data, d_output_image_data, out->width * out->height * out->channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    free(input_image.data);
    CUDA_CALL(cudaFree(d_mosaic_value));
    CUDA_CALL(cudaFree(d_sums));
    CUDA_CALL(cudaFree(d_input_image_data));
    CUDA_CALL(cudaFree(d_output_image_data));
	//CUDA_CALL(cudaFree(d_global_pixel_sum_origin));
    CUDA_CALL(cudaFree(d_global_sum));

#ifdef VALIDATION
    free(sums);
    free(cpu_mosaic_value);
    free(output_image.data);
#endif    

}
