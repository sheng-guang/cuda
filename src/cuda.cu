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
unsigned long long* d_global_pixel_sum_origin;
unsigned long long* d_global_pixel_sum_result;


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

void cuda_begin(const Image *in) {
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
    channels= input_image.channels;
    wide = input_image.width;
    memcpy(input_image.data, in->data, w_h_c_sizeof_c);

    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, w_h_c_sizeof_c));
    CUDA_CALL(cudaMemcpy(d_input_image_data, in->data, w_h_c_sizeof_c, cudaMemcpyHostToDevice));

    // Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, w_h_c_sizeof_c));

    //Allocate and zero buffer for calculation global pixel average
    CUDA_CALL(cudaMalloc(&d_global_pixel_sum_origin, tx_ty_c * sizeof(unsigned long long)));
    CUDA_CALL(cudaMalloc(&d_global_pixel_sum_result, tx_ty_c * sizeof(unsigned long long)));

#ifdef VALIDATION
    output_image = *in;
    output_image.data = (unsigned char*)malloc(in->width * in->height * in->channels * sizeof(unsigned char));
#endif

}
int cfg1(int total,int cfg2) {
    int re = total / cfg2;
    if (total % cfg2 != 0)re++;
    return re;
}

//1-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

__device__
void sum_4(int count, unsigned long long* from, unsigned long long* to, int _channel) {

}
__device__
void sum() {

}
__global__
void sum(int n, unsigned char* d_input_image_data, unsigned long long* d_sums
    ,int tile_x_count,int channels,int wide) 
{
    int t_x = blockIdx.x;
    int t_y = blockIdx.y;
    int ch = blockIdx.z;

    int p_x = threadIdx.x;
    int p_y = threadIdx.y;
    //printf("%d,%d,%d,%d,\n", t_x, t_y, p_x, p_y);
    const unsigned int tile_offset = (t_y * tile_x_count * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * channels;
    const unsigned int pixel_offset = (p_y * wide + p_x) * channels;
    int data_index = tile_offset + pixel_offset;


    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockSize = TILE_PIXELS;
    __shared__ unsigned long long sdata[TILE_PIXELS];
    sdata[tid] = d_input_image_data[data_index + ch];
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = 1; s < blockSize; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    int T_Index = (t_y * tile_x_count + t_x) * channels;
    // write result for this block to global mem
    if (tid == 0) d_sums[T_Index+ch] = sdata[0];

}


//0.271ms
//4096:6.693ms
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
    blocks.z = channels;

    dim3 threads;
    threads.x = TILE_SIZE;
    threads.y = TILE_SIZE;
    threads.z = 1;
    sum<<<blocks, threads >>>(tx_ty_c, d_input_image_data, d_sums
        , tile_x_count, channels,wide);

#ifdef VALIDATION
    cudaDeviceSynchronize();
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    CUDA_CALL(cudaMemcpy(sums, d_sums, tx_ty_c * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
     validate_tile_sum(&input_image, sums);
#endif
}

//2-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//void stage2_sum(unsigned char* mosaic, int  mosaic_count, int channels, unsigned long long* whole_sum)
//{
//    for (size_t i = 0; i < mosaic_count; i++) { int ch = i % channels;        whole_sum[ch] += mosaic[i]; }
//}

__global__
void average(int n, unsigned long long* d_sums, unsigned long long* d_global_pixel_sum, unsigned char* d_mosaic_value,int count)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n)return;
    d_mosaic_value[index] = d_sums[index] / count;
    d_global_pixel_sum[index] = d_mosaic_value[index];
}


//__global__
//void sun_4(int count,unsigned long long* arr, unsigned long long* sum,int _channel)
//{
////#define _channel 3
//    unsigned int thread_index=blockDim.x*blockIdx.x+threadIdx.x;
//    int to = thread_index * _channel;
//    int tid = thread_index *4* _channel;
//    long long re[4];
//
//    //printf("from to  %d  %d  \n", tid, tid + _channel * 3 + 2);
//    
//     for (int i = 0; i < _channel; i++)
//     {
//         re[i] = arr[i+tid] + arr[i+tid + _channel] + arr[i+tid + _channel * 2] + arr[i+tid + _channel * 3];
//         //__syncthreads();
//         //sum[to + i] = re[i];
//     }
//    __syncthreads();
//    for (int i = 0; i < _channel; i++)
//    {
//        sum[to + i] = re[i];
//    }
//
//    //printf("set %d %d %d\n", to, to + 1, to + 2);
//}

__global__
void sun_4_v2(int count, unsigned long long* arr, unsigned long long* sum, int _channel)
{
    //#define _channel 3
    unsigned int thread_index = blockDim.x * blockIdx.x + threadIdx.x;
    int to = thread_index * _channel;
    int tid = thread_index  * _channel*4;

    for (int i = 0; i < _channel; i++)
    {
        sum[to + i] = arr[i + tid] + arr[i + tid + _channel] + arr[i + tid + _channel * 2] + arr[i + tid + _channel * 3];
    }


}

//0.167ms
void cuda_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_compact_mosaic(tile_x_count, tile_y_count, sums, cpu_mosaic_value, output_global_average);
    {
        int c1 = cfg1(tx_ty_c, 32);
        average << <c1, 32 >> > (tx_ty_c, d_sums, d_global_pixel_sum_result, d_mosaic_value, TILE_PIXELS);
    }
    //{
    //    int count = tx_ty;
    //    //printf("total= %d\n" , tx_ty_c);
    //    while (count>=4)
    //    {
    //        count >>= 2;
    //        //printf("count= %d\n", count);
    //        int c1 = cfg1(count, 32);
    //        sun_4 << <c1, 32 >> > (count,d_global_pixel_sum, d_global_pixel_sum,channels);
    //    }
    //}
    {
        //todo count
        int count = tx_ty;
        while (count >= 4)
        {
            //count/=4;
            count >>= 2;
            int c1 = cfg1(count, 32);
            //exchange
            unsigned long long* temp = d_global_pixel_sum_origin;
            d_global_pixel_sum_origin = d_global_pixel_sum_result;
            d_global_pixel_sum_result = temp;
            //run
            sun_4_v2 << <c1, 32 >> > (count, d_global_pixel_sum_origin, d_global_pixel_sum_result, channels);
        }
    }


    cudaDeviceSynchronize();

    unsigned long long arr4[] = { 0,0,0,0 };
    CUDA_CALL(cudaMemcpy(arr4, d_global_pixel_sum_result, channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));


    for (size_t i = 0; i < channels; i++)
    {
        output_global_average[i] = arr4[i] / (tile_x_count * tile_y_count);
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


__global__
void broadcast(int w_h, unsigned char* d_output_image_data, unsigned char* mosaic_value
    , const int count_in_tile_line, const int count_in_img_line, const int t_size
    , const int tile_x_count,const int channels)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= w_h)return;
    int t_y = index / count_in_tile_line;
    int left = index % count_in_img_line;
    int t_x = left / t_size;

    int index1 = index * channels;
    int index2 = (t_y * tile_x_count + t_x) * channels;

    for (size_t i = 0; i < channels; i++)
    {
        d_output_image_data[index1 +i] = mosaic_value[index2 + i];
    }
}

//__global__
//void broadcast2(int mosaic_count, unsigned char* d_output_image_data, unsigned char* d_mosaic_value
//    , int tile_x_count, int channels, int wide)
//{
//
//    unsigned int mosaic_index = blockDim.x * blockIdx.x + threadIdx.x;
//    if (mosaic_index >= mosaic_count)return;
//
//    int t_y = mosaic_index / (tile_x_count * channels);
//    int t_x = (mosaic_index - t_y * (tile_x_count * channels)) / channels;
//    int ch = mosaic_index - t_y * (tile_x_count * channels) - t_x * channels;
//    const unsigned int tile_offset = (t_y * tile_x_count * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * channels;
//
//    int p_y;
//    for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
//        int p_x;
//        for (p_x = 0; p_x < TILE_SIZE; ++p_x) {
//            // For each colour channel
//            const unsigned int pixel_offset = (p_y * wide + p_x) * channels;
//            // Load pixel
//            d_output_image_data[tile_offset + pixel_offset + ch] = d_mosaic_value[mosaic_index];
//        }
//    }
//}

//1.104ms
void cuda_stage3() {

    int c1 = cfg1(w_h, 32);
    broadcast <<<c1, 32 >> > (w_h, d_output_image_data, d_mosaic_value
        , tile_x_count * TILE_PIXELS, wide, TILE_SIZE
        ,tile_x_count,channels);
    //broadcast2 << <c1, 32 >> > (tx_ty_c, d_output_image_data, d_mosaic_value
    //    , tile_x_count, channels, wide);
#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    cudaDeviceSynchronize();
    CUDA_CALL(cudaMemcpy(output_image.data,d_output_image_data , w_h_c_sizeof_c, cudaMemcpyDeviceToHost));
    validate_broadcast(&input_image, cpu_mosaic_value, &output_image);
#endif    
}
void cuda_end(Image *out) {
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
    CUDA_CALL(cudaFree(d_global_pixel_sum_origin));
    CUDA_CALL(cudaFree(d_global_pixel_sum_result));

#ifdef VALIDATION
    free(sums);
    free(cpu_mosaic_value);
    free(output_image.data);
#endif    

}
