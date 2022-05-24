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
//// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
//unsigned long long* d_global_pixel_sum;


// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;

size_t w_h_c_sizeof_c;
size_t w_h_c;
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

    // Allocate and zero buffer for calculation global pixel average
    //CUDA_CALL(cudaMalloc(&d_global_pixel_sum, tx_ty_c * sizeof(unsigned long long)));

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
__global__
void sum(int n, unsigned char* d_input_image_data, unsigned long long* d_sums
    ,int tile_x_count,int channels,int wide) 
{

    unsigned int sum_index= blockDim.x*blockIdx.x+ threadIdx.x;
    if (sum_index >= n)return;

    int t_y = sum_index / (tile_x_count*channels);
    int t_x = (sum_index - t_y * (tile_x_count * channels))/channels;
    int ch = sum_index - t_y * (tile_x_count * channels) - t_x * channels;
    const unsigned int tile_offset = (t_y * tile_x_count * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * channels;

    long long sum = 0;
    int p_y;
    for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
        int p_x;
        for (p_x = 0; p_x < TILE_SIZE; ++p_x) {
            // For each colour channel
            const unsigned int pixel_offset = (p_y * wide + p_x) * channels;
            // Load pixel
            const unsigned char pixel = d_input_image_data[tile_offset + pixel_offset + ch];
            sum += pixel;
        }
    }
    d_sums[sum_index] = sum;
}
//0.271ms
void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_tile_sum(&input_image, sums);
    //printf("sums count: %d\n", tx_ty_c);
    //printf("tile count: %d\n", tx_ty_c/channels);
    //printf("channels count: %d\n",channels);

    int c1 = cfg1(tx_ty_c, 32);
    sum<<<c1,32>>>(tx_ty_c, d_input_image_data, d_sums
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
void average(int n, unsigned long long* d_sums, unsigned char* d_mosaic_value,int count)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n)return;
    d_sums[index] = d_sums[index] / count;
    d_mosaic_value[index] = d_sums[index];
}

__global__
void sun_4(unsigned long long* arr, unsigned long long* sum, int channels){

}
__global__ 
void sum_(unsigned char* arr, unsigned long long* sum,int to_channel,int channels) {
    extern __shared__ int sdata[128];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    //printf("%d", blockDim.x);
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int i2= i + blockDim.x;
    unsigned int arr_index = i * channels + to_channel;
    unsigned int arr_index2 = i2 * channels + to_channel;

    sdata[tid] = arr[arr_index]+arr[arr_index2];
    //printf("%d\n", arr_index);
    __syncthreads();
    // do reduction in shared mem
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) sum[to_channel] = sdata[0];
}
 
//0.167ms
void cuda_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_compact_mosaic(tile_x_count, tile_y_count, sums, cpu_mosaic_value, output_global_average);
    int c1 = cfg1(tx_ty_c, 32);
    average <<<c1, 32 >> > (tx_ty_c, d_sums, d_mosaic_value, TILE_PIXELS);
    int toChannel = -1;
    while (++toChannel<channels)
    {
        //printf("%d\n", tx_ty);
        sum_ << <1, tx_ty/2>> > (d_mosaic_value, d_sums, toChannel, channels);
    }

    unsigned long long whole_image_sum[4];
    cudaDeviceSynchronize();
    CUDA_CALL(cudaMemcpy(whole_image_sum, d_sums, channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
   
    for (size_t i = 0; i < channels; i++)
    {
        output_global_average[i] = whole_image_sum[i] / (tile_x_count * tile_y_count);
    }
#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    CUDA_CALL(cudaMemcpy(cpu_mosaic_value, d_mosaic_value, tx_ty_c * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    validate_compact_mosaic(tile_x_count, tile_y_count, sums, cpu_mosaic_value, output_global_average);
#endif    
}


//printf("(%d:tile_x_count)*(%d:t_size_y)*(%d:t_size_x)*(%d:channels)=(%d:count_in_t_line)\n", tile_x_count, t_size_y, t_size_x, channels, count_in_t_line);

//3-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


__global__
void broadcast(int w_h_c, unsigned char* d_output_image_data, unsigned char* mosaic_value
    ,int count_in_tile_line,int count_in_img_line,int TILE_SIZE_channels
    ,int tile_x_count,int channels)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= w_h_c)return;
    int t_y = index / count_in_tile_line;
    int left = index % count_in_img_line;
    int t_x = left / TILE_SIZE_channels;
    int ch = left % channels;
    d_output_image_data[index] = mosaic_value[(t_y*tile_x_count+t_x)*channels+ch];
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
    int count_in_tile_line = tile_x_count * TILE_PIXELS * channels;
    int count_in_img_line = wide * channels;
    int TILE_SIZE_channels = TILE_SIZE * channels;
    int c1 = cfg1(w_h_c, 32);
    broadcast <<<c1, 32 >> > (w_h_c, d_output_image_data, d_mosaic_value
        , count_in_tile_line, count_in_img_line, TILE_SIZE_channels
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
    //CUDA_CALL(cudaFree(d_global_pixel_sum));
#ifdef VALIDATION
    free(sums);
    free(cpu_mosaic_value);
    free(output_image.data);
#endif    

}
