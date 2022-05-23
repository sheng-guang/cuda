#include "cuda.cuh"

#include <cstring>

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
// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
unsigned long long* d_global_pixel_sum;
size_t w_h_c_sizeof_c;

//-------------------------------------------------------------------------------------------------------------------
unsigned long long* sums;
unsigned char* cpu_mosaic_value;
Image output_image;
void cuda_begin(const Image *in) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

    tile_x_count = in->width / TILE_SIZE;
    tile_y_count = in->height / TILE_SIZE;

    // Allocate buffer for calculating the sum of each tile mosaic
    CUDA_CALL(cudaMalloc(&d_sums, tile_x_count * tile_y_count * in->channels * sizeof(unsigned long long)));
    sums = (unsigned long long*)malloc(tile_x_count * tile_y_count * in->channels * sizeof(unsigned long long));

    // Allocate buffer for storing the output pixel value of each tile
    CUDA_CALL(cudaMalloc(&d_mosaic_value, tile_x_count * tile_y_count * in->channels * sizeof(unsigned char)));
    cpu_mosaic_value = (unsigned char*)malloc(tile_x_count * tile_y_count * in->channels * sizeof(unsigned char));


    w_h_c_sizeof_c = in->width * in->height * in->channels * sizeof(unsigned char);
    // Allocate copy of input image
    input_image = *in;
    input_image.data = (unsigned char*)malloc(w_h_c_sizeof_c);
    memcpy(input_image.data, in->data, w_h_c_sizeof_c);

    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, w_h_c_sizeof_c));
    CUDA_CALL(cudaMemcpy(d_input_image_data, in->data, w_h_c_sizeof_c, cudaMemcpyHostToDevice));

    // Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, w_h_c_sizeof_c));

    // Allocate and zero buffer for calculation global pixel average
    CUDA_CALL(cudaMalloc(&d_global_pixel_sum, in->channels * sizeof(unsigned long long)));
    output_image = *in;
    output_image.data = (unsigned char*)malloc(in->width * in->height * in->channels * sizeof(unsigned char));
}
void cuda_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    skip_tile_sum(&input_image, sums);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
     validate_tile_sum(&input_image, sums);
#endif
}
void cuda_stage2(unsigned char* output_global_average) {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    skip_compact_mosaic(tile_x_count, tile_y_count, sums, cpu_mosaic_value, output_global_average);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    validate_compact_mosaic(tile_x_count, tile_y_count, sums, cpu_mosaic_value, output_global_average);
#endif    
}
void cuda_stage3() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    skip_broadcast(&input_image, cpu_mosaic_value, &output_image);
    CUDA_CALL(cudaMemcpy(d_output_image_data, output_image.data, w_h_c_sizeof_c, cudaMemcpyHostToDevice));
#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
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

}
