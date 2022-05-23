#include "openmp.h"
#include "helper.h"

#include <stdlib.h>
#include <string.h>
#include <omp.h>
//#include <iostream>

Image input_image;
Image output_image;
unsigned int tile_x_count, tile_y_count;
unsigned long long* sums;
unsigned char* output;

int sums_count;
int sums_len;


void openmp_begin(const Image* in) {
    tile_x_count = in->width / TILE_SIZE;
    tile_y_count = in->height / TILE_SIZE;

    // Allocate buffer for calculating the sum of each tile mosaic
    sums = (unsigned long long*)malloc(tile_x_count * tile_y_count * in->channels * sizeof(unsigned long long));

    // Allocate buffer for storing the output pixel value of each tile
    output = (unsigned char*)malloc(tile_x_count * tile_y_count * in->channels * sizeof(unsigned char));

    // Allocate copy of input image
    input_image = *in;
    input_image.data = (unsigned char*)malloc(in->width * in->height * in->channels * sizeof(unsigned char));
    memcpy(input_image.data, in->data, in->width * in->height * in->channels * sizeof(unsigned char));

    // Allocate output image
    output_image = *in;
    output_image.data = (unsigned char*)malloc(in->width * in->height * in->channels * sizeof(unsigned char));
    sums_count = tile_x_count * tile_y_count * input_image.channels;
    sums_len = sums_count * sizeof(unsigned long long);
}

//8.378ms
void openmp_stage1() {

    // Reset sum memory to 0
    memset(sums, 0, tile_x_count * tile_y_count * input_image.channels * sizeof(unsigned long long));
       
    // Sum pixel data within each tile
    int channels = input_image.channels;
    int wide = input_image.width;
    int t_x;
#pragma omp parallel for
    for (t_x = 0; t_x < tile_x_count; ++t_x) {
        int t_y;
#pragma omp parallel for
        for (t_y = 0; t_y < tile_y_count; ++t_y) {
            const unsigned int tile_index = (t_y * tile_x_count + t_x) * channels;
            const unsigned int tile_offset = (t_y * tile_x_count * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * channels;
            int ch;
            //#pragma omp parallel for
            for (ch = 0; ch < channels; ++ch) {
                
                long long sum = 0;
                int p_x;
#pragma omp parallel for reduction(+: sum)
                for ( p_x = 0; p_x < TILE_SIZE; ++p_x) {
                    int p_y;
                    //#pragma omp parallel for
                    for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
                        // For each colour channel
                        const unsigned int pixel_offset = (p_y * wide + p_x) * channels;
                        // Load pixel
                        const unsigned char pixel = input_image.data[tile_offset + pixel_offset + ch];
                        sum += pixel;
                    }
                }
                sums[tile_index + ch] = sum;

            }

        }
    }
#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_tile_sum(&input_image, sums);
#endif
}

//0.107ms
void openmp_stage2(unsigned char* output_global_average) {
    // Calculate the average of each tile, and sum these to produce a whole image average.
    unsigned long long whole_image_sum[4] = { 0, 0, 0, 0 };  // Only 3 is required for the assignment, but this version hypothetically supports upto 4 channels
    int channels = input_image.channels;
    int tile_x_y_total = tile_x_count * tile_y_count;

    int ch;
    #pragma omp parallel for
    for (ch = 0; ch < channels; ++ch) {
        int t;
        long long sum=0;
        #pragma omp parallel for reduction(+: sum)
        for (t = 0; t < tile_x_y_total; ++t) {
            output[t * channels + ch] = (unsigned char)(sums[t * channels + ch] / TILE_PIXELS);  // Integer division is fine here
            sum += output[t * channels + ch];
        }
        whole_image_sum[ch] = sum;
    }

    // Reduce the whole image sum to whole image average for the return value
    //int ch;
    #pragma omp parallel for
    for (ch = 0; ch < channels; ++ch) {
        output_global_average[ch] = (unsigned char)(whole_image_sum[ch] / (tile_x_y_total));
    }
#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    validate_compact_mosaic(tile_x_count, tile_y_count, sums, output, output_global_average);
#endif    
}

//4.243ms
void openmp_stage3() {

    int channels = input_image.channels;
    int wide = input_image.width;
    // Broadcast the compact mosaic pixels back out to the full image size
    // For each tile

    int t_x; 
#pragma omp parallel for
    for ( t_x = 0; t_x < tile_x_count; ++t_x) {

        int t_y;
#pragma omp parallel for
        for ( t_y = 0; t_y < tile_y_count; ++t_y) {
            const unsigned int tile_index = (t_y * tile_x_count + t_x) * channels;
            const unsigned int tile_offset = (t_y * tile_x_count * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * channels;

            // For each pixel within the tile
            int p_x;
#pragma omp parallel for
            for ( p_x = 0; p_x < TILE_SIZE; ++p_x) {

                int p_y;
//#pragma omp parallel for
                for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    const unsigned int pixel_offset = (p_y * wide + p_x) * channels;
                    // Copy whole pixel
                    memcpy(output_image.data + tile_offset + pixel_offset, output + tile_index, channels);
                }
            }
        }
    }
#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_broadcast(&input_image, output, &output_image);
#endif    
}
void openmp_end(Image* out) {
    out->width = output_image.width;
    out->height = output_image.height;
    out->channels = output_image.channels;
    memcpy(out->data, output_image.data, out->width * out->height * out->channels * sizeof(unsigned char));
    // Release allocations
    free(output_image.data);
    free(input_image.data);
    free(output);
    free(sums);
}

