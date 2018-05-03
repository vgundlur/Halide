#include <stdio.h>
#include <memory.h>
#include <assert.h>
#include <stdlib.h>
#include "halide_benchmark.h"
#include "pipeline_write_nv12.h"
#include "HalideRuntimeHexagonDma.h"
#include "HalideBuffer.h"
#include "../../src/runtime/mini_hexagon_dma.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s width height\n", argv[0]);
        return 0;
    }

    const int width = atoi(argv[1]);
    const int height = atoi(argv[2]);

    // Fill the input buffer with random data. This is just a plain old memory buffer
    uint8_t *memory_to_dma_from = (uint8_t *)malloc(width * height * 1.5  );
    for (int i = 0; i < width * height * 1.5 ;  i++) {
        memory_to_dma_from[i] = ((uint8_t)rand()) >> 1;
    }

    Halide::Runtime::Buffer<uint8_t> input_validation(memory_to_dma_from, width, height, 2);
    Halide::Runtime::Buffer<uint8_t> input(nullptr, width, (3 * height) / 2);
    
    // In order to actually do a DMA transfer, we need to allocate a
    // DMA engine.
    void *dma_engine = nullptr;
    halide_hexagon_dma_allocate_engine(nullptr, &dma_engine);

    Halide::Runtime::Buffer<uint8_t> input_y = input.cropped(1, 0, height);    // Luma plane only
    Halide::Runtime::Buffer<uint8_t> input_uv = input.cropped(1, height, height / 2);  // Chroma plane only, with reduced height


    // TODO: We shouldn't need to allocate a host buffer here, but the
    // current implementation of cropping + halide_buffer_copy needs
    // it to work correctly.
    input_uv.allocate();
    input_y.allocate();

    input_uv.embed(2, 0);
    input_uv.raw_buffer()->dim[2].extent = 2;
    input_uv.raw_buffer()->dim[2].stride = 1;

    input_uv.raw_buffer()->dim[0].stride = 2;
    input_uv.raw_buffer()->dim[0].extent = width / 2;


    // Give the input the buffer we want to DMA from.
    input_uv.device_wrap_native(halide_hexagon_dma_device_interface(),
                             reinterpret_cast<uint64_t>(memory_to_dma_from));


    input_y.device_wrap_native(halide_hexagon_dma_device_interface(),
                             reinterpret_cast<uint64_t>(memory_to_dma_from));

    input_y.set_device_dirty();
    input_uv.set_device_dirty();


    uint8_t *memory_to_dma_to = (uint8_t *)malloc(width * ((height * 3) / 2));

    Halide::Runtime::Buffer<uint8_t> output(nullptr, width, (3 * height) / 2);

    Halide::Runtime::Buffer<uint8_t> output_y = output.cropped(1, 0, height);    // Luma plane only
    Halide::Runtime::Buffer<uint8_t> output_uv = output.cropped(1, height, height / 2);  // Chroma plane only, with reduced height


    // TODO: We shouldn't need to allocate a host buffer here, but the
    // current implementation of cropping + halide_buffer_copy needs
    // it to work correctly.
    output_uv.allocate();
    output_y.allocate();

    output_uv.embed(2, 0);
    output_uv.raw_buffer()->dim[2].extent = 2;
    output_uv.raw_buffer()->dim[2].stride = 1;

    output_uv.raw_buffer()->dim[0].stride = 2;
    output_uv.raw_buffer()->dim[0].extent = width / 2;


    // Give the input the buffer we want to DMA from.
    output_uv.device_wrap_native(halide_hexagon_dma_device_interface(),
                             reinterpret_cast<uint64_t>(memory_to_dma_to));
    output_y.device_wrap_native(halide_hexagon_dma_device_interface(),
                             reinterpret_cast<uint64_t>(memory_to_dma_to));
    // We then need to prepare for copying to host. Attempting to copy
    // to host without doing this is an error.
    halide_hexagon_dma_prepare_for_copy_to_device(nullptr, output_y, dma_engine, false, eDmaFmt_NV12_Y);
    halide_hexagon_dma_prepare_for_copy_to_device(nullptr, output_uv, dma_engine, false, eDmaFmt_NV12_UV);


    output_y.set_host_dirty();
    output_uv.set_host_dirty();

      
    int result = pipeline_write_nv12(input_y, input_uv, output_y, output_uv);
    if (result != 0) {
        printf("pipeline failed! %d\n", result);
    }
    printf("Verification Begin!! \n");

    for (int y = 0; y < 1.5 * height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t correct = memory_to_dma_from[x +  y * width];
            if (correct != memory_to_dma_to[x +  y * width] ) {
                static int cnt = 0;
                printf("Mismatch at x=%d y=%d : %d != %d\n", x, y, correct, memory_to_dma_to[x +  y * width] );
                if (++cnt > 20) abort();
            }
        }
    }

    halide_hexagon_dma_unprepare(nullptr, input_y);
    halide_hexagon_dma_unprepare(nullptr, input_uv);

    // We're done with the DMA engine, release it. This would also be
    // done automatically by device_free.
    halide_hexagon_dma_deallocate_engine(nullptr, dma_engine);

    free(memory_to_dma_from);
    free(memory_to_dma_to);

    printf("Success!\n");
    return 0;
}
