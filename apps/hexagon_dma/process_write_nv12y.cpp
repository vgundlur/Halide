#include <stdio.h>
#include <memory.h>
#include <assert.h>
#include <stdlib.h>
#include "halide_benchmark.h"
#include "pipeline_write_nv12y.h"
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
    uint8_t *memory_to_dma_from = (uint8_t *)malloc(width * height  );
    for (int i = 0; i < width * height ;  i++) {
        memory_to_dma_from[i] = ((uint8_t)rand()) >> 1;
    }

    Halide::Runtime::Buffer<uint8_t> input_validation(memory_to_dma_from, width, height);
    Halide::Runtime::Buffer<uint8_t> input(memory_to_dma_from, width, height);

    // TODO: We shouldn't need to allocate a host buffer here, but the
    // current implementation of cropping + halide_buffer_copy needs
    // it to work correctly.
    //input.allocate();

    // Give the input the buffer we want to DMA from.
    //input.device_wrap_native(halide_hexagon_dma_device_interface(),
    //                         reinterpret_cast<uint64_t>(memory_to_dma_from));
    //input.set_device_dirty();

    uint8_t *memory_to_dma_to = (uint8_t *)malloc(width * height  );
    Halide::Runtime::Buffer<uint8_t> output(nullptr, width, height); 
    // TODO: We shouldn't need to allocate a host buffer here, but the
    // current implementation of cropping + halide_buffer_copy needs
    // it to work correctly.
    output.allocate();

    // Give the Output buffer we want to DMA to
    output.device_wrap_native(halide_hexagon_dma_device_interface(),
                             reinterpret_cast<uint64_t>(memory_to_dma_to));
    output.set_host_dirty();

    // In order to actually do a DMA transfer, we need to allocate a
    // DMA engine.
    void *dma_engine = nullptr;
    halide_hexagon_dma_allocate_engine(nullptr, &dma_engine);
      
    // We then need to prepare for copying to host. Attempting to copy
    // to host without doing this is an error.
    // The Last parameter 0 indicate DMA Read
    halide_hexagon_dma_prepare_for_copy_to_device(nullptr, output, dma_engine, false, eDmaFmt_NV12_Y);

    int result = pipeline_write_nv12y(input, output);
    if (result != 0) {
        printf("pipeline failed! %d\n", result);
    }
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t correct = memory_to_dma_from[x + y*width ] ;
            if (correct != output(x, y)) {
                static int cnt = 0;
                printf("Mismatch at x=%d y=%d  : %d != %d\n", x, y, correct, output(x, y));
                if (++cnt > 20) abort();
            }
        }
    }


    halide_hexagon_dma_unprepare(nullptr, output);

    // We're done with the DMA engine, release it. This would also be
    // done automatically by device_free.
    halide_hexagon_dma_deallocate_engine(nullptr, dma_engine);

    free(memory_to_dma_from);
    free(memory_to_dma_to);

    printf("Success!\n");
    return 0;
}
