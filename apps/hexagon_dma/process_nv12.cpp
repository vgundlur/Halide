#include <stdio.h>
#include <memory.h>
#include <assert.h>
#include <stdlib.h>
#include "halide_benchmark.h"
#include "pipeline_nv12.h"
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
    
    const int buf_size = (width * height * 3) / 2;
    uint8_t *memory_to_dma_from = (uint8_t *)malloc(buf_size);
    for (int i = 0; i < buf_size;  i++) {
        memory_to_dma_from[i] = ((uint8_t)rand()) >> 1;
    }

    Halide::Runtime::Buffer<uint8_t> input_validation(memory_to_dma_from, width, height, 2);
    Halide::Runtime::Buffer<uint8_t> input(nullptr, width, ((height * 3) / 2));

    void *dma_engine = nullptr;
    halide_hexagon_dma_allocate_engine(nullptr, &dma_engine);

    Halide::Runtime::Buffer<uint8_t> input_y = input.cropped(1, 0, height);    // Luma plane only
    Halide::Runtime::Buffer<uint8_t> input_uv = input.cropped(1, height, (height / 2));  // Chroma plane only, with reduced height

    input_uv.allocate();
    input_y.allocate();

    input_uv.device_wrap_native(halide_hexagon_dma_device_interface(),
                             reinterpret_cast<uint64_t>(memory_to_dma_from));

    halide_hexagon_dma_prepare_for_copy_to_host(nullptr, input_uv, dma_engine, false, eDmaFmt_NV12_UV);

    input_y.device_wrap_native(halide_hexagon_dma_device_interface(),
                             reinterpret_cast<uint64_t>(memory_to_dma_from));

    halide_hexagon_dma_prepare_for_copy_to_host(nullptr, input_y, dma_engine, false, eDmaFmt_NV12_Y);

    input_y.set_device_dirty();
    input_uv.set_device_dirty();
    
    Halide::Runtime::Buffer<uint8_t> output(width, ((height * 3) / 2));
    Halide::Runtime::Buffer<uint8_t> output_y = output.cropped(1, 0, height);    // Luma plane only
    Halide::Runtime::Buffer<uint8_t> output_uv = output.cropped(1, height, (height / 2));  // Chroma plane only, with reduced height

    int result = pipeline_nv12(input_y, input_uv, output_y, output_uv);
    if (result != 0) {
        printf("pipeline failed! %d\n", result);
    }

    for (int y = 0; y < ((height * 3) / 2); y++) {
        for (int x = 0; x < width; x++) {
            uint8_t correct = memory_to_dma_from[x + (y * width)] * 2;
            if (y > (height - 1)) {
                correct = (x % 2) ? correct : correct / 2;
            }
            if (correct != output(x, y)) {
                static int cnt = 0;
                printf("Mismatch at x=%d y=%d : %d != %d\n", x, y, correct, output(x, y));
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

    printf("Success!\n");
    return 0;
}
