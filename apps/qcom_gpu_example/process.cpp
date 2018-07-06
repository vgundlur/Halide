#include <stdio.h>
#include <memory.h>
#include <assert.h>
#include <stdlib.h>

#include "halide_benchmark.h"

#include "pipeline_cpu.h"
#include "pipeline_gpu.h"

#include "HalideRuntimeHexagonHost.h"
#include "HalideBuffer.h"

#include "ion_allocation.h"


using namespace Halide;

template <typename T>
T clamp(T x, T min, T max) {
    if (x < min) x = min;
    if (x > max) x = max;
    return x;
}

int main(int argc, char **argv) {

    if (argc < 3) {
        printf("Usage: %s (cpu|gpu) timing_iterations\n", argv[0]);
        return 0;
    }

    bool using_gpu = false;
    int (*pipeline)(halide_buffer_t *, halide_buffer_t*);
    if (strcmp(argv[1], "cpu") == 0) {
        pipeline = pipeline_cpu;
        printf("Using CPU schedule\n");
    } else if (strcmp(argv[1], "gpu") == 0) {
        using_gpu = true;
        pipeline = pipeline_gpu;
        printf("Using GPU schedule\n");
    } else {
        printf("Unknown schedule, valid schedules are cpu, gpu\n");
        return -1;
    }

    int iterations = atoi(argv[2]);

    const int W = 1024;
    const int H = 1024;

    Halide::Runtime::Buffer<uint8_t> in(nullptr, W, H, 3);
    Halide::Runtime::Buffer<uint8_t> out(nullptr, W, H, 3);

    alloc_init();
    uint8_t  *inbuf = NULL;
    uint8_t  *outbuf = NULL;
    int in_fd, out_fd;
    inbuf = (uint8_t  * ) alloc_ion(W*H*3,&in_fd);
    outbuf = (uint8_t * ) alloc_ion(W*H*3,&out_fd);

    if (using_gpu) {
        // Hexagon's device_malloc implementation will also set the host
        // pointer if it is null, giving a zero copy buffer.
        in = Halide::Runtime::Buffer<uint8_t>(inbuf, W, H, 3);
        out = Halide::Runtime::Buffer<uint8_t>(outbuf, W, H, 3);
       
        halide_buffer_t *input_buff = in.raw_buffer();
        input_buff->flags = in_fd;
        input_buff->use_host_ptr_extension = 1;
	input_buff->read_only = 1;
        input_buff->is_cached = 1;

        halide_buffer_t *output_buff = out.raw_buffer();
        output_buff->flags =out_fd;
        output_buff->use_host_ptr_extension = 1;
        output_buff->read_only = 1;
        output_buff->is_cached = 1;

        in.device_malloc(halide_opencl_device_interface());
        out.device_malloc(halide_opencl_device_interface());
        printf("After CL interface in = %p, out = %p \n ",inbuf,outbuf);
    } else {
        in = Halide::Runtime::Buffer<uint8_t>(inbuf, W, H, 3);
        out = Halide::Runtime::Buffer<uint8_t>(outbuf, W, H, 3);
    }
    // Fill the input buffer with random data.
    in.for_each_value([&](uint8_t &x) {
        x = static_cast<uint8_t>(rand());
    });

    printf("Running pipeline...\n");
    double time = Halide::Tools::benchmark(iterations, 1, [&]() {
        int result = pipeline(in, out);
        if (result != 0) {
            printf("pipeline failed! %d\n", result);
        }
    });

    printf("Done, time: %g s\n", time);

    // Processing done Copy to Host
    if (using_gpu) {
        out.copy_to_host((void *)halide_opencl_device_interface());
    }

    printf("Verification Begin!! \n");
    // Validate that the algorithm did what we expect.
    const uint16_t gaussian5[] = { 1, 4, 6, 4, 1 };
    out.for_each_element([&](int x, int y, int c) {
        uint16_t blur = 0;
        for (int rx = -2; rx <= 2; rx++) {
            uint16_t blur_y = 0;
            for (int ry = -2; ry <= 2; ry++) {
                uint16_t in_rxy =
                    in(clamp(x + rx, 0, W - 1), clamp(y + ry, 0, H - 1), c);
                blur_y += in_rxy * gaussian5[ry + 2];
            }
            blur_y += 8;
            blur_y /= 16;

            blur += blur_y * gaussian5[rx + 2];
        }
        blur += 8;
        blur /= 16;

        uint8_t out_xy = out(x, y, c);
        if (blur != out_xy) {
            printf("Mismatch at %d %d %d: %d != %d\n", x, y, c, out_xy, blur);
            abort();
        }

    });

   
    alloc_ion_free ((void *) inbuf);
    alloc_ion_free ((void *)outbuf);
    alloc_finalize();
    printf("Success!\n");
    return 0;
}
