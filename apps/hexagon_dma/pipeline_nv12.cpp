#include "Halide.h"

using namespace Halide;

class DmaPipeline : public Generator<DmaPipeline> {
public:
    Input<Buffer<uint8_t>> input_y{"input_y", 2};
    Input<Buffer<uint8_t>> input_uv{"input_uv", 2};
    Output<Buffer<uint8_t>> output_y{"output_y", 2};
    Output<Buffer<uint8_t>> output_uv{"output_uv", 2};

    void generate() {
        Var x{"x"}, y{"y"}, c{"c"};

        // We need a wrapper for the output so we can schedule the
        // multiply update in tiles.
        Func copy_y("copy_y");
        Func copy_uv("copy_uv");
        Func process_u("process_u");
        Func process_v("process_v");

        copy_y(x, y) = input_y(x, y);
        copy_uv(x, y) =  input_uv(x, y);

        process_u(x, y) = copy_uv((2 * x), y);
        process_v(x, y) = copy_uv((2 * x + 1), y) * 2; // different processing for u and v

        output_y(x, y) = copy_y(x, y) * 2;
        output_uv(x, y) = select(x%2 == 0, process_u((x / 2), y), process_v((x / 2), y));

        Var tx("tx"), ty("ty");

        // Break the output into tiles.
        const int tile_width = 256;
        const int tile_height = 128;
  
        // tweak stride/extent to handle UV deinterleaving
        output_y
            .compute_root()
            .tile(x, y, tx, ty, x, y, tile_width, tile_height, TailStrategy::RoundUp);

        output_uv
            .compute_root()
            .tile(x, y, tx, ty, x, y, tile_width, tile_height, TailStrategy::RoundUp);

        // Schedule the copy to be computed at tiles with a
        // circular buffer of two tiles.
        copy_y
            .compute_at(output_y, tx)
            .store_root()
            .fold_storage(x, (tile_width * 2))
            .copy_to_host();

        copy_uv
            .compute_at(output_uv, tx)
            .store_root()
            .fold_storage(x, (tile_width * 2))
            .copy_to_host();
    }

};

HALIDE_REGISTER_GENERATOR(DmaPipeline, dma_pipeline_nv12)
