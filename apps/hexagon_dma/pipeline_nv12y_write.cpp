#include "Halide.h"

using namespace Halide;

class DmaPipeline : public Generator<DmaPipeline> {
public:
    Input<Buffer<uint8_t>> input_y{"input_y", 2};
    Output<Buffer<uint8_t>> output_y{"output_y", 2};

    void generate() {
        Var x{"x"}, y{"y"};

        // We need a wrapper for the output so we can schedule the
        // multiply update in tiles.
        Func read_y("read_y");
        Func write_y("write_y");
        Func processed_y("processed_y");
        
        // DMA Read
        read_y(x, y) = input_y(x, y);

        // Do Processing and DMA Write
        processed_y(x, y) = read_y(x, y) * 2;
        // DMA Write
        write_y(x, y) = processed_y(x, y);
 
        output_y(x, y) =  write_y(x, y);

        Var tx("tx"), ty("ty");

        // Break the output into tiles.
        const int tile_width = 256;
        const int tile_height = 128;
  

        output_y
            .compute_root()
            .tile(x, y, tx, ty, x, y, tile_width, tile_height, TailStrategy::RoundUp);
        processed_y 
            .compute_root()
            .tile(x, y, tx, ty, x, y, tile_width, tile_height, TailStrategy::RoundUp);

        // Schedule the copy to be computed at tiles with a
        // circular buffer of two tiles.
        read_y
            .compute_at(processed_y, tx)
            .store_root()
            .fold_storage(x, tile_width * 2 )
            .copy_to_host();
        write_y
            .compute_at(output_y, tx)
            .store_root()
            .fold_storage(x, tile_width * 2 )
            .copy_to_device();

    }

};

HALIDE_REGISTER_GENERATOR(DmaPipeline, dma_pipeline_nv12y_write)
