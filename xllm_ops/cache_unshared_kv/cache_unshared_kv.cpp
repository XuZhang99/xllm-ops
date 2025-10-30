#include "kernel_operator.h"
#include "cache_unshared_kv.h"

extern "C" __global__ __aicore__ void cache_unshared_kv(GM_ADDR x_key_block, GM_ADDR x_value_block, 
        GM_ADDR curr_key, GM_ADDR curr_value, GM_ADDR decode_step, 
        GM_ADDR select_key_block, GM_ADDR select_value_block, 
        GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe pipe;
    kernels::CacheUnsharedKvKernel<DTYPE_X_KEY_BLOCK> op(&pipe);
    op.Init(x_key_block, x_value_block, curr_key, curr_value, decode_step, &tiling_data, select_key_block, select_value_block);
    op.process();
}
