#include "select_unshared_kv.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void select_unshared_kv(GM_ADDR beam_index, GM_ADDR x_key_block, GM_ADDR x_value_block, 
    GM_ADDR group_token_num,
    GM_ADDR select_key_block, GM_ADDR select_value_block, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    TPipe pipe;
    kernels::SelectUnsharedKVKernel<DTYPE_X_KEY_BLOCK, DTYPE_BEAM_INDEX> op(&pipe);
    op.Init(beam_index, x_key_block, x_value_block, group_token_num, workspace, &tiling_data, select_key_block, select_value_block);
    op.process();
}
