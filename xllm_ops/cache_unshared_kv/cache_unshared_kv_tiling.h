
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CacheUnsharedKvTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, total_tokens);
  TILING_DATA_FIELD_DEF(uint32_t, head_num);
  TILING_DATA_FIELD_DEF(uint32_t, head_dim);
  TILING_DATA_FIELD_DEF(uint32_t, copy_beam_per_task);
  TILING_DATA_FIELD_DEF(uint32_t, total_task);
  TILING_DATA_FIELD_DEF(uint32_t, copy_head_num_per_loop);
  TILING_DATA_FIELD_DEF(uint32_t, copy_repeat_times);
  TILING_DATA_FIELD_DEF(uint32_t, copy_head_num_tail);
  TILING_DATA_FIELD_DEF(uint32_t, max_decode_step);
  TILING_DATA_FIELD_DEF(uint32_t, used_core_num);
  TILING_DATA_FIELD_DEF(uint32_t, block_beam_stride);
  TILING_DATA_FIELD_DEF(uint32_t, block_head_stride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CacheUnsharedKv, CacheUnsharedKvTilingData)
}
