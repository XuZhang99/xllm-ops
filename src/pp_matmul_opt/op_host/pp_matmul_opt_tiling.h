
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PpMatmulOptTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, m);
  TILING_DATA_FIELD_DEF(uint32_t, k);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, rows);
  TILING_DATA_FIELD_DEF(uint32_t, mmadOffsetMax);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PpMatmulOpt, PpMatmulOptTilingData)
}
