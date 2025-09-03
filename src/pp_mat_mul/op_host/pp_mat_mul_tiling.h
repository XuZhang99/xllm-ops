/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file pp_mat_mul_tiling.h
 * \brief
 */
#ifndef __OP_HOST_PP_MAT_MUL_TILING_H__
#define __OP_HOST_PP_MAT_MUL_TILING_H__
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PpMatmulTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, m);
  TILING_DATA_FIELD_DEF(uint32_t, k);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, m0);
  TILING_DATA_FIELD_DEF(uint32_t, k0);
  TILING_DATA_FIELD_DEF(uint32_t, n0);
  TILING_DATA_FIELD_DEF(uint32_t, mLoop);
  TILING_DATA_FIELD_DEF(uint32_t, kLoop);
  TILING_DATA_FIELD_DEF(uint32_t, nLoop);
  TILING_DATA_FIELD_DEF(uint32_t, coreLoop);
  TILING_DATA_FIELD_DEF(uint32_t, swizzlCount);
  TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint32_t, swizzlDirect);
  TILING_DATA_FIELD_DEF(uint32_t, splitk);
  TILING_DATA_FIELD_DEF(uint32_t, enShuffleK);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PpMatMul, PpMatmulTilingData)
}
#endif // __OP_HOST_BATCH_MAT_MUL_V3_TILING_H__