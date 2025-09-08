/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef MLA_TILING_H
#define MLA_TILING_H

#include "multi_latent_attention_tiling_dependency.h"
#include "mla.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/tiling_api.h"
 
namespace AtbOps {
ge::graphStatus MLATiling(gert::TilingContext *context);
ge::graphStatus GetMLATilingParam(OpParam::MLA param, const MLAInfo &mmInfo,
    uint32_t &blockDim, uint32_t *tilingParam, uint64_t tilingParamSize);
}

#endif // MLA_TILING_H
