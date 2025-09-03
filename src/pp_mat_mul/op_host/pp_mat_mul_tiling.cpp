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
 * \file pp_mat_mul_tiling.cc
 * \brief
 */
#include "pp_mat_mul_tiling.h"
#include "pp_mat_mul_default.h"
#include "pp_mat_mul_info.h"

// #include "tiling/tiling_templates_registry.h"
#include "register/op_def_registry.h"
// #include "cube_tiling_runtime.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"

using namespace optiling::pp_mat_mul;

namespace optiling {

static ge::graphStatus PpMatmulTilingFunc(gert::TilingContext* context)
{
    // OP_TILING_CHECK(context == nullptr,
    //                 CUBE_INNER_ERR_REPORT("PpMatMul", "context is null"),
    //                 return ge::GRAPH_FAILED);
    size_t sysWorkspaceSize = static_cast<size_t>(16 * 1024 * 1024);
    size_t *currentWorkSpace = context->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;

    PpMatMulDefault ppMatmulDefault(context);
    // OP_LOGD(context->GetNodeName(), "Tiling start.");
    ppMatmulDefault.DoTiling();
    auto res = ppMatmulDefault.PostTiling();
    // OP_LOGD(context->GetNodeName(), "Tiling end.");
    return res;
}

static ge::graphStatus TilingPrepareForPpMatmul(gert::TilingParseContext *context) {
    // OP_TILING_CHECK(context == nullptr,
                    // CUBE_INNER_ERR_REPORT("PpMatMul", "context is null"),
                    // return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    // OP_TILING_CHECK(platformInfo == nullptr,
    //                 CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
    //                 return ge::GRAPH_FAILED);

    auto hardwareInfoPtr = context->GetCompiledInfo<HardwareInfo>();
    // OP_TILING_CHECK(hardwareInfoPtr == nullptr,
    //                 CUBE_INNER_ERR_REPORT(context->GetNodeName(), "hardwareInfoPtr is null"),
    //                 return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    hardwareInfoPtr->coreNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, hardwareInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, hardwareInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, hardwareInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, hardwareInfoPtr->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, hardwareInfoPtr->l0cSize);

    // OP_LOGI(context->GetNodeName(),
    //         "parse hardware info success coreNum:%lu, l2Size:%lu, l1Size:%lu, l0aSize:%lu, l0bSize:%lu, \
    //         l0cSize:%lu, hbmBandWidth:%lu, l2BandWidth:%lu",
    //         hardwareInfoPtr->coreNum,
    //         hardwareInfoPtr->l2Size,
    //         hardwareInfoPtr->l1Size,
    //         hardwareInfoPtr->l0aSize,
    //         hardwareInfoPtr->l0bSize,
    //         hardwareInfoPtr->l0cSize,
    //         hardwareInfoPtr->hbmBandWidth,
    //         hardwareInfoPtr->l2BandWidth);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PpMatMul)
    .Tiling(PpMatmulTilingFunc)
    .TilingParse<HardwareInfo>(TilingPrepareForPpMatmul);
}