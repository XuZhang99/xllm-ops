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
 * \file pp_mat_mul_default.h
 * \brief
 */
#ifndef __OP_HOST_PP_MAT_MUL_DEFAULT_H__
#define __OP_HOST_PP_MAT_MUL_DEFAULT_H__

#include "register/op_def_registry.h"
#include "pp_mat_mul_info.h"

namespace optiling {
namespace pp_mat_mul {

class PpMatMulDefault{
public:
    explicit PpMatMulDefault(gert::TilingContext* context) : context_(context) {}
    virtual ~PpMatMulDefault() {}

    virtual bool GetMatMulInfo();
    void GetHardwareInfo();
    virtual bool GetTilingKey();
    bool GetMatMulTilingData();
    virtual void DoTiling();
    void PrintTiling();
    virtual ge::graphStatus PostTiling();
    gert::TilingContext *context_ = nullptr;
    MatMulInfo matMulInfo_;
    PpMatmulDefaultTilingData ppMatmulDefaultTilingData_;
    HardwareInfo hardwareInfo_;
    uint64_t kernelKey_;
};
}
}
#endif
