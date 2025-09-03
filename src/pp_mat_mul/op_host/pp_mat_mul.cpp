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
 * \file pp_mat_mul.cpp
 * \brief
 */
#include <cstdint>
#include "register/op_def_registry.h"
#include "pp_mat_mul_tiling.h"



namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const bool trans_a = *context->GetAttrs()->GetBool(0);
    const bool trans_b = *context->GetAttrs()->GetBool(1);
    const gert::Shape* x1_shape = context->GetInputShape(0);
    const gert::Shape* x2_shape = context->GetInputShape(1);
    const gert::Shape* bias_shape = context->GetInputShape(2);
    size_t dimNumA = x1_shape->GetDimNum();
    size_t dimNumB = x2_shape->GetDimNum();
    size_t dimNumBias = bias_shape->GetDimNum();
    gert::Shape* y_shape = context->GetOutputShape(0);
    size_t dimNumY = y_shape->GetDimNum();
    if (dimNumA == 3) {
        // context->SetOutputDim(0, 0, context->GetInputDim(0, 0));
        y_shape->SetDim(0,x1_shape->GetDim(0));
        if (trans_a) {
            y_shape->SetDim(1,x1_shape->GetDim(2));
        } else {
            y_shape->SetDim(1,x1_shape->GetDim(1));
        }
        if (trans_b) {
            y_shape->SetDim(2,x2_shape->GetDim(1));
        } else {
            y_shape->SetDim(2,x2_shape->GetDim(2));
        }
    }else{
        if (trans_a) {
            y_shape->SetDim(0,x1_shape->GetDim(1));
        } else {
            y_shape->SetDim(0,x1_shape->GetDim(0));
        }
        if (trans_b) {
            y_shape->SetDim(1,x2_shape->GetDim(0));
        }   else {
            y_shape->SetDim(1,x2_shape->GetDim(1));
        }
    }
    return GRAPH_SUCCESS;

}
} // namespace ge

namespace ops {
class PpMatMul : public OpDef {
public:
    explicit PpMatMul(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("yRef")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("transpose_x1")
            .AttrType(OPTIONAL)
            .Bool(false);
        this->Attr("transpose_x2")
            .AttrType(OPTIONAL)
            .Bool(false);
        this->Attr("en_shuffle_K")
            .AttrType(OPTIONAL)
            .Bool(false);
        this->Attr("compute_type")
            .AttrType(OPTIONAL)
            .Int(0);
        this->SetInferShape(ge::InferShape);
        OpAICoreConfig aicConfig;
        aicConfig.DynamicCompileStaticFlag(true)
                .DynamicFormatFlag(false)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .NeedCheckSupportFlag(false)
                .ExtendCfgInfo("opFile.value", "pp_mat_mul")
                .ExtendCfgInfo("opInterface.value", "pp_mat_mul");
        this->AICore().AddConfig("ascend910b", aicConfig);
        this->AICore().AddConfig("ascend910_93", aicConfig);
        
    }
};

OP_ADD(PpMatMul);
}
