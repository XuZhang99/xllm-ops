/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "rec_constrained_top_k_tiling.h"

#include <algorithm>
#include <cstdlib>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

enum InputIndex {
  LOGITS = 0,
  SEQUENCE_GROUP,
  FIRST_TOKEN_IDS,
  PREFIX1_OFFSETS,
  PREFIX1_VALUES,
  PREFIX1_PAIR_KEYS,
  PREFIX2_VALUE_OFFSETS,
  PREFIX2_VALUES,
  TEMPERATURES,
};

enum AttrIndex {
  CURRENT_STEP = 0,
  TOP_K,
  MAX_PREFIX1_DEGREE,
  MAX_PREFIX2_DEGREE,
};

enum OutputIndex {
  OUT_TOKENS = 0,
  OUT_LOGPROBS,
};

constexpr int64_t kDefaultSequenceStride = 3;
constexpr int32_t kOutputElementBytes = 4;
constexpr int32_t kGmBlockBytes = 32;

int32_t get_env_int(const char* name, int32_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value) {
    return default_value;
  }
  return static_cast<int32_t>(parsed);
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  auto platform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  const uint32_t core_num = platform.GetCoreNumAiv();

  const gert::StorageShape* logits_shape = context->GetInputShape(LOGITS);
  const gert::StorageShape* sequence_shape =
      context->GetInputShape(SEQUENCE_GROUP);
  const gert::StorageShape* first_token_shape =
      context->GetInputShape(FIRST_TOKEN_IDS);
  const gert::StorageShape* prefix1_pair_shape =
      context->GetInputShape(PREFIX1_PAIR_KEYS);
  const gert::StorageShape* temperature_shape =
      context->GetInputShape(TEMPERATURES);

  const int32_t num_rows =
      static_cast<int32_t>(logits_shape->GetStorageShape().GetDim(0));
  const int32_t vocab_size =
      static_cast<int32_t>(logits_shape->GetStorageShape().GetDim(1));
  const int32_t current_step =
      static_cast<int32_t>(*context->GetAttrs()->GetAttrPointer<int>(
          CURRENT_STEP));
  const int32_t top_k =
      static_cast<int32_t>(*context->GetAttrs()->GetAttrPointer<int>(TOP_K));
  const int32_t max_prefix1_degree =
      static_cast<int32_t>(*context->GetAttrs()->GetAttrPointer<int>(
          MAX_PREFIX1_DEGREE));
  const int32_t max_prefix2_degree =
      static_cast<int32_t>(*context->GetAttrs()->GetAttrPointer<int>(
          MAX_PREFIX2_DEGREE));

  int32_t sequence_stride = kDefaultSequenceStride;
  if (sequence_shape != nullptr) {
    const gert::Shape& storage_shape = sequence_shape->GetStorageShape();
    const size_t dim_num = storage_shape.GetDimNum();
    if (dim_num > 0) {
      sequence_stride =
          static_cast<int32_t>(storage_shape.GetDim(dim_num - 1));
    }
  }

  int32_t temperature_count = 1;
  if (temperature_shape != nullptr) {
    const gert::Shape& storage_shape = temperature_shape->GetStorageShape();
    if (storage_shape.GetDimNum() > 0) {
      temperature_count = static_cast<int32_t>(storage_shape.GetDim(0));
    }
  }

  RecConstrainedTopKTilingData tiling;
  tiling.set_num_rows(num_rows);
  tiling.set_vocab_size(vocab_size);
  tiling.set_top_k(top_k);
  tiling.set_current_step(current_step);
  tiling.set_sequence_stride(sequence_stride);
  tiling.set_first_token_count(
      static_cast<int32_t>(first_token_shape->GetStorageShape().GetDim(0)));
  tiling.set_prefix1_pair_count(
      static_cast<int32_t>(prefix1_pair_shape->GetStorageShape().GetDim(0)));
  tiling.set_temperature_count(temperature_count);
  int32_t max_candidate_count = static_cast<int32_t>(
      first_token_shape->GetStorageShape().GetDim(0));
  if (current_step == 1) {
    max_candidate_count = max_prefix1_degree;
  } else if (current_step >= 2) {
    max_candidate_count = max_prefix2_degree;
  }
  max_candidate_count = std::max(max_candidate_count, 1);
  tiling.set_max_candidate_count(max_candidate_count);
  const bool row_write_aligned =
      top_k > 0 && (top_k * kOutputElementBytes) % kGmBlockBytes == 0;
  const int32_t max_core_num =
      std::min(static_cast<int32_t>(core_num), num_rows);
  const int32_t default_core_num = row_write_aligned ? max_core_num : 1;
  const int32_t requested_core_num =
      get_env_int("XLLM_REC_CONSTRAINED_TOPK_FORCE_CORES", default_core_num);
  const int32_t used_core_num =
      row_write_aligned ? std::max(1, std::min(requested_core_num, max_core_num))
                        : 1;
  tiling.set_used_core_num(used_core_num);
  tiling.set_debug_mode(
      get_env_int("XLLM_REC_CONSTRAINED_TOPK_DEBUG_MODE", 0));

  context->SetBlockDim(used_core_num);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* logits_shape = context->GetInputShape(LOGITS);
  const int64_t top_k =
      static_cast<int64_t>(*context->GetAttrs()->GetAttrPointer<int>(TOP_K));

  gert::Shape* out_tokens_shape = context->GetOutputShape(OUT_TOKENS);
  out_tokens_shape->SetDimNum(2);
  out_tokens_shape->SetDim(0, logits_shape->GetDim(0));
  out_tokens_shape->SetDim(1, top_k);

  gert::Shape* out_logprobs_shape = context->GetOutputShape(OUT_LOGPROBS);
  out_logprobs_shape->SetDimNum(2);
  out_logprobs_shape->SetDim(0, logits_shape->GetDim(0));
  out_logprobs_shape->SetDim(1, top_k);
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(OUT_TOKENS, ge::DT_INT32);
  context->SetOutputDataType(OUT_LOGPROBS, ge::DT_FLOAT);
  return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class RecConstrainedTopK : public OpDef {
 public:
  explicit RecConstrainedTopK(const char* name) : OpDef(name) {
    this->Input("logits")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("sequence_group")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("first_token_ids")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("prefix1_offsets")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("prefix1_values")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("prefix1_pair_keys")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("prefix2_value_offsets")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("prefix2_values")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("temperatures")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("out_tokens")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("out_logprobs")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

    this->Attr("current_step").AttrType(REQUIRED).Int(0);
    this->Attr("top_k").AttrType(REQUIRED).Int(1);
    this->Attr("max_prefix1_degree").AttrType(REQUIRED).Int(1);
    this->Attr("max_prefix2_degree").AttrType(REQUIRED).Int(1);

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
  }
};

OP_ADD(RecConstrainedTopK);

}  // namespace ops
