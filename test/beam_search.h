#pragma once
#ifndef BEAM_SEARCH_H
#define BEAM_SEARCH_H

#include <torch/torch.h>

#include <cstdint>
#include <iostream>
#include <vector>
#include "base/utils_tensor.h"
#include "aclnn_beam_search.h"

namespace beam_search {
struct TensorShapes {
  std::vector<std::vector<int64_t>> token_ids_shape;
  std::vector<std::vector<int64_t>> log_probs_shape;
  std::vector<std::vector<int64_t>> top_tokens_shape;
  std::vector<std::vector<int64_t>> top_probs_shape;
  std::vector<std::vector<int64_t>> output_shape;
};
class BeamSearchBase {
 public:
  BeamSearchBase(int64_t beam_width,
                 int64_t top_k,
                 int64_t request_num,
                 int64_t sequence_length)
      : beam_width_(beam_width),
        top_k_(top_k),
        request_num_(request_num),
        sequence_length_(sequence_length) {
    n_sequences_ = request_num_ * beam_width_;
    shapes.token_ids_shape = {{n_sequences_, sequence_length_}};
    shapes.log_probs_shape = {{n_sequences_, 1}};
    shapes.top_tokens_shape = {{n_sequences_, top_k_}};
    shapes.top_probs_shape = {{n_sequences_, top_k_}};
    shapes.output_shape = {{n_sequences_, sequence_length_ + 1}};
  }
  void create_torch_tensors() {
    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32);
    auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32);
    token_ids = torch::randint(
        /*low=*/4, /*high=*/10, {n_sequences_, sequence_length_}, opt_i32);
    log_probs = torch::rand({n_sequences_, 1}, opt_f32);
    top_tokens =
        torch::randint(/*low=*/4, /*high=*/10, {n_sequences_, top_k_}, opt_i32);
    top_probs = torch::rand({n_sequences_, top_k_}, opt_f32);
    output_token_ids_torch =
        torch::zeros({n_sequences_, sequence_length_ + 1}, opt_i32);
    output_token_ids_op =
        torch::zeros({n_sequences_, sequence_length_ + 1}, opt_i32);
  }
  int32_t beam_width_;
  int32_t top_k_;
  int32_t request_num_;
  int32_t sequence_length_;
  int32_t n_sequences_;
  torch::Tensor token_ids;
  torch::Tensor log_probs;
  torch::Tensor top_tokens;
  torch::Tensor top_probs;
  torch::Tensor output_token_ids_torch;
  torch::Tensor output_token_ids_op;
  TensorShapes shapes;
};
class BeamSearchTorch {
 public:
  BeamSearchTorch(BeamSearchBase& base) : base(base) {}
  void process() {
    torch::Tensor expanded_log_probs = base.log_probs.repeat({1, base.top_k_});
    torch::Tensor candidate_scores = (expanded_log_probs + base.top_probs);
    candidate_scores = candidate_scores.view(
        {base.request_num_, base.beam_width_ * base.top_k_});
    // std::cout << "[BeamSearchTorch] expanded_log_probs:\n" <<
    // expanded_log_probs << std::endl; std::cout << "[BeamSearchTorch]
    // top_probs:\n" << base.top_probs << std::endl; std::cout <<
    // "[BeamSearchTorch] candidate_scores (reshaped):\n" << candidate_scores <<
    // std::endl;
    auto topk_result = at::topk(candidate_scores, base.top_k_, 1, true, true);
    torch::Tensor topk_scores = std::get<0>(topk_result);
    torch::Tensor topk_indices = std::get<1>(topk_result);
    // std::cout << "[BeamSearchTorch] topk_scores:\n" << topk_scores <<
    // std::endl; std::cout << "[BeamSearchTorch] topk_indices:\n" <<
    // topk_indices << std::endl;

    torch::Tensor selected_beam = at::floor_divide(topk_indices, base.top_k_);
    torch::Tensor selected_within_top =
        at::remainder(topk_indices, base.top_k_);
    // std::cout << "[BeamSearchTorch] selected_beam:\n" << selected_beam <<
    // std::endl; std::cout << "[BeamSearchTorch] selected_within_top:\n" <<
    // selected_within_top << std::endl;

    auto device = base.token_ids.device();
    auto options_long =
        torch::TensorOptions().dtype(torch::kLong);
    torch::Tensor request_ids =
        torch::arange(base.request_num_, options_long).view({-1, 1});
    torch::Tensor base_indices = (request_ids * base.beam_width_);
    torch::Tensor orig_seq_indices =
        (base_indices + selected_beam).reshape({-1});
    // std::cout << "[BeamSearchTorch] request_ids:\n" << request_ids <<
    // std::endl; std::cout << "[BeamSearchTorch] base_indices:\n" <<
    // base_indices << std::endl; std::cout << "[BeamSearchTorch]
    // orig_seq_indices (flattened):\n" << orig_seq_indices << std::endl;

    torch::Tensor selected_token_ids =
        base.token_ids.index_select(0, orig_seq_indices);
    torch::Tensor selected_top =
        base.top_tokens.index_select(0, orig_seq_indices);
    torch::Tensor selected_within_top_flat =
        selected_within_top.reshape({-1}).to(torch::kLong);
    torch::Tensor next_tokens =
        selected_top.gather(1, selected_within_top_flat.unsqueeze(1));
    // std::cout << "[BeamSearchTorch] selected_token_ids:\n" <<
    // selected_token_ids << std::endl; std::cout << "[BeamSearchTorch]
    // selected_top:\n" << selected_top << std::endl; std::cout <<
    // "[BeamSearchTorch] selected_within_top_flat:\n" <<
    // selected_within_top_flat << std::endl; std::cout << "[BeamSearchTorch]
    // next_tokens:\n" << next_tokens << std::endl;

    torch::Tensor new_token_ids =
        torch::cat({selected_token_ids, next_tokens}, 1);
    base.output_token_ids_torch = new_token_ids;
    std::cout << "[BeamSearchTorch] new_token_ids (output):\n" <<
    new_token_ids << std::endl;
  }

 private:
  BeamSearchBase& base;
};

class BeamSearchOp {
 public:
  BeamSearchOp(BeamSearchBase& base) : base(base) {}
  void process(aclrtStream stream) {
    this->create_tensors();
    CHECK_ACL_SUCCESS(execute_beam_search_operator(token_ids,
                                                   log_probs,
                                                   top_tokens,
                                                   top_probs,
                                                   output_token_ids,
                                                   stream),
                      "execute_beam_search_operator failed");
  }
  int execute_beam_search_operator(aclTensor* token_ids,
                                   aclTensor* log_probs,
                                   aclTensor* top_tokens,
                                   aclTensor* top_probs,
                                   aclTensor* output_token_ids,
                                   aclrtStream stream);
  void destroy_tensors() {
    aclDestroyTensor(token_ids);
    aclDestroyTensor(log_probs);
    aclDestroyTensor(top_tokens);
    aclDestroyTensor(top_probs);
    aclDestroyTensor(output_token_ids);
  }

 private:
  void create_tensors() {
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(
            base.shapes.token_ids_shape[0], base.token_ids, &token_ids),
        "create input_a Tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(
            base.shapes.log_probs_shape[0], base.log_probs, &log_probs),
        "create input_b Tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(
            base.shapes.top_tokens_shape[0], base.top_tokens, &top_tokens),
        "create top_tokens Tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(
            base.shapes.top_probs_shape[0], base.top_probs, &top_probs),
        "create top_probs Tensor failed");
    CHECK_ACL_SUCCESS(
        utils::create_tensor_from_torch(base.shapes.output_shape[0],
                                        base.output_token_ids_op,
                                        &output_token_ids),
        "create output_token_ids_op Tensor failed");
  }
  aclTensor* token_ids = nullptr;
  aclTensor* log_probs = nullptr;
  aclTensor* top_tokens = nullptr;
  aclTensor* top_probs = nullptr;
  aclTensor* output_token_ids = nullptr;
  BeamSearchBase& base;
};
int BeamSearchOp::execute_beam_search_operator(aclTensor* token_ids,
                                               aclTensor* log_probs,
                                               aclTensor* top_tokens,
                                               aclTensor* top_probs,
                                               aclTensor* output_token_ids,
                                               aclrtStream stream) {
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  auto ret = aclnnBeamSearchGetWorkspaceSize(
      token_ids, log_probs, top_tokens, top_probs, output_token_ids, &workspaceSize, &executor);
  CHECK_ACL_SUCCESS(ret, "aclnn_beam_search_operator failed");
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_ACL_SUCCESS(ret, "aclnn_beam_search_operator failed");
  }
  ret = aclnnBeamSearch(workspaceAddr, workspaceSize, executor, stream);
  CHECK_ACL_SUCCESS(ret, "aclnn_beam_search_operator failed");

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  return ACL_SUCCESS;
}

}  // namespace beam_search

#endif