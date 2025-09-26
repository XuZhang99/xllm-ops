#include "./beam_search.h"

#include "kernel_operator.h"
template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::Init(
    GM_ADDR token_ids, GM_ADDR log_probs, GM_ADDR top_tokens, GM_ADDR top_probs,
    GM_ADDR out_token_ids, int32_t num_sequences, int32_t sequence_length,
    int32_t beam_width, int32_t top_k, int32_t request_num, int32_t core_num,
    int32_t min_size, TopkTiling &topkTilingData, TopkTiling &topKTilingData1) {
  token_ids_gm.SetGlobalBuffer((__gm__ TokenIdType *)token_ids);
  log_probs_gm.SetGlobalBuffer((__gm__ LogProbType *)log_probs);
  top_tokens_gm.SetGlobalBuffer((__gm__ TokenIdType *)top_tokens);
  top_probs_gm.SetGlobalBuffer((__gm__ LogProbType *)top_probs);
  out_token_ids_gm.SetGlobalBuffer((__gm__ TokenIdType *)out_token_ids);
  this->num_sequences = num_sequences;
  this->sequence_length = sequence_length;
  this->beam_width = beam_width;
  this->top_k = top_k;
  this->request_num = request_num;
  this->core_num = core_num;
  this->topkTilingData = topkTilingData;
  this->topKTilingData1 = topKTilingData1;
  this->align_beam_width = AlignUp(this->beam_width, 32);
  this->align_top_k = AlignUp(this->top_k, 32 / sizeof(LogProbType));
  this->min_size = min_size;
  this->round_length = 24;
  pipe.InitBuffer(log_probs_in_que, 1,
                  this->align_beam_width * sizeof(LogProbType)); // 1k
  pipe.InitBuffer(tmp_queue, 1, this->round_length * 1024); // 48 is logic number 96k
  pipe.InitBuffer(tmp_buf2,
                  this->align_beam_width * sizeof(LogProbType) +
                      this->align_beam_width * sizeof(int64_t)); // 5k
  int32_t copy_align_mask =
      AlignUp(this->align_beam_width, 256 / sizeof(LogProbType));
  pipe.InitBuffer(tmp_buf3, this->align_beam_width * sizeof(LogProbType));    // 1k
  pipe.InitBuffer(tmp_buf4, this->align_beam_width * sizeof(int32_t)); // 2k
  pipe.InitBuffer(tmp_buf5, this->beam_width * sizeof(LogProbType));   // 1k
  pipe.InitBuffer(tmp_buf6,
                  this->align_beam_width * sizeof(LogProbType) * 2);       // 2k
  pipe.InitBuffer(tmp_buf7, this->align_beam_width * sizeof(int32_t) * 2); // 4k
  pipe.InitBuffer(tmp_buf8, this->align_beam_width * sizeof(int32_t));     // 2k
  pipe.InitBuffer(tmp_buf9, this->round_length * 1024);                                    // 2k
  int32_t align_sequence_length = AlignUp(this->sequence_length, 32);
  pipe.InitBuffer(out_token_ids_out_que, 1,
                  align_sequence_length * sizeof(TokenIdType)); // 8k or 16k
  pipe.InitBuffer(token_ids_in_que, 1,
                  align_sequence_length * sizeof(TokenIdType)); // 1k
  pipe.InitBuffer(tmp_buf10, this->min_size * sizeof(int32_t));
}
template <typename TokenIdType, typename LogProbType>
__aicore__ inline int32_t
BeamSearch<TokenIdType, LogProbType>::AlignUp(int32_t value,
                                              int32_t alignment) {
  if (value % alignment != 0) {
    value = (value / alignment + 1) * alignment;
  }
  return value;
}
template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::Process() {
  for (int32_t i = 0; i < this->request_num; i++) {
    Psum(i);
  }
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::AlignUpDataCopy(
    AscendC::LocalTensor<TokenIdType> dst,
    AscendC::GlobalTensor<TokenIdType> src, int32_t length) {
  int32_t block_size = 32 / sizeof(TokenIdType);
  DataCopyPadExtParams<TokenIdType> params;
  params.isPad = true;
  params.paddingValue = 0;
  params.leftPadding = 0;
  params.rightPadding = block_size - length;
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(TokenIdType);
  copyParams.blockCount = 1;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  AscendC::DataCopyPad(dst, src, copyParams, params);
}
template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::AlignUpDataCopyGm(
    AscendC::GlobalTensor<TokenIdType> dst,
    AscendC::LocalTensor<TokenIdType> src, int32_t length) {
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(TokenIdType);
  copyParams.blockCount = 1;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  copyParams.rsv = 0;
  AscendC::DataCopyPad(dst, src, copyParams);
}
template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearch<TokenIdType, LogProbType>::AlignUpDataCopyProb(
    AscendC::LocalTensor<LogProbType> dst,
    AscendC::GlobalTensor<LogProbType> src, int32_t length) {
  int32_t block_size = 32 / sizeof(LogProbType);
  AscendC::DataCopyPadExtParams<LogProbType> params;
  params.isPad = true;
  params.paddingValue = 0;
  params.leftPadding = 0;
  params.rightPadding = block_size - length;
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(LogProbType);
  copyParams.blockCount = 1;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  AscendC::DataCopyPad(dst, src, copyParams, params);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearch<TokenIdType, LogProbType>::AlignUpDataCopyProbSlice(
    AscendC::LocalTensor<LogProbType> dst,
    AscendC::GlobalTensor<LogProbType> src, int32_t length,
    int32_t slice_length) {
  int32_t block_size = 32 / sizeof(LogProbType);
  AscendC::DataCopyPadExtParams<LogProbType> params;
  params.isPad = true;
  params.paddingValue = 0;
  params.leftPadding = 0;
  params.rightPadding = block_size - length;
  AscendC::DataCopyExtParams copyParams;
  copyParams.blockLen = length * sizeof(LogProbType);
  copyParams.blockCount = slice_length;
  copyParams.srcStride = 0;
  copyParams.dstStride = 0;
  AscendC::DataCopyPad(dst, src, copyParams, params);
  AscendC::PipeBarrier<PIPE_MTE2>();
}
template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::AlignUpCopyInt(
    LocalTensor<int32_t> dst, AscendC::LocalTensor<int32_t> src,
    int32_t length) {
  int32_t mask = 64;
  AscendC::CopyRepeatParams copyParams;
  copyParams.dstStride = 1;
  copyParams.srcStride = 1;
  copyParams.dstRepeatSize = 8;
  copyParams.srcRepeatSize = 8;
  AscendC::Copy<int32_t, true>(dst, src, mask, length / 64, copyParams);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::AlignUpCopyProb(
    LocalTensor<LogProbType> dst, AscendC::LocalTensor<LogProbType> src,
    int32_t length) {
  int32_t mask = 256 / sizeof(LogProbType);
  AscendC::CopyRepeatParams copyParams;
  copyParams.dstStride = 1;
  copyParams.srcStride = 1;
  copyParams.dstRepeatSize = 8;
  copyParams.srcRepeatSize = 8;
  AscendC::Copy<LogProbType, true>(dst, src, mask, length / mask, copyParams);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void
BeamSearch<TokenIdType, LogProbType>::Psum(int32_t request_idx) {
  LocalTensor<LogProbType> prefix_top_probs = tmp_buf3.Get<LogProbType>();
  AscendC::Duplicate<LogProbType>(prefix_top_probs, static_cast<LogProbType>(0),this->align_beam_width);
  LocalTensor<int32_t> prefix_top_index = tmp_buf4.GetWithOffset<int32_t>(this->align_beam_width, 0);
  AscendC::Duplicate<int32_t>(prefix_top_index, static_cast<int32_t>(0),this->align_beam_width);
  LocalTensor<LogProbType> log_probs_local = log_probs_in_que.AllocTensor<LogProbType>();
  AlignUpDataCopyProb(log_probs_local,log_probs_gm[request_idx * this->beam_width],this->beam_width);
  log_probs_in_que.EnQue(log_probs_local);
  int32_t beam_width_round = (this->beam_width + this->round_length - 1) / this->round_length;
  LocalTensor<LogProbType> log_probs_per = log_probs_in_que.DeQue<LogProbType>();
  // printf("log_probs_per:\n");
  // AscendC::DumpTensor(log_probs_per,0,this->align_beam_width);
  int32_t tail_number = this->beam_width % this->round_length;
  // printf("tail_number: %d\n", tail_number);
  // printf("beam_width_round:%d\n", beam_width_round);
  for (int32_t i = 0; i < beam_width_round; i++) {

    LocalTensor<LogProbType> top_probs_local = tmp_queue.AllocTensor<LogProbType>();

    if (i == beam_width_round - 1 && tail_number != 0) {
      AlignUpDataCopyProbSlice(
          top_probs_local,
          top_probs_gm[request_idx * this->beam_width * this->top_k + i * this->round_length*this->top_k], this->top_k,
          tail_number);
      this->round_length = tail_number;
    } else {
      AlignUpDataCopyProbSlice(
          top_probs_local,
          top_probs_gm[request_idx * this->beam_width * this->top_k + i * this->round_length*this->top_k], this->top_k,
          this->round_length);
    }
    tmp_queue.EnQue(top_probs_local);

    LocalTensor<LogProbType> top_probs_local_tmp2 = tmp_queue.DeQue<LogProbType>();
    // printf("request_idx:%d\n", request_idx);
    // printf("top_probs_local:\n");
    // AscendC::DumpTensor(top_probs_local,0,this->align_beam_width);
    // printf("offset:%d\n", request_idx * this->beam_width*this->top_k + i * this->round_length*this->top_k);
    // printf("top_probs_gm:\n");
    // AscendC::DumpTensor(top_probs_gm[request_idx * this->beam_width * this->top_k + i * this->round_length*this->top_k],0,this->align_beam_width);
    // AscendC::DumpTensor(top_probs_gm,0,2*this->align_beam_width);
    for (int beam_width_offset = 0; beam_width_offset < this->round_length;
         beam_width_offset++) {
      int32_t log_probs_beam_offset = beam_width_offset;
      // printf("log_probs_beam_offset: %d\n", log_probs_beam_offset);
      LogProbType log_probs_value = log_probs_per.GetValue(log_probs_beam_offset);
      // printf("log_probs_value: %f\n", log_probs_value);
      AscendC::Adds<LogProbType>(
          top_probs_local_tmp2[beam_width_offset * this->align_top_k],
          top_probs_local_tmp2[beam_width_offset * this->align_top_k],
          log_probs_value, this->top_k);
    }
    // printf("top_probs_local_tmp2:\n");
    // AscendC::DumpTensor(top_probs_local_tmp2,0,this->align_beam_width);
    TopKWithSorted(request_idx, top_probs_local_tmp2, prefix_top_probs,
                   prefix_top_index, this->round_length);
    tmp_queue.FreeTensor(top_probs_local_tmp2);
  }
  StackWithOutput(request_idx, prefix_top_probs, prefix_top_index);
  log_probs_in_que.FreeTensor(log_probs_per);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::TopKWithSorted(
    int32_t request_idx, AscendC::LocalTensor<LogProbType> &top_probs_local,
    AscendC::LocalTensor<LogProbType> &prefix_top_probs,
    AscendC::LocalTensor<int32_t> &prefix_top_index, int32_t round_length_top_k) {
  // printf("TopKWithSorted top_probs_local:\n");
  // AscendC::DumpTensor(top_probs_local,0,this->align_beam_width);
  // printf("TopKWithSorted prefix_top_probs:\n");
  // AscendC::DumpTensor(prefix_top_probs,0,this->align_beam_width);
  // printf("TopKWithSorted prefix_top_index:\n");
  // AscendC::DumpTensor(prefix_top_index,0,this->align_beam_width);
  // printf("TopKWithSorted round_length_top_k: %d\n", round_length_top_k);
  int32_t offset = 0;
  LocalTensor<LogProbType> dst_local_value =
      tmp_buf2.GetWithOffset<LogProbType>(
          this->align_beam_width * sizeof(LogProbType), offset);
  offset += this->align_beam_width * sizeof(LogProbType);
  LocalTensor<int32_t> dst_local_index = tmp_buf2.GetWithOffset<int32_t>(
      this->align_beam_width * sizeof(int32_t), offset);
  offset += this->align_beam_width * sizeof(int32_t);
  LocalTensor<int32_t> src_local_index;
  LocalTensor<bool> src_local_finish;
  int32_t block_size = AlignUp(round_length_top_k * this->align_top_k, 32);
  TopKInfo topKInfo;
  topKInfo.outter = 1;
  topKInfo.inner = block_size;
  topKInfo.n = block_size;
  bool isLargest = true;
  int32_t k = this->top_k;
  LocalTensor<uint8_t> tmp_local = tmp_buf10.Get<uint8_t>();

  AscendC::TopK<LogProbType, false, false, false,AscendC::TopKMode::TOPK_NORMAL>(
      dst_local_value, dst_local_index, top_probs_local, src_local_index,
      src_local_finish, tmp_local, k, this->topkTilingData, topKInfo,
      isLargest);
  // printf("TopKWithSorted first topK result dst_local_value:\n");
  // AscendC::DumpTensor(dst_local_value,0,this->align_beam_width);
  // printf("TopKWithSorted first topK result dst_local_index:\n");
  // AscendC::DumpTensor(dst_local_index,0,this->align_beam_width);
  // AscendC::Adds<int32_t>(dst_local_index, dst_local_index,
  //                        static_cast<int32_t>(request_idx * this->beam_width),
  //                        this->beam_width);
  LocalTensor<LogProbType> merge_top_probs = tmp_buf6.Get<LogProbType>();
  int32_t length_probs = AlignUp(this->beam_width, 256 / sizeof(LogProbType));
  int32_t length_index = AlignUp(this->beam_width, 64 / sizeof(int32_t));
  AscendC::DataCopy(merge_top_probs, dst_local_value, this->align_beam_width);
  AscendC::DataCopy(merge_top_probs[this->align_beam_width], prefix_top_probs, this->align_beam_width);
  LocalTensor<int32_t> merge_index = tmp_buf7.Get<int32_t>();
  AscendC::DataCopy(merge_index,dst_local_index,this->align_beam_width);
  AscendC::DataCopy(merge_index[this->align_beam_width], prefix_top_index, this->align_beam_width);
  LocalTensor<LogProbType> dst_merge_probs = tmp_buf5.Get<LogProbType>();
  LocalTensor<int32_t> dst_merge_index = tmp_buf8.Get<int32_t>();
  AscendC::Duplicate<int32_t>(dst_merge_index, static_cast<int32_t>(0),
                              this->align_beam_width);
  int32_t block_size_merge = AlignUp(this->beam_width * 2, 32);
  TopKInfo topKInfo2;
  topKInfo2.outter = 1;
  topKInfo2.inner = block_size_merge;
  topKInfo2.n = block_size_merge;
  AscendC::TopK<LogProbType, true, false, false,AscendC::TopKMode::TOPK_NORMAL>(
          dst_merge_probs,
          dst_merge_index,
          merge_top_probs,
          merge_index,
          src_local_finish,
          tmp_local,
          this->top_k,
          this->topKTilingData1,
          topKInfo2,
          isLargest);
  AscendC::DataCopy(prefix_top_probs, dst_merge_probs, this->align_beam_width);
  // printf("TopKWithSorted second topK result after copy dst_merge_probs:\n");
  // AscendC::DumpTensor(dst_merge_probs,0,this->align_beam_width);
  // printf("TopKWithSorted second topK result after copy dst_merge_index:\n");
  // AscendC::DumpTensor(dst_merge_index,0,this->align_beam_width);
  // AlignUpCopyProb(prefix_top_probs, dst_merge_probs, length_probs);

  AscendC::DataCopy(prefix_top_index, dst_merge_index, this->align_beam_width);
  // AlignUpCopyInt(prefix_top_index, dst_merge_index, length_index);
}

template <typename TokenIdType, typename LogProbType>
__aicore__ inline void BeamSearch<TokenIdType, LogProbType>::StackWithOutput(
    int32_t request_idx, AscendC::LocalTensor<LogProbType> &prefix_top_probs,
    AscendC::LocalTensor<int32_t> &prefix_top_index) {
  //   TokenIdType top_tokens[512];
  // printf("StackWithOutput prefix_top_index:\n");
  // AscendC::DumpTensor(prefix_top_index,0,this->align_beam_width);
  // printf("StackWithOutput prefix_top_probs:\n");
  // AscendC::DumpTensor(prefix_top_probs,0,this->align_beam_width);
  // printf("dump gm top_tokens_gm:\n");
  // AscendC::DumpTensor(top_tokens_gm,0,this->align_beam_width);
  // int32_t token_idx[512];
  int32_t out_token_ids_offset = 0;
  for (int32_t i = 0; i < this->beam_width; i++) {
    LocalTensor<TokenIdType> token_ids_local =
        token_ids_in_que.AllocTensor<TokenIdType>();
    int32_t prefix_top_index_value = prefix_top_index.GetValue(i);
    printf("prefix_top_index_value :%d\n", prefix_top_index_value);
    int32_t true_top_index = prefix_top_index_value;
    int32_t q = true_top_index / this->align_top_k;
    int32_t c = true_top_index % this->align_top_k;
    true_top_index = q * this->top_k + c;
    printf("true_top_index :%d\n", true_top_index);
    printf("q,c:%d,%d\n", q, c);
   
    int32_t token_idx = request_idx * this->beam_width + q;

    int32_t top_tokens_index = request_idx * this->beam_width * this->top_k + true_top_index;
    int32_t top_tokens_gm_value = top_tokens_gm.GetValue(top_tokens_index);
    
    out_token_ids_offset = (request_idx * this->beam_width + i) * (this->sequence_length + 1);
    printf("token idx :%d, top tokens index :%d\n", token_idx, top_tokens_index);
    printf("top_tokens_gm value:%d\n", top_tokens_gm_value);
    printf("offset:%d\n", (request_idx * beam_width + i) * (this->sequence_length + 1)+this->sequence_length);
    printf("out_token_ids_gm value:%d\n", out_token_ids_gm.GetValue((request_idx * beam_width + i) * (this->sequence_length + 1)+this->sequence_length));
    int32_t length_token = AlignUp(this->sequence_length, 32);
    uint32_t offset = token_idx * (this->sequence_length); 
    AlignUpDataCopy(token_ids_local,
                    token_ids_gm[offset],
                    this->sequence_length);
    // printf("token_ids_local:\n");
    // AscendC::DumpTensor(token_ids_local,0,length_token);
    token_ids_in_que.EnQue(token_ids_local);
    LocalTensor<TokenIdType> token_ids_local_tmp =
        token_ids_in_que.DeQue<TokenIdType>();
    LocalTensor<TokenIdType> out_token_ids_local_tmp =
        out_token_ids_out_que.AllocTensor<TokenIdType>();
    int32_t length_align = AlignUp(length_token, 64 / sizeof(int32_t));
    AscendC::DataCopy(out_token_ids_local_tmp, token_ids_local_tmp, length_token);
    out_token_ids_out_que.EnQue(out_token_ids_local_tmp);
    LocalTensor<TokenIdType> out_token_ids_local_tmp_tmp =
        out_token_ids_out_que.DeQue<TokenIdType>();
    AlignUpDataCopyGm(out_token_ids_gm[out_token_ids_offset],
                      out_token_ids_local_tmp_tmp, this->sequence_length);
    out_token_ids_gm.SetValue(
      (request_idx * beam_width + i) * (this->sequence_length + 1) + this->sequence_length,
      top_tokens_gm_value);
    token_ids_in_que.FreeTensor(token_ids_local_tmp);
    out_token_ids_out_que.FreeTensor(out_token_ids_local_tmp_tmp);
  }
}

extern "C" __global__ __aicore__ void
beam_search(GM_ADDR token_ids, GM_ADDR log_probs, GM_ADDR top_tokens,
            GM_ADDR top_probs, GM_ADDR out_token_ids, GM_ADDR workspace,
            GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  BeamSearch<int32_t, float> op;
  op.Init(token_ids, log_probs, top_tokens, top_probs, out_token_ids,
          tiling_data.num_sequences, tiling_data.sequence_length,
          tiling_data.beam_width, tiling_data.top_k, tiling_data.request_num,
          tiling_data.core_num, tiling_data.min_size,
          tiling_data.topkTilingData, tiling_data.topKTilingData1);
  op.Process();
}