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
 * \file pp_mat_mul_with_bias_kernel.h
 * \brief
 */

#ifndef __BATCH_MAT_MUL_V4_WITH_BIAS_KERNEL_H__
#define __BATCH_MAT_MUL_V4_WITH_BIAS_KERNEL_H__

#ifdef __CCE_KT_TEST__
#include "stub_def.h"
#include "stub_fun.h"
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

#include "utils/kernel/common.h"
#include "utils/kernel/hardware.h"
#include "utils/kernel/iterator.h"
#include "utils/kernel/mma.h"
#include "utils/kernel/utils.h"
#include "pp_mat_mul_common.h"

namespace PpMatMulNS {

#ifdef __DAV_C220_CUBE__
template <uint32_t SwizzleDirect, bool TA, bool TB, bool WithBias, typename InDtype, typename OutDtype, typename BiasDtype>
class PpMatmulWithBias {
public:
    __aicore__ explicit PpMatmulWithBias(){};

    __aicore__ FORCE_INLINE void SetArgs(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, const PpMatmulTilingData* tilingData)
    {
        batch_size = tilingData->batch;
        m = tilingData->m;
        k = tilingData->k;
        n = tilingData->n;
        m0 = tilingData->m0;
        k0 = tilingData->k0;
        n0 = tilingData->n0;
        tdim.m = tilingData->mLoop;
        tdim.k = tilingData->kLoop;
        tdim.n = tilingData->nLoop;
        core_loop = tilingData->coreLoop;
        swizzle_cnt = tilingData->swizzlCount;
        en_shuffle_k = tilingData->enShuffleK;

        gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(a));
        gm_b.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(b));
        gm_bias.SetGlobalBuffer(reinterpret_cast<__gm__ BiasDtype *>(bias));
        gm_c.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype *>(c));

        AsdopsBuffer<ArchType::ASCEND_V220> buf;
        uint32_t tile_a_size = RoundUp<CONST_256>(m0 * k0 * sizeof(InDtype));
        uint32_t tile_b_size = RoundUp<CONST_256>(n0 * k0 * sizeof(InDtype));
        l1a_ping_ = buf.GetBuffer<BufferType::ASCEND_CB, InDtype>(0);
        l1b_ping_ = buf.GetBuffer<BufferType::ASCEND_CB, InDtype>(tile_a_size);
        l1_bias_ping_ = buf.GetBuffer<BufferType::ASCEND_CB, BiasDtype>(L1_BIAS_OFFSET);
        l1a_pong_ = buf.GetBuffer<BufferType::ASCEND_CB, InDtype>(tile_a_size + tile_b_size);
        l1b_pong_ = buf.GetBuffer<BufferType::ASCEND_CB, InDtype>(2 * tile_a_size + tile_b_size);
        l1_bias_pong_ = buf.GetBuffer<BufferType::ASCEND_CB, BiasDtype>(L1_BIAS_OFFSET + MAX_BT_SIZE);

        l0a_base = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
        l0b_base = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);
        num_core = AscendC::GetBlockNum();
        core_idx = AscendC::GetBlockIdx();
        ping_flag = 1;
    }

    __aicore__ FORCE_INLINE void GetBlockIdx(uint64_t index, MatCoord &tidx)
    {
        uint64_t in_batch_idx = index % (tdim.m * tdim.n);
        if constexpr (SwizzleDirect == 0) { // Zn
            uint64_t tile_block_loop = (tdim.m + swizzle_cnt - 1) / swizzle_cnt;
            uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * tdim.n);
            uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * tdim.n);

            uint64_t n_row = swizzle_cnt;
            if (tile_block_idx == tile_block_loop - 1) {
                n_row = tdim.m - swizzle_cnt * tile_block_idx;
            }
            tidx.m = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_row;
            tidx.n = in_tile_block_idx / n_row;
            if (tile_block_idx % 2 != 0) {
                tidx.n = tdim.n - tidx.n - 1;
            }
        } else if constexpr (SwizzleDirect == 1) { // Nz
            uint64_t tile_block_loop = (tdim.n + swizzle_cnt - 1) / swizzle_cnt;
            uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * tdim.m);
            uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * tdim.m);

            uint64_t n_col = swizzle_cnt;
            if (tile_block_idx == tile_block_loop - 1) {
                n_col = tdim.n - swizzle_cnt * tile_block_idx;
            }
            tidx.m = in_tile_block_idx / n_col;
            tidx.n = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_col;
            if (tile_block_idx % 2 != 0) {
                tidx.m = tdim.m - tidx.m - 1;
            }
        }
    }

    __aicore__ FORCE_INLINE void Run()
    {
        using InTensor = AscendC::LocalTensor<InDtype>;
        using BiasTensor = AscendC::LocalTensor<BiasDtype>;
        using CopyGmToCbuf = gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::ND, DataFormat::ND>;
        using CopyBiasGmToCbuf = gm_to_l1<ArchType::ASCEND_V220, BiasDtype, DataFormat::ND, DataFormat::ND>;
        using CopyGmToCbufNd2Nz = gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::ND, DataFormat::NZ>;
        using LoadCbufToCa = l1_to_l0_a<ArchType::ASCEND_V220, InDtype, TA, DataFormat::ZN, DataFormat::ZZ>;
        using LoadCbufToCb = l1_to_l0_b<ArchType::ASCEND_V220, InDtype, TB, DataFormat::ZN, DataFormat::NZ>;
        using Mad = mmad<ArchType::ASCEND_V220, InDtype, InDtype, float, false>;
        using CopyCcToGm = l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, OutDtype, float>;
        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID2);
        SET_FLAG(MTE1, MTE2, EVENT_ID3);
        SET_FLAG(MTE1, MTE2, EVENT_ID4);
        SET_FLAG(MTE1, MTE2, EVENT_ID5);
        SET_FLAG(FIX, M, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(M, MTE1, EVENT_ID2);
        SET_FLAG(M, MTE1, EVENT_ID3);

        for (uint64_t loop_idx = core_idx; loop_idx < core_loop; loop_idx += num_core) {
            uint64_t batch_idx = loop_idx / tdim.n / tdim.m;
            MatCoord tidx{0};
            GetBlockIdx(loop_idx, tidx);
            uint64_t offset_a = 0, offset_b = 0, offset_a_next = 0, offset_b_next = 0;
            uint64_t offset_c = batch_idx * m * n + tidx.m * m0 * n + tidx.n * n0;
            uint64_t m_actual = (tidx.m == (tdim.m - 1)) ? (m - tidx.m * m0) : m0;
            uint64_t n_actual = (tidx.n == (tdim.n - 1)) ? (n - tidx.n * n0) : n0;
            uint64_t m_round = RoundUp<CONST_16>(m_actual);
            uint64_t n_round = RoundUp<CONST_16>(n_actual);
            uint64_t mn_max = m_round > n_round ? m_round : n_round;
            uint64_t k_part_len = L0_PINGPONG_BUFFER_SIZE / mn_max / CONST_16 * CONST_16;
            uint64_t shuffle_k = en_shuffle_k ? (core_idx % tdim.k) : 0;
            if (TB) {
                offset_b = batch_idx * k * n + tidx.n * n0 * k + shuffle_k * k0;
            } else {
                offset_b = batch_idx * k * n + shuffle_k * k0 * n + tidx.n * n0;
            }
            if (TA) {
                offset_a = batch_idx * m * k + shuffle_k * k0 * m + tidx.m * m0;
            } else {
                offset_a = batch_idx * m * k + tidx.m * m0 * k + shuffle_k * k0;
            }

            uint64_t k_actual = (shuffle_k == tdim.k - 1) ? k - shuffle_k * k0 : k0;
            uint64_t k_round = (k_actual + CONST_16 - 1) / CONST_16 * CONST_16;

            InTensor l1_buf_a = ping_flag ? l1a_ping_ : l1a_pong_;
            InTensor l1_buf_b = ping_flag ? l1b_ping_ : l1b_pong_;
            BiasTensor l1_bias = ping_flag ? l1_bias_ping_ : l1_bias_pong_;
            event_t event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            if (loop_idx == core_idx) {
                // Load Bias from gm to bt.
                if constexpr (WithBias) {
                    uint64_t offset_bias = batch_idx * n + tidx.n * n0;
                    WAIT_FLAG(MTE1, MTE2, event_id + 4);
                    CopyBiasGmToCbuf(l1_bias,              // dst
                                     gm_bias[offset_bias], // src
                                     1,                    // nTileActual
                                     CONST_16,             // nTileCeil
                                     1,                    // nVal
                                     n_actual,             // kTileActual
                                     n_round,              // kTileCeil
                                     n);                   // dVal
                    SET_FLAG(MTE2, MTE1, event_id + 4);
                }
                // Load matrix B from gm to l1.
                WAIT_FLAG(MTE1, MTE2, event_id + 2);
                if (TB) {
                    CopyGmToCbufNd2Nz(l1_buf_b,       // dst
                                      gm_b[offset_b], // src
                                      n_actual,       // nTileActual
                                      n_round,        // nTileCeil
                                      n,              // nVal
                                      k_actual,       // dTileActual
                                      k_round,        // dTileCeil
                                      k);             // dVal
                } else {
                    CopyGmToCbufNd2Nz(l1_buf_b,       // dst
                                      gm_b[offset_b], // src
                                      k_actual,       // nTileActual
                                      k_round,        // nTileCeil
                                      k,              // nVal
                                      n_actual,       // dTileActual
                                      n_round,        // dTileCeil
                                      n);             // dVal
                }
                SET_FLAG(MTE2, MTE1, event_id + 2);

                // Load matrix A from gm to l1.
                WAIT_FLAG(MTE1, MTE2, event_id); // write_l1_a
                if ((m == 1) || (m_actual == 1 && !TA)) {
                    CopyGmToCbuf(l1_buf_a,       // dst
                                 gm_a[offset_a], // src
                                 1,              // nTileActual
                                 16,             // nTileCeil
                                 1,              // nVal
                                 k_actual,       // kTileActual
                                 k_round,        // kTileCeil
                                 k);             // dVal
                } else {
                    if (TA) {
                        CopyGmToCbufNd2Nz(l1_buf_a,       // dst
                                          gm_a[offset_a], // src
                                          k_actual,       // nTileActual
                                          k_round,        // nTileCeil
                                          k,              // nVal
                                          m_actual,       // dTileActual
                                          m_round,        // dTileCeil
                                          m);             // dVal
                    } else {
                        CopyGmToCbufNd2Nz(l1_buf_a,       // dst
                                          gm_a[offset_a], // src
                                          m_actual,       // nTileActual
                                          m_round,        // nTileCeil
                                          m,              // nVal
                                          k_actual,       // dTileActual
                                          k_round,        // dTileCeil
                                          k);             // dVal
                    }
                }
                SET_FLAG(MTE2, MTE1, event_id); // read_l1_a
            }

            for (tidx.k = 0; tidx.k < tdim.k; ++tidx.k) {
                shuffle_k = en_shuffle_k ? (tidx.k + core_idx) % tdim.k : tidx.k;
                uint64_t k_actual = (shuffle_k == (tdim.k - 1)) ? (k - shuffle_k * k0) : k0;
                uint64_t k_round = (k_actual + CONST_16 - 1) / CONST_16 * CONST_16;
                fdim.k = (k_actual + k_part_len - 1) / k_part_len;

                InTensor l1_buf_a = ping_flag ? l1a_ping_ : l1a_pong_;
                InTensor l1_buf_b = ping_flag ? l1b_ping_ : l1b_pong_;
                BiasTensor l1_bias = ping_flag ? l1_bias_ping_ : l1_bias_pong_;
                auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

                if (tidx.k < tdim.k - 1) {
                    uint64_t shuffle_k_next = en_shuffle_k ? (core_idx + tidx.k + 1) % tdim.k : (tidx.k + 1);
                    if (TA) {
                        offset_a_next = batch_idx * m * k + shuffle_k_next * k0 * m + tidx.m * m0;
                    } else {
                        offset_a_next = batch_idx * m * k + tidx.m * m0 * k + shuffle_k_next * k0;
                    }

                    if (TB) {
                        offset_b_next = batch_idx * k * n + tidx.n * n0 * k + shuffle_k_next * k0;
                    } else {
                        offset_b_next = batch_idx * k * n + shuffle_k_next * k0 * n + tidx.n * n0;
                    }

                    uint64_t k_actual_next = (shuffle_k_next == (tdim.k - 1)) ? (k - shuffle_k_next * k0) : k0;
                    uint64_t k_round_next = (k_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;

                    InTensor l1_buf_a_next = (1 - ping_flag) ? l1a_ping_ : l1a_pong_;
                    InTensor l1_buf_b_next = (1 - ping_flag) ? l1b_ping_ : l1b_pong_;
                    event_t event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                    // Load matrix B from gm to L1
                    WAIT_FLAG(MTE1, MTE2, event_id_next + 2);
                    if (TB) {
                        CopyGmToCbufNd2Nz(l1_buf_b_next,       // dst
                                          gm_b[offset_b_next], // src
                                          n_actual,            // nTileActual
                                          n_round,             // nTileCeil
                                          n,                   // nVal
                                          k_actual_next,       // dTileActual
                                          k_round_next,        // dTileCeil
                                          k);                  // dVal
                    } else {
                        CopyGmToCbufNd2Nz(l1_buf_b_next,       // dst
                                          gm_b[offset_b_next], // src
                                          k_actual_next,       // nTileActual
                                          k_round_next,        // nTileCeil
                                          k,                   // nVal
                                          n_actual,            // dTileActual
                                          n_round,             // dTileCeil
                                          n);                  // dVal
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next + 2);

                    // Load matrix A from gm to L1
                    WAIT_FLAG(MTE1, MTE2, event_id_next);
                    if ((m == 1) || (m_actual == 1 && !TA)) {
                        CopyGmToCbuf(l1_buf_a_next,       // dst
                                     gm_a[offset_a_next], // src
                                     m_actual,            // nTileActual
                                     m_round,             // nTileCeil
                                     m,                   // nVal
                                     k_actual_next,       // kTileActual
                                     k_round_next,        // kTileCeil
                                     k);                  // dVal
                    } else {
                        if (TA) {
                            CopyGmToCbufNd2Nz(l1_buf_a_next,       // dst
                                              gm_a[offset_a_next], // src
                                              k_actual_next,       // nTileActual
                                              k_round_next,        // nTileCeil
                                              k,                   // nVal
                                              m_actual,            // dTileActual
                                              m_round,             // dTileCeil
                                              m);                  // dVal
                        } else {
                            CopyGmToCbufNd2Nz(l1_buf_a_next,       // dst
                                              gm_a[offset_a_next], // src
                                              m_actual,            // nTileActual
                                              m_round,             // nTileCeil
                                              m,                   // nVal
                                              k_actual_next,       // dTileActual
                                              k_round_next,        // dTileCeil
                                              k);                  // dVal
                        }
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next);
                }

                if (tidx.k == tdim.k - 1 && loop_idx + num_core < core_loop) {
                    uint64_t b_idx_next = (loop_idx + num_core) / tdim.n / tdim.m;
                    MatCoord tidx{0};
                    GetBlockIdx(loop_idx + num_core, tidx);
                    uint64_t shuffle_k_next = en_shuffle_k ? (core_idx % tdim.k) : 0;
                    uint64_t m_actual_next = (tidx.m == (tdim.m - 1)) ? (m - tidx.m * m0) : m0;
                    uint64_t n_actual_next = (tidx.n == (tdim.n - 1)) ? (n - tidx.n * n0) : n0;
                    uint64_t m_round_next = (m_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
                    uint64_t n_round_next = (n_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
                    uint64_t k_actual_next = (shuffle_k_next == (tdim.k - 1)) ? (k - shuffle_k_next * k0) : k0;
                    uint64_t k_round_next = (k_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
                    if (TA) {
                        offset_a_next = b_idx_next * m * k + shuffle_k_next * k0 * m + tidx.m * m0;
                    } else {
                        offset_a_next = b_idx_next * m * k + tidx.m * m0 * k + shuffle_k_next * k0;
                    }
                    if (TB) {
                        offset_b_next = b_idx_next * k * n + tidx.n * n0 * k + shuffle_k_next * k0;
                    } else {
                        offset_b_next = b_idx_next * k * n + shuffle_k_next * k0 * n + tidx.n * n0;
                    }

                    InTensor l1_buf_a_next = (1 - ping_flag) ? l1a_ping_ : l1a_pong_;
                    InTensor l1_buf_b_next = (1 - ping_flag) ? l1b_ping_ : l1b_pong_;
                    BiasTensor l1_bias_next = (1 - ping_flag) ? l1_bias_ping_ : l1_bias_pong_;
                    event_t event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                    // Load Bias from gm to bt.
                    if constexpr (WithBias) {
                        uint64_t offset_bias_next = b_idx_next * n + tidx.n * n0;
                        WAIT_FLAG(MTE1, MTE2, event_id_next + 4);
                        CopyBiasGmToCbuf(l1_bias_next,              // dst
                                         gm_bias[offset_bias_next], // src
                                         1,                         // nTileActual
                                         CONST_16,                  // nTileCeil
                                         1,                         // nVal
                                         n_actual_next,             // kTileActual
                                         n_round_next,              // kTileCeil
                                         n);                        // dVal
                        SET_FLAG(MTE2, MTE1, event_id_next + 4);
                    }
                    // Load matrix B from gm to l1.
                    WAIT_FLAG(MTE1, MTE2, event_id_next + 2);
                    if (TB) {
                        CopyGmToCbufNd2Nz(l1_buf_b_next,       // dst
                                          gm_b[offset_b_next], // src
                                          n_actual_next,       // nTileActual
                                          n_round_next,        // nTileCeil
                                          n,                   // nVal
                                          k_actual_next,       // dTileActual
                                          k_round_next,        // dTileCeil
                                          k);                  // dVal
                    } else {
                        CopyGmToCbufNd2Nz(l1_buf_b_next,       // dst
                                          gm_b[offset_b_next], // src
                                          k_actual_next,       // nTileActual
                                          k_round_next,        // nTileCeil
                                          k,                   // nVal
                                          n_actual_next,       // dTileActual
                                          n_round_next,        // dTileCeil
                                          n);                  // dVal
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next + 2);

                    // Load matrix A from gm to l1.
                    WAIT_FLAG(MTE1, MTE2, event_id_next);
                    if (m == 1 || m_actual_next == 1 && !TA) {
                        CopyGmToCbuf(l1_buf_a_next,       // dst
                                     gm_a[offset_a_next], // src
                                     m_actual_next,       // nTileActual
                                     m_round_next,        // nTileCeil
                                     m,                   // nVal
                                     k_actual_next,       // kTileActual
                                     k_round_next,        // kTileCeil
                                     k);                  // dVal
                    } else {
                        if (TA) {
                            CopyGmToCbufNd2Nz(l1_buf_a_next,       // dst
                                              gm_a[offset_a_next], // src
                                              k_actual_next,       // nTileActual
                                              k_round_next,        // nTileCeil
                                              k,                   // nVal
                                              m_actual_next,       // dTileActual
                                              m_round_next,        // dTileCeil
                                              m);                  // dVal
                        } else {
                            CopyGmToCbufNd2Nz(l1_buf_a_next,       // dst
                                              gm_a[offset_a_next], // src
                                              m_actual_next,       // nTileActual
                                              m_round_next,        // nTileCeil
                                              m,                   // nVal
                                              k_actual_next,       // dTileActual
                                              k_round_next,        // dTileCeil
                                              k);                  // dVal
                        }
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next);
                }

                MatCoord fidx{0};
                for (fidx.k = 0; fidx.k < fdim.k; ++fidx.k) {
                    uint32_t k0_round = (fidx.k < fdim.k - 1) ? k_part_len : k_round - fidx.k * k_part_len;
                    uint32_t k0_actual = (fidx.k < fdim.k - 1) ? k_part_len : k_actual - fidx.k * k_part_len;

                    auto mte1_mad_ping_flag = 1 - fidx.k % 2;
                    auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                    auto l0a_buf = l0a_base[(fidx.k % 2) * L0_PINGPONG_BUFFER_SIZE];
                    auto l0b_buf = l0b_base[(fidx.k % 2) * L0_PINGPONG_BUFFER_SIZE];

                    WAIT_FLAG(M, MTE1, mte1_mad_event_id);
                    // Load matrix B from L1 to L0B
                    if (fidx.k == 0) {
                        WAIT_FLAG(MTE2, MTE1, event_id + 2);
                    }
                    if (TB) {
                        LoadCbufToCb(l0b_buf,                                 // l0Tensor
                                     l1_buf_b[fidx.k * k_part_len * n_round], // l1Tensor
                                     n_round,                                 // nTileCeil
                                     k0_round,                                // kPartCeil
                                     1,                                       // nSrcStride
                                     n_round / CONST_16,                      // kSrcStride
                                     1,                                       // nDstStride
                                     k0_round / CONST_16);                    // kDstStride
                    } else {
                        LoadCbufToCb(l0b_buf,                                  // l0Tensor
                                     l1_buf_b[fidx.k * k_part_len * CONST_16], // l1Tensor
                                     n_round,                                  // nTileCeil
                                     k0_round,                                 // kPartCeil
                                     k_round / CONST_16,                       // nSrcStride
                                     1,                                        // kSrcStride
                                     1,                                        // nDstStride
                                     n_round / CONST_16);                      // kDstStride
                    }
                    if (fidx.k == fdim.k - 1) {
                        SET_FLAG(MTE1, MTE2, event_id + 2);
                    }

                    // Load matrix A from L1 to L0A
                    if (fidx.k == 0) {
                        WAIT_FLAG(MTE2, MTE1, event_id);
                    }
                    if ((m == 1) || (m_actual == 1 && !TA)) {
                        l1_to_l0_a<ArchType::ASCEND_V220, InDtype, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0a_buf,                                // dst
                            l1_buf_a[fidx.k * k_part_len],          // src
                            0,                                      // mTileCeil
                            (k0_round + CONST_256 - 1) / CONST_256, // kPartCeil
                            0,                                      // mSrcStride
                            1,                                      // kSrcStride
                            0,                                      // mDstStride
                            0);                                     // kDstStride
                    } else {
                        if (TA) {
                            LoadCbufToCa(l0a_buf,                                  // l0Tensor
                                         l1_buf_a[fidx.k * k_part_len * CONST_16], // l1Tensor
                                         m_round,                                  // mTileCeil
                                         k0_round,                                 // kPartCeil
                                         k_round / CONST_16,                       // mSrcStride
                                         1,                                        // kSrcStride
                                         k0_round / CONST_16,                      // mDstStride
                                         1);                                       // kDstStride
                        } else {
                            LoadCbufToCa(l0a_buf,                                 // l0Tensor
                                         l1_buf_a[fidx.k * k_part_len * m_round], // l1Tensor
                                         m_round,                                 // mTileCeil
                                         k0_round,                                // kPartCeil
                                         1,                                       // mSrcStride
                                         m_round / CONST_16,                      // kSrcStride
                                         k0_round / CONST_16,                     // mDstStride
                                         1);                                      // kDstStride
                        }
                    }
                    if (fidx.k == fdim.k - 1) {
                        SET_FLAG(MTE1, MTE2, event_id);
                    }

                    SET_FLAG(MTE1, M, mte1_mad_event_id);
                    WAIT_FLAG(MTE1, M, mte1_mad_event_id);

                    bool init_c = (tidx.k == 0 && fidx.k == 0);
                    if (init_c) {
                        WAIT_FLAG(FIX, M, EVENT_ID0);
                    }
                    bool sp_mode = (m != 1 && m_actual == 1 && TA);
                    if (init_c) {
                        if constexpr (WithBias) {
                            WAIT_FLAG(MTE2, MTE1, event_id + 4);
                            l1_to_bt<ArchType::ASCEND_V220, BiasDtype>(
                                bt_addr,                                   // dst
                                l1_bias,                                   // src
                                0,                                         // convControl
                                1,                                         // nBurst
                                CeilDiv<64>(n_actual * sizeof(BiasDtype)), // lenBurst
                                0,                                         // srcGap
                                0);                                        // dstGap
                            SET_FLAG(MTE1, MTE2, event_id + 4);
                            SET_FLAG(MTE1, M, EVENT_ID7);
                            WAIT_FLAG(MTE1, M, EVENT_ID7);
                            Mad(l0c_buf,                       // c
                                l0a_buf,                       // a
                                l0b_buf,                       // b
                                bt_addr,                       // bt
                                sp_mode ? CONST_16 : m_actual, // mTileActual
                                n_actual,                      // nTileActual
                                k0_actual,                     // kTileActual
                                init_c);                       // initC
                        }
                    } else {
                        Mad(l0c_buf,                       // c
                            l0a_buf,                       // a
                            l0b_buf,                       // b
                            sp_mode ? CONST_16 : m_actual, // mTileActual
                            n_actual,                      // nTileActual
                            k0_actual,                     // kTileActual
                            init_c);                       // initC
                    }

                    PIPE_BARRIER(M);
                    SET_FLAG(M, MTE1, mte1_mad_event_id);
                }

                ping_flag = 1 - ping_flag;
            }

            SET_FLAG(M, FIX, EVENT_ID0);
            WAIT_FLAG(M, FIX, EVENT_ID0);

            // copy from L0C to gm
            CopyCcToGm(gm_c[offset_c], // dst
                       l0c_buf,        // src
                       m_actual,       // mTileActual
                       n_actual,       // nTileActual
                       m_round,        // mTileCeil
                       n);             // nActual
            SET_FLAG(FIX, M, EVENT_ID0);
        }

        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(M, MTE1, EVENT_ID2);
        WAIT_FLAG(M, MTE1, EVENT_ID3);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID4);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID5);
        WAIT_FLAG(FIX, M, EVENT_ID0);
        PIPE_BARRIER(ALL);
    }

private:
    AscendC::GlobalTensor<InDtype> gm_a;
    AscendC::GlobalTensor<InDtype> gm_b;
    AscendC::GlobalTensor<BiasDtype> gm_bias;
    AscendC::GlobalTensor<OutDtype> gm_c;
    AscendC::LocalTensor<InDtype> l1a_ping_;
    AscendC::LocalTensor<InDtype> l1a_pong_;
    AscendC::LocalTensor<InDtype> l1b_ping_;
    AscendC::LocalTensor<InDtype> l1b_pong_;
    AscendC::LocalTensor<BiasDtype> l1_bias_ping_;
    AscendC::LocalTensor<BiasDtype> l1_bias_pong_;

    AscendC::LocalTensor<InDtype> l0a_base;
    AscendC::LocalTensor<InDtype> l0b_base;
    AscendC::LocalTensor<float> l0c_buf;

    uint32_t num_core{0};
    uint32_t batch_size{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
    MatCoord tdim{0};
    MatCoord fdim{0};
    uint32_t core_loop{0};
    uint32_t swizzle_cnt{1};
    uint32_t core_idx{0};
    uint32_t ping_flag{0};
    uint32_t en_shuffle_k{0};
    uint64_t bt_addr{0};
};

#endif
}
#endif