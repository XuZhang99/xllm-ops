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
 * \file pp_mat_mul_bf16_nd_nz_nd_kernel.h
 * \brief
 */

#ifndef __BATCH_MAT_MUL_V4_ND_NZ_ND_BF16_KERNEL_H__
#define __BATCH_MAT_MUL_V4_ND_NZ_ND_BF16_KERNEL_H__

#ifdef __CCE_KT_TEST__
#include "stub_def.h"
#include "stub_fun.h"
using __bf16 = bfloat16_t;
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
template <uint32_t SwizzleDirect, bool TA, bool TB, typename InDtype = __bf16, typename OutDtype = __bf16>
class PpMatmulBF16NDNZND {
public:
    __aicore__ explicit PpMatmulBF16NDNZND(){};

    __aicore__ FORCE_INLINE void SetArgs(GM_ADDR a, GM_ADDR b,
                                             GM_ADDR c, const PpMatmulTilingData* tilingData)
    {
        gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(a));
        gm_b.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(b));
        gm_c.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype *>(c));
        batch_size = tilingData->batch;
        m = tilingData->m;
        k = tilingData->k;
        n = tilingData->n;
        m0 = tilingData->m0;
        k0 = tilingData->k0;
        n0 = tilingData->n0;
        m_loop = tilingData->mLoop;
        k_loop = tilingData->kLoop;
        n_loop = tilingData->nLoop;
        core_loop = tilingData->coreLoop;
        swizzle_cnt = tilingData->swizzlCount;

        l1_base_a = buf.GetBuffer<BufferType::ASCEND_CB, InDtype>(0);
        l1_base_b = buf.GetBuffer<BufferType::ASCEND_CB, InDtype>(RoundUp256(m0 * k0 * sizeof(InDtype)));

        num_core = AscendC::GetBlockNum();
        core_idx = AscendC::GetBlockIdx();
        ping_flag = 1;
    }

    __aicore__ FORCE_INLINE void GetBlockIdx(uint64_t index, uint64_t &m_idx, uint64_t &n_idx)
    {
        uint64_t in_batch_idx = index % (m_loop * n_loop);
        if constexpr (SwizzleDirect == 0) { // Zn
            uint64_t tile_block_loop = (m_loop + swizzle_cnt - 1) / swizzle_cnt;
            uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * n_loop);
            uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * n_loop);

            uint64_t n_row = swizzle_cnt;
            if (tile_block_idx == tile_block_loop - 1) {
                n_row = m_loop - swizzle_cnt * tile_block_idx;
            }
            m_idx = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_row;
            n_idx = in_tile_block_idx / n_row;
            if (tile_block_idx % 2 != 0) {
                n_idx = n_loop - n_idx - 1;
            }
        } else if constexpr (SwizzleDirect == 1) { // Nz
            uint64_t tile_block_loop = (n_loop + swizzle_cnt - 1) / swizzle_cnt;
            uint64_t tile_block_idx = in_batch_idx / (swizzle_cnt * m_loop);
            uint64_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * m_loop);

            uint64_t n_col = swizzle_cnt;
            if (tile_block_idx == tile_block_loop - 1) {
                n_col = n_loop - swizzle_cnt * tile_block_idx;
            }
            m_idx = in_tile_block_idx / n_col;
            n_idx = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_col;
            if (tile_block_idx % 2 != 0) {
                m_idx = m_loop - m_idx - 1;
            }
        }
    }

    __aicore__ FORCE_INLINE void run()
    {
        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID2);
        SET_FLAG(MTE1, MTE2, EVENT_ID3);
        SET_FLAG(FIX, M, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        for (uint64_t loop_idx = core_idx; loop_idx < core_loop; loop_idx += num_core) {
            uint64_t batch_idx = loop_idx / n_loop / m_loop;
            uint64_t m_idx = 0, n_idx = 0;
            GetBlockIdx(loop_idx, m_idx, n_idx);
            uint64_t offset_a = 0, offset_b = 0, offset_a_next = 0, offset_b_next = 0;
            uint64_t offset_c = batch_idx * m * n + m_idx * m0 * n + n_idx * n0;
            uint64_t m_actual = (m_idx == (m_loop - 1)) ? (m - m_idx * m0) : m0;
            uint64_t n_actual = (n_idx == (n_loop - 1)) ? (n - n_idx * n0) : n0;
            uint64_t m_round = RoundUp16(m_actual);
            uint64_t n_round = RoundUp16(n_actual);
            uint64_t mn_max = m_round > n_round ? m_round : n_round;
            uint64_t k_part_len = L0_PINGPONG_BUFFER_SIZE / mn_max / CONST_16 * CONST_16;
            uint64_t shuffle_k = core_idx % k_loop;
            if (TA) {
                offset_a = batch_idx * m * k + shuffle_k * k0 * m + m_idx * m0;
            } else {
                offset_a = batch_idx * m * k + m_idx * m0 * k + shuffle_k * k0;
            }

            if (TB) {
                offset_b =
                    batch_idx * RoundUp16(k) * RoundUp16(n) + n_idx * n0 * CONST_16 + shuffle_k * k0 * RoundUp16(n);
            } else {
                offset_b =
                    batch_idx * RoundUp16(k) * RoundUp16(n) + shuffle_k * k0 * CONST_16 + n_idx * n0 * RoundUp16(k);
            }

            uint64_t k_actual = (shuffle_k == k_loop - 1) ? k - shuffle_k * k0 : k0;
            uint64_t k_round = RoundUp16(k_actual);

            AscendC::LocalTensor<InDtype> l1_buf_a = ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_SIZE];
            AscendC::LocalTensor<InDtype> l1_buf_b = ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_SIZE];
            AscendC::LocalTensor<InDtype> l0a_buf = ping_flag ? l0a_base : l0a_base[L0_PINGPONG_BUFFER_SIZE];
            AscendC::LocalTensor<InDtype> l0b_buf = ping_flag ? l0b_base : l0b_base[L0_PINGPONG_BUFFER_SIZE];
            event_t event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            if (loop_idx == core_idx) {
                WAIT_FLAG(MTE1, MTE2, event_id);
                // *** load matrix A to L1
                if ((m == 1) || (m_actual == 1 && !TA)) {
                    gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::ND, DataFormat::ND>(l1_buf_a,       // dst
                                                                                             gm_a[offset_a], // src
                                                                                             1, 0, 0,
                                                                                             k_round, // lenBurst
                                                                                             0, 0);
                } else {
                    if (TA) {
                        gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::ND, DataFormat::NZ>(
                            l1_buf_a, gm_a[offset_a], k_actual, k_round, k, m_actual, m_round, m);
                    } else {
                        gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::ND, DataFormat::NZ>(
                            l1_buf_a, gm_a[offset_a], m_actual, m_round, m, k_actual, k_round, k);
                    }
                }
                SET_FLAG(MTE2, MTE1, event_id);
                // *** load matrix B to L1
                WAIT_FLAG(MTE1, MTE2, event_id + 2);
                if (TB) {
                    gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::NZ, DataFormat::NZ>(
                        l1_buf_b, gm_b[offset_b], n_actual, n_round, RoundUp16(n), k_actual, k_round, RoundUp16(k));
                } else {
                    gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::NZ, DataFormat::NZ>(
                        l1_buf_b, gm_b[offset_b], k_actual, k_round, RoundUp16(k), n_actual, n_round, RoundUp16(n));
                }
                SET_FLAG(MTE2, MTE1, event_id + 2);
            }

            for (uint64_t k_idx = 0; k_idx < k_loop; k_idx++) {
                shuffle_k = (k_idx + core_idx) % k_loop;
                uint64_t k_actual = (shuffle_k == (k_loop - 1)) ? (k - shuffle_k * k0) : k0;
                uint64_t k_round = RoundUp16(k_actual);
                uint64_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

                AscendC::LocalTensor<InDtype> l1_buf_a = ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_SIZE];
                AscendC::LocalTensor<InDtype> l1_buf_b = ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_SIZE];
                auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

                if (k_idx < k_loop - 1) {
                    uint64_t shuffle_k_next = (core_idx + k_idx + 1) % k_loop;
                    if (TA) {
                        offset_a_next = batch_idx * m * k + shuffle_k_next * k0 * m + m_idx * m0;
                    } else {
                        offset_a_next = batch_idx * m * k + m_idx * m0 * k + shuffle_k_next * k0;
                    }

                    if (TB) {
                        offset_b_next = batch_idx * RoundUp16(k) * RoundUp16(n) + n_idx * n0 * CONST_16 +
                                        shuffle_k_next * k0 * RoundUp16(n);
                    } else {
                        offset_b_next = batch_idx * RoundUp16(k) * RoundUp16(n) + n_idx * n0 * RoundUp16(k) +
                                        shuffle_k_next * k0 * CONST_16;
                    }

                    uint64_t k_actual_next = (shuffle_k_next == (k_loop - 1)) ? (k - shuffle_k_next * k0) : k0;
                    uint64_t k_round_next = RoundUp16(k_actual_next);

                    AscendC::LocalTensor<InDtype> l1_buf_a_next =
                        (1 - ping_flag) ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_SIZE];
                    AscendC::LocalTensor<InDtype> l1_buf_b_next =
                        (1 - ping_flag) ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_SIZE];
                    event_t event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                    WAIT_FLAG(MTE1, MTE2, event_id_next);
                    // *** load matrix A to L1
                    if ((m == 1) || (m_actual == 1 && !TA)) {
                        gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::ND, DataFormat::ND>(
                            l1_buf_a_next, gm_a[offset_a_next], 1, RoundUp16(1), 1, k_round_next,
                            RoundUp16(k_round_next), k_round_next);
                    } else {
                        if (TA) {
                            gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::ND, DataFormat::NZ>(
                                l1_buf_a_next, gm_a[offset_a_next], k_actual_next, k_round_next, k, m_actual, m_round,
                                m);
                        } else {
                            gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::ND, DataFormat::NZ>(
                                l1_buf_a_next, gm_a[offset_a_next], m_actual, m_round, m, k_actual_next, k_round_next,
                                k);
                        }
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next);

                    // *** load matrix B to L1
                    WAIT_FLAG(MTE1, MTE2, event_id_next + 2);
                    if (TB) {
                        gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::NZ, DataFormat::NZ>(
                            l1_buf_b_next, gm_b[offset_b_next], n_actual, n_round, RoundUp16(n), k_actual_next,
                            k_round_next, RoundUp16(k));
                    } else {
                        gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::NZ, DataFormat::NZ>(
                            l1_buf_b_next, gm_b[offset_b_next], k_actual_next, k_round_next, RoundUp16(k), k_actual,
                            n_round, RoundUp16(n));
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next + 2);
                }

                if (k_idx == k_loop - 1 && loop_idx + num_core < core_loop) {
                    uint64_t b_idx_next = (loop_idx + num_core) / n_loop / m_loop;
                    uint64_t m_idx_next = 0, n_idx_next = 0;
                    GetBlockIdx(loop_idx + num_core, m_idx_next, n_idx_next);
                    uint64_t shuffle_k_next = core_idx % k_loop;
                    uint64_t m_actual_next = (m_idx_next == (m_loop - 1)) ? (m - m_idx_next * m0) : m0;
                    uint64_t n_actual_next = (n_idx_next == (n_loop - 1)) ? (n - n_idx_next * n0) : n0;
                    uint64_t m_round_next = (m_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
                    uint64_t n_round_next = (n_actual_next + CONST_16 - 1) / CONST_16 * CONST_16;
                    uint64_t k_actual_next = (shuffle_k_next == (k_loop - 1)) ? (k - shuffle_k_next * k0) : k0;
                    uint64_t k_round_next = RoundUp16(k_actual_next);
                    if (TA) {
                        offset_a_next = b_idx_next * m * k + shuffle_k_next * k0 * m + m_idx_next * m0;
                    } else {
                        offset_a_next = b_idx_next * m * k + m_idx_next * m0 * k + shuffle_k_next * k0;
                    }
                    if (TB) {
                        offset_b_next = b_idx_next * RoundUp16(k) * RoundUp16(n) + n_idx_next * n0 * CONST_16 +
                                        shuffle_k_next * k0 * RoundUp16(n);
                    } else {
                        offset_b_next = b_idx_next * RoundUp16(k) * RoundUp16(n) + shuffle_k_next * k0 * CONST_16 +
                                        n_idx_next * n0 * RoundUp16(k);
                    }

                    AscendC::LocalTensor<InDtype> l1_buf_a_next =
                        (1 - ping_flag) ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_SIZE];
                    AscendC::LocalTensor<InDtype> l1_buf_b_next =
                        (1 - ping_flag) ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_SIZE];
                    event_t event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                    WAIT_FLAG(MTE1, MTE2, event_id_next);
                    // *** load matrix A to L1
                    if (m == 1 || m_actual_next == 1 && !TA) {
                        gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::ND, DataFormat::ND>(
                            l1_buf_a_next, gm_a[offset_a_next], 1, RoundUp16(1), 1, k_round_next,
                            RoundUp16(k_round_next), k_round_next);
                    } else {
                        if (TA) {
                            gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::ND, DataFormat::NZ>(
                                l1_buf_a_next, gm_a[offset_a_next], k_actual_next, k_round_next, k, m_actual_next,
                                m_round_next, m);
                        } else {
                            gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::ND, DataFormat::NZ>(
                                l1_buf_a_next, gm_a[offset_a_next], m_actual_next, m_round_next, m, k_actual_next,
                                k_round_next, k);
                        }
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next);

                    // *** load matrix B to L1
                    WAIT_FLAG(MTE1, MTE2, event_id_next + 2);
                    if (TB) {
                        gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::NZ, DataFormat::NZ>(
                            l1_buf_b_next, gm_b[offset_b_next], n_actual_next, n_round_next, RoundUp16(n),
                            k_actual_next, k_round_next, RoundUp16(k));
                    } else {
                        gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::NZ, DataFormat::NZ>(
                            l1_buf_b_next, gm_b[offset_b_next], k_actual_next, k_round_next, RoundUp16(k),
                            n_actual_next, n_round_next, RoundUp16(n));
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next + 2);
                }

                for (uint32_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                    uint32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                    uint32_t k0_actual =
                        (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                    auto mte1_mad_ping_flag = 1 - k_part_idx % 2;
                    auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                    AscendC::LocalTensor<InDtype> l0a_buf = l0a_base[(k_part_idx % 2) * L0_PINGPONG_BUFFER_SIZE];
                    AscendC::LocalTensor<InDtype> l0b_buf = l0b_base[(k_part_idx % 2) * L0_PINGPONG_BUFFER_SIZE];

                    // *** load matrix A from L1 to L0A
                    if (k_part_idx == 0) {
                        WAIT_FLAG(MTE2, MTE1, event_id);
                    }
                    WAIT_FLAG(M, MTE1, mte1_mad_event_id);
                    if ((m == 1) || (m_actual == 1 && !TA)) {
                        l1_to_l0_a<ArchType::ASCEND_V220, InDtype, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0a_buf, l1_buf_a[k_part_idx * k_part_len], 0,
                            (k0_round + CONST_256 - 1) / CONST_256, // repeat
                            0,
                            1, // srcStride
                            0,
                            0 // dstStride
                        );
                    } else {
                        if (TA) {
                            l1_to_l0_a<ArchType::ASCEND_V220, InDtype, true, DataFormat::ZN, DataFormat::ZZ>(
                                l0a_buf, l1_buf_a[k_part_idx * k_part_len * CONST_16], m_round,
                                k0_round, // repeat
                                k_round / CONST_16,
                                1, // srcStride
                                k0_round / CONST_16,
                                1 // dstStride
                            );
                        } else {
                            l1_to_l0_a<ArchType::ASCEND_V220, InDtype, false, DataFormat::ZN, DataFormat::ZZ>(
                                l0a_buf, l1_buf_a[k_part_idx * k_part_len * m_round], m_round,
                                k0_round, // repeat
                                1,
                                m_round / CONST_16, // srcStride
                                k0_round / CONST_16,
                                1 // dstStride
                            );
                        }
                    }
                    if (k_part_idx == k_part_loop - 1) {
                        SET_FLAG(MTE1, MTE2, event_id);
                    }

                    // *** load matrix B from L1 to L0B
                    if (k_part_idx == 0) {
                        WAIT_FLAG(MTE2, MTE1, event_id + 2);
                    }
                    if (TB) {
                        l1_to_l0_b<ArchType::ASCEND_V220, InDtype, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0b_buf, l1_buf_b[k_part_idx * k_part_len * n_round], 0,
                            k0_round * n_round / CONST_256, // repeat
                            0,
                            1, // srcStride
                            0,
                            0 // dstStride
                        );
                    } else {
                        l1_to_l0_b<ArchType::ASCEND_V220, InDtype, false, DataFormat::ZN, DataFormat::NZ>(
                            l0b_buf, l1_buf_b[k_part_idx * k_part_len * CONST_16], n_round,
                            k0_round, // repeat
                            k_round / CONST_16,
                            1, // srcStride
                            1,
                            n_round / CONST_16 // dstStride
                        );
                    }
                    if (k_part_idx == k_part_loop - 1) {
                        SET_FLAG(MTE1, MTE2, event_id + 2);
                    }

                    SET_FLAG(MTE1, M, mte1_mad_event_id);
                    WAIT_FLAG(MTE1, M, mte1_mad_event_id);

                    bool init_c = (k_idx == 0 && k_part_idx == 0);
                    if (init_c) {
                        WAIT_FLAG(FIX, M, EVENT_ID0);
                    }

                    if (m != 1 && m_actual == 1 && TA) {
                        mmad<ArchType::ASCEND_V220, InDtype, InDtype, float, false>(l0c_buf, l0a_buf, l0b_buf,
                                                                                    16,        // m
                                                                                    n_actual,  // n
                                                                                    k0_actual, // k
                                                                                    init_c     // cmatrixInitVal
                        );
                    } else {
                        mmad<ArchType::ASCEND_V220, InDtype, InDtype, float, false>(l0c_buf, l0a_buf, l0b_buf,
                                                                                    m_actual,  // m
                                                                                    n_actual,  // n
                                                                                    k0_actual, // k
                                                                                    init_c     // cmatrixInitVal
                        );
                    }

                    AscendC::PipeBarrier<PIPE_M>();
                    SET_FLAG(M, MTE1, mte1_mad_event_id);
                }

                ping_flag = 1 - ping_flag;
            }

            SET_FLAG(M, FIX, EVENT_ID0);
            WAIT_FLAG(M, FIX, EVENT_ID0);

            // copy from L0C to gm
            l0c_to_gm<ArchType::ASCEND_V220, DataFormat::ND, OutDtype, float>(gm_c[offset_c], l0c_buf,
                                                                              m_actual, // MSize
                                                                              n_actual, // NSize
                                                                              m_round,  // srcStride
                                                                              n         // dstStride_dst_D
            );
            SET_FLAG(FIX, M, EVENT_ID0);
        }

        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        WAIT_FLAG(FIX, M, EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    AscendC::GlobalTensor<InDtype> gm_a;
    AscendC::GlobalTensor<InDtype> gm_b;
    AscendC::GlobalTensor<OutDtype> gm_c;
    AscendC::LocalTensor<InDtype> l1_base_a = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(0);
    AscendC::LocalTensor<InDtype> l1_base_b = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(128 * 1024);
    AscendC::LocalTensor<InDtype> l0a_base = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
    AscendC::LocalTensor<InDtype> l0b_base = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);
    AscendC::LocalTensor<float> l0c_buf = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);

    uint32_t num_core{0};
    uint32_t batch_size{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
    uint32_t m_loop{0};
    uint32_t n_loop{0};
    uint32_t k_loop{0};
    uint32_t core_loop{0};
    uint32_t swizzle_cnt{1};
    uint32_t core_idx{0};
    uint32_t ping_flag{0};
};

#endif
}

#endif