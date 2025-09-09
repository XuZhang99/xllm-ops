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
 * \file pp_mat_mul_f16_nd_nz_nd_kernel.h
 * \brief
 */

#ifndef __BATCH_MAT_MUL_V4_ND_NZ_ND_F16_KERNEL_H__
#define __BATCH_MAT_MUL_V4_ND_NZ_ND_F16_KERNEL_H__

#ifdef __CCE_KT_TEST__
#include "stub_def.h"
#include "stub_fun.h"
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
template <bool transA, bool transB, typename InDtype = half, typename OutDtype = half>
class PpMatmulF16NDNZND {
public:
    __aicore__ explicit PpMatmulF16NDNZND(GM_ADDR a, GM_ADDR b, GM_ADDR c, const PpMatmulTilingData* tilingData)
    {
        gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(a));
        gm_b.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(b));
        gm_c.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype *>(c));

        batch = tilingData->batch;
        m = tilingData->m;
        k = tilingData->k;
        n = tilingData->n;
        m0 = tilingData->m0;
        n0 = tilingData->n0;
        k0 = tilingData->k0;
        m_loop = tilingData->mLoop;
        k_loop = tilingData->kLoop;
        n_loop = tilingData->nLoop;
        core_loop = tilingData->coreLoop;
        l1_base_a = buf.GetBuffer<BufferType::ASCEND_CB, InDtype>(0);
        l1_base_b = buf.GetBuffer<BufferType::ASCEND_CB, InDtype>(RoundUp256(m0 * k0 * sizeof(InDtype)));
        core_num = AscendC::GetBlockNum();
        core_idx = AscendC::GetBlockIdx();

        ping_flag = 1;
    }

    __aicore__ FORCE_INLINE void Process()
    {
        SET_FLAG(MTE1, MTE2, EVENT_ID0);
        SET_FLAG(MTE1, MTE2, EVENT_ID1);
        SET_FLAG(MTE1, MTE2, EVENT_ID2);
        SET_FLAG(MTE1, MTE2, EVENT_ID3);
        SET_FLAG(M, MTE1, EVENT_ID0);
        SET_FLAG(M, MTE1, EVENT_ID1);
        SET_FLAG(FIX, M, EVENT_ID0);

        for (uint64_t loop_idx = core_idx; loop_idx < core_loop; loop_idx += core_num) {
            b_idx = loop_idx / m_loop / n_loop;
            m_idx = loop_idx / n_loop % m_loop;
            n_idx = loop_idx % n_loop;
            m_actual = (m_idx == (m_loop - 1)) ? (m - m_idx * m0) : m0;
            k_actual = (k_loop == 1) ? k : k0;
            n_actual = (n_idx == (n_loop - 1)) ? (n - n_idx * n0) : n0;
            m_round = RoundUp16(m_actual);
            k_round = RoundUp16(k_actual);
            n_round = RoundUp16(n_actual);
            if (transA) {
                offset_a = b_idx * m * k + m_idx * m0;
            } else {
                offset_a = b_idx * m * k + m_idx * m0 * k;
            }
            if (transB) {
                offset_b = b_idx * RoundUp16(n) * RoundUp16(k) + n_idx * n0 * BLOCK_SIZE_16;
            } else {
                offset_b = b_idx * RoundUp16(k) * RoundUp16(n) + n_idx * n0 * RoundUp16(k);
            }
            offset_c = b_idx * m * n + m_idx * m0 * n + n_idx * n0;
            k_part_len = RoundDown16(L0_PINGPONG_BUFFER_SIZE / Max(m_round, n_round));

            AscendC::LocalTensor<InDtype> l1_buf_a = ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_SIZE];
            AscendC::LocalTensor<InDtype> l1_buf_b = ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_SIZE];
            AscendC::LocalTensor<InDtype> l0a_buf = ping_flag ? l0a_base : l0a_base[L0_PINGPONG_BUFFER_SIZE];
            AscendC::LocalTensor<InDtype> l0b_buf = ping_flag ? l0b_base : l0b_base[L0_PINGPONG_BUFFER_SIZE];
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

            WAIT_FLAG(MTE1, MTE2, event_id);
            // *** load matrix A to L1
            if (m == 1 || m_actual == 1 && !transA) {
                gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::ND, DataFormat::ND>(
                    l1_buf_a, gm_a[offset_a], 1, RoundUp16(1), 1, k_round, RoundUp16(k_round), k_round);
            } else {
                if (transA) {
                    gm_to_l1<ArchType::ASCEND_V220, half, DataFormat::ND, DataFormat::NZ>(
                        l1_buf_a, gm_a[offset_a], k_actual, k_round, k, m_actual, m_round, m);
                } else {
                    gm_to_l1<ArchType::ASCEND_V220, half, DataFormat::ND, DataFormat::NZ>(
                        l1_buf_a, gm_a[offset_a], m_actual, m_round, m, k_actual, m_round, k);
                }
            }
            SET_FLAG(MTE2, MTE1, event_id);

            // *** load matrix B to L1
            WAIT_FLAG(MTE1, MTE2, event_id + 2);
            if (transB) {
                gm_to_l1<ArchType::ASCEND_V220, half, DataFormat::NZ, DataFormat::NZ>(
                    l1_buf_b, gm_b[offset_b], n_actual, n_round, RoundUp16(n), k_actual, k_round, RoundUp16(k));
            } else {
                gm_to_l1<ArchType::ASCEND_V220, half, DataFormat::NZ, DataFormat::NZ>(
                    l1_buf_b, gm_b[offset_b], k_actual, k_round, RoundUp16(k), n_actual, n_round, RoundUp16(n));
            }
            SET_FLAG(MTE2, MTE1, event_id + 2);

            for (uint64_t k_idx = 0; k_idx < k_loop; k_idx++) {

                uint64_t k_actual = (k_idx == (k_loop - 1)) ? (k - k_idx * k0) : k0;
                uint64_t k_round = RoundUp16(k_actual);
                uint64_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

                AscendC::LocalTensor<InDtype> l1_buf_a = ping_flag ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_SIZE];
                AscendC::LocalTensor<InDtype> l1_buf_b = ping_flag ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_SIZE];
                auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

                if (k_idx < k_loop - 1) {
                    if (transA) {
                        offset_a_next = b_idx * m * k + (k_idx + 1) * k0 * m + m_idx * m0;
                    } else {
                        offset_a_next = b_idx * m * k + m_idx * m0 * k + (k_idx + 1) * k0;
                    }

                    if (transB) {
                        offset_b_next = b_idx * RoundUp16(n) * RoundUp16(k) + (k_idx + 1) * k0 * RoundUp16(n) +
                                        n_idx * n0 * BLOCK_SIZE_16;
                    } else {
                        offset_b_next = b_idx * RoundUp16(k) * RoundUp16(n) + n_idx * n0 * RoundUp16(k) +
                                        (k_idx + 1) * k0 * BLOCK_SIZE_16;
                    }

                    uint64_t k_actual_next = ((k_idx + 1) == (k_loop - 1)) ? (k - (k_idx + 1) * k0) : k0;
                    uint64_t k_round_next = RoundUp16(k_actual_next);

                    AscendC::LocalTensor<InDtype> l1_buf_a_next =
                        (1 - ping_flag) ? l1_base_a : l1_base_a[L1_PINGPONG_BUFFER_SIZE];
                    AscendC::LocalTensor<InDtype> l1_buf_b_next =
                        (1 - ping_flag) ? l1_base_b : l1_base_b[L1_PINGPONG_BUFFER_SIZE];
                    auto event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                    WAIT_FLAG(MTE1, MTE2, event_id_next);
                    // *** load matrix A to L1
                    if (m == 1 || m_actual == 1 && !transA) {
                        gm_to_l1<ArchType::ASCEND_V220, InDtype, DataFormat::ND, DataFormat::ND>(
                            l1_buf_a_next,       // dst
                            gm_a[offset_a_next], // src
                            1, RoundUp16(1), 1, k_round_next, RoundUp16(k_round_next), k_round_next);
                    } else {
                        if (transA) {
                            gm_to_l1<ArchType::ASCEND_V220, half, DataFormat::ND, DataFormat::NZ>(
                                l1_buf_a_next, gm_a[offset_a_next], k_actual_next, k_round_next, k, m_actual, m_round,
                                m);
                        } else {
                            gm_to_l1<ArchType::ASCEND_V220, half, DataFormat::ND, DataFormat::NZ>(
                                l1_buf_a_next, gm_a[offset_a_next], m_actual, m_round, m, k_actual_next, m_round, k);
                        }
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next);

                    // *** load matrix B to L1
                    WAIT_FLAG(MTE1, MTE2, event_id_next + 2);
                    if (transB) {
                        gm_to_l1<ArchType::ASCEND_V220, half, DataFormat::NZ, DataFormat::NZ>(
                            l1_buf_b_next, gm_b[offset_b_next], n_actual, n_round, RoundUp16(n), k_actual_next,
                            k_round_next, RoundUp16(k));
                    } else {
                        gm_to_l1<ArchType::ASCEND_V220, half, DataFormat::NZ, DataFormat::NZ>(
                            l1_buf_b_next, gm_b[offset_b_next], k_actual_next, k_round_next, RoundUp16(k), n_actual,
                            n_round, RoundUp16(n));
                    }
                    SET_FLAG(MTE2, MTE1, event_id_next + 2);
                }

                for (uint64_t k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                    uint64_t k0_round =
                        (k_part_idx < (k_part_loop - 1)) ? k_part_len : k_round - k_part_idx * k_part_len;
                    uint64_t k0_actual =
                        (k_part_idx < (k_part_loop - 1)) ? k_part_len : k_actual - k_part_idx * k_part_len;

                    auto mte1_mad_ping_flag = 1 - k_part_idx % 2;
                    auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                    AscendC::LocalTensor<InDtype> l0a_buf = l0a_base[(k_part_idx % 2) * L0_PINGPONG_BUFFER_SIZE];
                    AscendC::LocalTensor<InDtype> l0b_buf = l0b_base[(k_part_idx % 2) * L0_PINGPONG_BUFFER_SIZE];

                    // *** load matrix A from L1 to L0A
                    if (k_part_idx == 0) {
                        WAIT_FLAG(MTE2, MTE1, event_id);
                    }
                    WAIT_FLAG(M, MTE1, mte1_mad_event_id);
                    if (m == 1 || m_actual == 1 && !transA) {
                        l1_to_l0_a<ArchType::ASCEND_V220, InDtype, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0a_buf, l1_buf_a[k_part_idx * k_part_len], 0,
                            CeilDiv256(k0_round), // repeat
                            0,
                            1, // srcStride
                            0,
                            0 // dstStride
                        );
                    } else {
                        if (transA) {
                            l1_to_l0_a<ArchType::ASCEND_V220, InDtype, true, DataFormat::ZN, DataFormat::ZZ>(
                                l0a_buf, l1_buf_a[k_part_idx * k_part_len * BLOCK_SIZE_16], m_round,
                                k0_round, // repeat
                                k_round / BLOCK_SIZE_16,
                                1, // srcStride
                                k0_round / BLOCK_SIZE_16,
                                1 // dstStride
                            );
                        } else {
                            l1_to_l0_a<ArchType::ASCEND_V220, InDtype, false, DataFormat::ZN, DataFormat::ZZ>(
                                l0a_buf, l1_buf_a[k_part_idx * k_part_len * m_round], m_round,
                                k0_round, // repeat
                                1,
                                m_round / BLOCK_SIZE_16, // srcStride
                                k0_round / BLOCK_SIZE_16,
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
                    if (transB) {
                        l1_to_l0_b<ArchType::ASCEND_V220, InDtype, false, DataFormat::VECTOR, DataFormat::VECTOR>(
                            l0b_buf,                                     // dst
                            l1_buf_b[k_part_idx * k_part_len * n_round], // src
                            0,
                            k0_round * n_round / CUBE_MATRIX_SIZE_256, // repeat
                            0,
                            1, // srcStride
                            0,
                            0 // dstStride
                        );
                    } else {
                        l1_to_l0_b<ArchType::ASCEND_V220, InDtype, false, DataFormat::ZN, DataFormat::NZ>(
                            l0b_buf,                                           // dst
                            l1_buf_b[k_part_idx * k_part_len * BLOCK_SIZE_16], // src
                            n_round,
                            k0_round, // repeat
                            k_round / BLOCK_SIZE_16,
                            1, // srcStride
                            1,
                            n_round / BLOCK_SIZE_16 // dstStride
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
                    if (m != 1 && m_actual == 1 && transA) {
                        mmad<ArchType::ASCEND_V220, InDtype, InDtype, float, false>(l0c_buf, l0a_buf, l0b_buf,
                                                                                    BLOCK_SIZE_16, // m
                                                                                    n_actual,      // n
                                                                                    k0_actual,     // k
                                                                                    init_c         // cmatrixInitVal
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

        WAIT_FLAG(MTE1, MTE2, EVENT_ID0);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID1);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID2);
        WAIT_FLAG(MTE1, MTE2, EVENT_ID3);
        WAIT_FLAG(M, MTE1, EVENT_ID0);
        WAIT_FLAG(M, MTE1, EVENT_ID1);
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

    uint64_t batch{0};
    uint64_t m{0};
    uint64_t k{0};
    uint64_t n{0};
    uint64_t m0{0};
    uint64_t k0{0};
    uint64_t n0{0};
    uint64_t m_loop{0};
    uint64_t n_loop{0};
    uint64_t k_loop{0};

    uint64_t b_idx{0};
    uint64_t m_idx{0};
    uint64_t n_idx{0};
    uint64_t offset_a{0};
    uint64_t offset_b{0};
    uint64_t offset_c{0};
    uint64_t offset_a_next{0};
    uint64_t offset_b_next{0};
    uint64_t m_actual{0};
    uint64_t k_actual{0};
    uint64_t n_actual{0};
    uint64_t m_round{0};
    uint64_t k_round{0};
    uint64_t n_round{0};
    uint64_t k_part_len{0};

    uint64_t core_loop{0};
    uint64_t core_idx{0};
    uint64_t core_num{0};
    uint64_t ping_flag{0};
};

#endif
}
#endif