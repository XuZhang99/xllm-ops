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

#include "pp_mat_mul_fp16_kernel.h"
#include "pp_mat_mul_bf16_kernel.h"
#include "pp_mat_mul_accum_atomic_kernel.h"
#include "pp_mat_mul_bf16_nd_nz_nd_kernel.h"
#include "pp_mat_mul_ein_sum_kernel.h"
#include "pp_mat_mul_f16_nd_nz_nd_kernel.h"
#include "pp_mat_mul_with_bias_kernel.h"
#include "pp_mat_mul_f16_nz_kernel.h"
#include "lib/matmul_intf.h"

using namespace PpMatMulNS;

extern "C" __global__ __aicore__ void pp_mat_mul(
    GM_ADDR gm_a,
    GM_ADDR gm_b,
    GM_ADDR gm_bias,
    GM_ADDR gm_c_ref,
    GM_ADDR gm_c,
    GM_ADDR workspaceGM,
    GM_ADDR tiling_data
    )
{
    GET_TILING_DATA(tilingData, tiling_data);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
#ifdef __DAV_C220_CUBE__
#if defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2) && defined(ORIG_DTYPE_Y) && \
defined(FORMAT_X1) && defined(FORMAT_X2) && defined(FORMAT_Y) && \
((ORIG_DTYPE_X1 == DT_FLOAT16) && (ORIG_DTYPE_X2 == DT_FLOAT16) && (ORIG_DTYPE_Y == DT_FLOAT16) && \
(FORMAT_X1 == FORMAT_ND) && (FORMAT_X2 == FORMAT_ND) && (FORMAT_Y == FORMAT_ND))
    if (TILING_KEY_IS(21760)) { // 0b1'01'01'01'0'0'0'0'0'000
        PpMatmulFp16<0, false, false> ppmatmul_fp16_000; // swizzleDir[0] transA[0] transB[0]
        PpMatmulFp16<1, false, false> ppmatmul_fp16_100; // swizzleDir[1] transA[0] transB[0]

        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_fp16_000.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_fp16_000.run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_fp16_100.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_fp16_100.run();
        }

    } else if (TILING_KEY_IS(21768)) { // 0b1'01'01'01'0'0'0'0'1'000

        PpMatmulFp16<0, false, true> ppmatmul_fp16_001; // swizzleDir[0] transA[0] transB[1]
        PpMatmulFp16<1, false, true> ppmatmul_fp16_101; // swizzleDir[1] transA[0] transB[1]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_fp16_001.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_fp16_001.run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_fp16_101.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_fp16_101.run();
        }

    } else if (TILING_KEY_IS(21776)) { // 0b1'01'01'01'0'0'0'1'0'000
        PpMatmulFp16<0, true, false> ppmatmul_fp16_010; // swizzleDir[0] transA[1] transB[0]
        PpMatmulFp16<1, true, false> ppmatmul_fp16_110; // swizzleDir[1] transA[1] transB[0]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_fp16_010.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_fp16_010.run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_fp16_110.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_fp16_110.run();
        }

    } else if (TILING_KEY_IS(21784)) { // 0b1'01'01'01'0'0'0'1'1'000
        PpMatmulFp16<0, true, true> ppmatmul_fp16_011; // swizzleDir[0] transA[1] transB[1]
        PpMatmulFp16<1, true, true> ppmatmul_fp16_111; // swizzleDir[1] transA[1] transB[1]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_fp16_011.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_fp16_011.run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_fp16_111.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_fp16_111.run();
        }
    } else if (TILING_KEY_IS(21764)) { // 0b1'01'01'01'0'0'0'0'0'100
        // swizzleDir[0] transA[0] transB[0] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0] DataFormatB[0]
        PpMatmulEinSum<0, false, false, half, half, DataFormat::ND> einsum_0_n_fp16_nd;
        // swizzleDir[1] transA[0] transB[0] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0] DataFormatB[0]
        PpMatmulEinSum<1, false, false, half, half, DataFormat::ND> einsum_1_n_fp16_nd;
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();

        if (tilingData.swizzlDirect == 0) {
            einsum_0_n_fp16_nd.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_0_n_fp16_nd.Process();
        } else if (tilingData.swizzlDirect == 1) {
            einsum_1_n_fp16_nd.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_1_n_fp16_nd.Process();
        }
    } else if (TILING_KEY_IS(21772)) { // 0b1'01'01'01'0'0'0'0'1'100
        // swizzleDir[0] transA[0] transB[1] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0] DataFormatB[0]
        PpMatmulEinSum<0, false, true, half, half, DataFormat::ND> einsum_0_t_fp16_nd;
        // swizzleDir[1] transA[0] transB[1] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0] DataFormatB[0]
        PpMatmulEinSum<1, false, true, half, half, DataFormat::ND> einsum_1_t_fp16_nd;
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();

        if (tilingData.swizzlDirect == 0) {
            einsum_0_t_fp16_nd.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_0_t_fp16_nd.Process();
        } else if (tilingData.swizzlDirect == 1) {
            einsum_1_t_fp16_nd.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_1_t_fp16_nd.Process();
        }
    } else if (TILING_KEY_IS(21763)) { // 0b1'01'01'01'0'0'0'0'0'011
        PpMatmulWithBias<0, false, false, true, half, half, float> ppmatmul_with_bias_000001; // swizzleDir[0] transA[0] transB[0] DtypeA[001]
        PpMatmulWithBias<1, false, false, true, half, half, float> ppmatmul_with_bias_100001; // swizzleDir[1] transA[0] transB[0] DtypeA[001]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        SetMasknorm();

        if (tilingData.swizzlDirect == 0) {
            ppmatmul_with_bias_000001.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_000001.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_with_bias_100001.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_100001.Run();
        }
    } else if (TILING_KEY_IS(21771)) { // 0b1'01'01'01'0'0'0'0'1'011
        PpMatmulWithBias<0, false, true, true, half, half, float> ppmatmul_with_bias_001001; // swizzleDir[0] transA[0] transB[1] DtypeA[001]
        PpMatmulWithBias<1, false, true, true, half, half, float> ppmatmul_with_bias_101001; // swizzleDir[1] transA[0] transB[1] DtypeA[001]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        SetMasknorm();

        if (tilingData.swizzlDirect == 0) {
            ppmatmul_with_bias_001001.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_001001.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_with_bias_101001.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_101001.Run();
        }
    } else if (TILING_KEY_IS(21779)) { // 0b1'01'01'01'0'0'0'1'0'011
        PpMatmulWithBias<0, true, false, true, half, half, float> ppmatmul_with_bias_010001; // swizzleDir[0] transA[1] transB[0] DtypeA[001]
        PpMatmulWithBias<1, true, false, true, half, half, float> ppmatmul_with_bias_110001; // swizzleDir[1] transA[1] transB[0] DtypeA[001]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        SetMasknorm();

        if (tilingData.swizzlDirect == 0) {
            ppmatmul_with_bias_010001.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_010001.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_with_bias_110001.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_110001.Run();
        }
    } else if (TILING_KEY_IS(21787)) { // 0b1'01'01'01'0'0'0'1'1'011
        PpMatmulWithBias<0, true, true, true, half, half, float> ppmatmul_with_bias_011001; // swizzleDir[0] transA[1] transB[1] DtypeA[001]
        PpMatmulWithBias<1, true, true, true, half, half, float> ppmatmul_with_bias_111001; // swizzleDir[1] transA[1] transB[1] DtypeA[001]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        SetMasknorm();

        if (tilingData.swizzlDirect == 0) {
            ppmatmul_with_bias_011001.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_011001.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_with_bias_111001.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_111001.Run();
        }
    }
#endif

#if defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2) && defined(ORIG_DTYPE_Y) && \
defined(FORMAT_X1) && defined(FORMAT_X2) && defined(FORMAT_Y) && \
((ORIG_DTYPE_X1 == DT_BF16) && (ORIG_DTYPE_X2 == DT_BF16) && (ORIG_DTYPE_Y == DT_BF16) && \
(FORMAT_X1 == FORMAT_ND) && (FORMAT_X2 == FORMAT_ND) && (FORMAT_Y == FORMAT_ND))
    if (TILING_KEY_IS(27136)) { // 0b1'10'10'10'0'0'0'0'0'000
        PpMatmulBf16<0, false, false> ppmatmul_bf16_000; // swizzleDir[0] transA[0] transB[0]
        PpMatmulBf16<1, false, false> ppmatmul_bf16_100; // swizzleDir[1] transA[0] transB[0]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_bf16_000.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_000.run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_bf16_100.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_100.run();
        }
    } else if (TILING_KEY_IS(27144)) { // 0b1'10'10'10'0'0'0'0'1'000
        PpMatmulBf16<0, false, true> ppmatmul_bf16_001; // swizzleDir[0] transA[0] transB[1]
        PpMatmulBf16<1, false, true> ppmatmul_bf16_101; // swizzleDir[1] transA[0] transB[1]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_bf16_001.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_001.run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_bf16_101.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_101.run();
        }
    } else if (TILING_KEY_IS(27152)) { // 0b1'10'10'10'0'0'0'1'0'000
        PpMatmulBf16<0, true, false> ppmatmul_bf16_010; // swizzleDir[0] transA[1] transB[0]
        PpMatmulBf16<1, true, false> ppmatmul_bf16_110; // swizzleDir[1] transA[1] transB[0]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_bf16_010.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_010.run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_bf16_110.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_110.run();
        }
    } else if (TILING_KEY_IS(27160)) { // 0b1'10'10'10'0'0'0'1'1'000
        PpMatmulBf16<0, true, true> ppmatmul_bf16_011; // swizzleDir[0] transA[1] transB[1]
        PpMatmulBf16<1, true, true> ppmatmul_bf16_111; // swizzleDir[1] transA[1] transB[1]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_bf16_011.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_011.run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_bf16_111.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_111.run();
        }
    } else if (TILING_KEY_IS(27140)) { // 0b1'10'10'10'0'0'0'0'0'100
        // swizzleDir[0] transA[0] transB[0] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0] DataFormatB[0]
        PpMatmulEinSum<0, false, false, __bf16, __bf16, DataFormat::ND> einsum_0_n_bf16_nd;
        // swizzleDir[1] transA[0] transB[0] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0] DataFormatB[0]
        PpMatmulEinSum<1, false, false, __bf16, __bf16, DataFormat::ND> einsum_1_n_bf16_nd;
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();

        if (tilingData.swizzlDirect == 0) {
            einsum_0_n_bf16_nd.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_0_n_bf16_nd.Process();
        } else if (tilingData.swizzlDirect == 1) {
            einsum_1_n_bf16_nd.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_1_n_bf16_nd.Process();
        }
    } else if (TILING_KEY_IS(27148)) { // 0b1'10'10'10'0'0'0'0'1'100
        // swizzleDir[0] transA[0] transB[1] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0] DataFormatB[0]
        PpMatmulEinSum<0, false, true, __bf16, __bf16, DataFormat::ND> einsum_0_t_bf16_nd;
        // swizzleDir[1] transA[0] transB[1] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0] DataFormatB[0]
        PpMatmulEinSum<1, false, true, __bf16, __bf16, DataFormat::ND> einsum_1_t_bf16_nd;
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();

        if (tilingData.swizzlDirect == 0) {
            einsum_0_t_bf16_nd.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_0_t_bf16_nd.Process();
        } else if (tilingData.swizzlDirect == 1) {
            einsum_1_t_bf16_nd.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_1_t_bf16_nd.Process();
        }
    } else if (TILING_KEY_IS(27139)) { // 0b1'10'10'10'0'0'0'0'0'011
        PpMatmulWithBias<0, false, false, true, __bf16, __bf16, float> ppmatmul_with_bias_000010; // swizzleDir[0] transA[0] transB[0] DtypeA[010]
        PpMatmulWithBias<1, false, false, true, __bf16, __bf16, float> ppmatmul_with_bias_100010; // swizzleDir[1] transA[0] transB[0] DtypeA[010]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        SetMasknorm();

        if (tilingData.swizzlDirect == 0) {
            ppmatmul_with_bias_000010.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_000010.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_with_bias_100010.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_100010.Run();
        }
    } else if (TILING_KEY_IS(27147)) { // 0b1'10'10'10'0'0'0'0'1'011
        PpMatmulWithBias<0, false, true, true, __bf16, __bf16, float> ppmatmul_with_bias_001010; // swizzleDir[0] transA[0] transB[1] DtypeA[010]
        PpMatmulWithBias<1, false, true, true, __bf16, __bf16, float> ppmatmul_with_bias_101010; // swizzleDir[1] transA[0] transB[1] DtypeA[010]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        SetMasknorm();

        if (tilingData.swizzlDirect == 0) {
            ppmatmul_with_bias_001010.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_001010.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_with_bias_101010.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_101010.Run();
        }
    } else if (TILING_KEY_IS(27155)) { // 0b1'10'10'10'0'0'0'1'0'011
        PpMatmulWithBias<0, true, false, true, __bf16, __bf16, float> ppmatmul_with_bias_010010; // swizzleDir[0] transA[1] transB[0] DtypeA[010]
        PpMatmulWithBias<1, true, false, true, __bf16, __bf16, float> ppmatmul_with_bias_110010; // swizzleDir[1] transA[1] transB[0] DtypeA[010]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        SetMasknorm();

        if (tilingData.swizzlDirect == 0) {
            ppmatmul_with_bias_010010.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_010010.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_with_bias_110010.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_110010.Run();
        }
    } else if (TILING_KEY_IS(27163)) { // 0b1'10'10'10'0'0'0'1'1'011
        PpMatmulWithBias<0, true, true, true, __bf16, __bf16, float> ppmatmul_with_bias_011010; // swizzleDir[0] transA[1] transB[1] DtypeA[010]
        PpMatmulWithBias<1, true, true, true, __bf16, __bf16, float> ppmatmul_with_bias_111010; // swizzleDir[1] transA[1] transB[1] DtypeA[010]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();
        SetMasknorm();

        if (tilingData.swizzlDirect == 0) {
            ppmatmul_with_bias_011010.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_011010.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_with_bias_111010.SetArgs(gm_a, gm_b, gm_bias, gm_c, &tilingData);
            ppmatmul_with_bias_111010.Run();
        }
    }
#endif

#if defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2) && defined(ORIG_DTYPE_Y) && \
defined(FORMAT_X1) && defined(FORMAT_X2) && defined(FORMAT_Y) && \
((ORIG_DTYPE_X1 == DT_FLOAT16) && (ORIG_DTYPE_X2 == DT_FLOAT16) && (ORIG_DTYPE_Y == DT_FLOAT) && \
(FORMAT_X1 == FORMAT_ND) && (FORMAT_X2 == FORMAT_ND) && (FORMAT_Y == FORMAT_ND))
    if (TILING_KEY_IS(22274)) { // 0b1'01'01'11'0'0'0'0'0'010
        PpMatmulAccumAtomic<0, false, false, half, float, float> ppmatmul_accum_atomic_000001; // swizzleDir[0] transA[0] transB[0] DtypeA[001]
        PpMatmulAccumAtomic<1, false, false, half, float, float> ppmatmul_accum_atomic_100001; // swizzleDir[1] transA[0] transB[0] DtypeA[001]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        AscendC::SetAtomicAdd<float>();
        AscendC::SetMaskNorm();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_accum_atomic_000001.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_000001.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_accum_atomic_100001.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_100001.Run();
        }
    }else if (TILING_KEY_IS(22282)) { // 0b1'01'01'11'0'0'0'0'1'010
        PpMatmulAccumAtomic<0, false, true, half, float, float> ppmatmul_accum_atomic_001001;  // swizzleDir[0] transA[0] transB[1] DtypeA[001]
        PpMatmulAccumAtomic<1, false, true, half, float, float> ppmatmul_accum_atomic_101001;  // swizzleDir[1] transA[0] transB[1] DtypeA[001]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        AscendC::SetAtomicAdd<float>();
        AscendC::SetMaskNorm();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_accum_atomic_001001.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_001001.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_accum_atomic_101001.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_101001.Run();
        }
    } else if (TILING_KEY_IS(22290)) { // 0b1'01'01'11'0'0'0'1'0'010
        PpMatmulAccumAtomic<0, true, false, half, float, float> ppmatmul_accum_atomic_010001;  // swizzleDir[0] transA[1] transB[0] DtypeA[001]
        PpMatmulAccumAtomic<1, true, false, half, float, float> ppmatmul_accum_atomic_110001;  // swizzleDir[1] transA[1] transB[0] DtypeA[001]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        AscendC::SetAtomicAdd<float>();
        AscendC::SetMaskNorm();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_accum_atomic_010001.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_010001.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_accum_atomic_110001.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_110001.Run();
        }
    } else if (TILING_KEY_IS(22298)) { // 0b1'01'01'11'0'0'0'1'1'010
        PpMatmulAccumAtomic<0, true, true, half, float, float> ppmatmul_accum_atomic_011001;   // swizzleDir[0] transA[1] transB[1] DtypeA[001]
        PpMatmulAccumAtomic<1, true, true, half, float, float> ppmatmul_accum_atomic_111001;   // swizzleDir[1] transA[1] transB[1] DtypeA[001]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        AscendC::SetAtomicAdd<float>();
        AscendC::SetMaskNorm();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_accum_atomic_011001.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_011001.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_accum_atomic_111001.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_111001.Run();
        }
    }
#endif

#if defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2) && defined(ORIG_DTYPE_Y) && \
defined(FORMAT_X1) && defined(FORMAT_X2) && defined(FORMAT_Y) && \
((ORIG_DTYPE_X1 == DT_BF16) && (ORIG_DTYPE_X2 == DT_BF16) && (ORIG_DTYPE_Y == DT_FLOAT) && \
(FORMAT_X1 == FORMAT_ND) && (FORMAT_X2 == FORMAT_ND) && (FORMAT_Y == FORMAT_ND))
    if (TILING_KEY_IS(27394)) { // 0b1'10'10'11'0'0'0'0'0'010
        PpMatmulAccumAtomic<0, false, false, __bf16, float, float> ppmatmul_accum_atomic_000010; // swizzleDir[0] transA[0] transB[0] DtypeA[010]
        PpMatmulAccumAtomic<1, false, false, __bf16, float, float> ppmatmul_accum_atomic_100010; // swizzleDir[1] transA[0] transB[0] DtypeA[010]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        AscendC::SetAtomicAdd<float>();
        AscendC::SetMaskNorm();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_accum_atomic_000010.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_000010.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_accum_atomic_100010.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_100010.Run();
        }
    } else if (TILING_KEY_IS(27402)) { // 0b1'10'10'11'0'0'0'0'1'010
        PpMatmulAccumAtomic<0, false, true, __bf16, float, float> ppmatmul_accum_atomic_001010;  // swizzleDir[0] transA[0] transB[1] DtypeA[010]
        PpMatmulAccumAtomic<1, false, true, __bf16, float, float> ppmatmul_accum_atomic_101010;  // swizzleDir[1] transA[0] transB[1] DtypeA[010]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        AscendC::SetAtomicAdd<float>();
        AscendC::SetMaskNorm();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_accum_atomic_001010.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_001010.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_accum_atomic_101010.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_101010.Run();
        }
    } else if (TILING_KEY_IS(27410)) { // 0b1'10'10'11'0'0'0'1'0'010
        PpMatmulAccumAtomic<0, true, false, __bf16, float, float> ppmatmul_accum_atomic_010010;  // swizzleDir[0] transA[1] transB[0] DtypeA[010]
        PpMatmulAccumAtomic<1, true, false, __bf16, float, float> ppmatmul_accum_atomic_110010;  // swizzleDir[1] transA[1] transB[0] DtypeA[010]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        AscendC::SetAtomicAdd<float>();
        AscendC::SetMaskNorm();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_accum_atomic_010010.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_010010.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_accum_atomic_110010.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_110010.Run();
        }
    } else if (TILING_KEY_IS(27418)) { // 0b1'10'10'11'0'0'0'1'1'010
        PpMatmulAccumAtomic<0, true, true, __bf16, float, float> ppmatmul_accum_atomic_011010;   // swizzleDir[0] transA[1] transB[1] DtypeA[010]
        PpMatmulAccumAtomic<1, true, true, __bf16, float, float> ppmatmul_accum_atomic_111010;   // swizzleDir[1] transA[1] transB[1] DtypeA[010]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        AscendC::SetAtomicAdd<float>();
        AscendC::SetMaskNorm();
        if (tilingData.swizzlDirect == 0) {
            ppmatmul_accum_atomic_011010.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_011010.Run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_accum_atomic_111010.Init(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_accum_atomic_111010.Run();
        }
    }
#endif

#if defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2) && defined(ORIG_DTYPE_Y) && \
defined(FORMAT_X1) && defined(FORMAT_X2) && defined(FORMAT_Y) && \
((ORIG_DTYPE_X1 == DT_FLOAT16) && (ORIG_DTYPE_X2 == DT_FLOAT16) && (ORIG_DTYPE_Y == DT_FLOAT16) && \
(FORMAT_X1 == FORMAT_ND) && (FORMAT_X2 == FORMAT_FRACTAL_NZ) && (FORMAT_Y == FORMAT_ND))
    if (TILING_KEY_IS(21824)) { // 0b1'01'01'01'0'1'0'0'0'000
        PpMatmulF16NDNZND<false, false> ppmatmul_f16_nd_nz_nd_nn(gm_a, gm_b, gm_c, &tilingData);
        SetPadding<uint64_t>((uint64_t)0);
        SetAtomicnone();
        SetNdpara(1, 0, 0);
        ppmatmul_f16_nd_nz_nd_nn.Process();
    } else if (TILING_KEY_IS(21832)) { // 0b1'01'01'01'0'1'0'0'1'000
        PpMatmulF16NDNZND<false, true> ppmatmul_f16_nd_nz_nd_nt(gm_a, gm_b, gm_c, &tilingData);
        SetPadding<uint64_t>((uint64_t)0);
        SetAtomicnone();
        SetNdpara(1, 0, 0);
        ppmatmul_f16_nd_nz_nd_nt.Process();
    } else if (TILING_KEY_IS(21840)) { // 0b1'01'01'01'0'1'0'1'0'000
        PpMatmulF16NDNZND<true, false> ppmatmul_f16_nd_nz_nd_tn(gm_a, gm_b, gm_c, &tilingData);
        SetPadding<uint64_t>((uint64_t)0);
        SetAtomicnone();
        SetNdpara(1, 0, 0);
        ppmatmul_f16_nd_nz_nd_tn.Process();
    } else if (TILING_KEY_IS(21848)) { // 0b1'01'01'01'0'1'0'1'1'000
        PpMatmulF16NDNZND<true, true> ppmatmul_f16_nd_nz_nd_tt(gm_a, gm_b, gm_c, &tilingData);
        SetPadding<uint64_t>((uint64_t)0);
        SetAtomicnone();
        SetNdpara(1, 0, 0);
        ppmatmul_f16_nd_nz_nd_tt.Process();
    } else if (TILING_KEY_IS(21828)) { // 0b1'01'01'01'0'1'0'0'0'100
        // swizzleDir[0] transA[0] transB[0] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0] DataFormatB[1]
        PpMatmulEinSum<0, false, false, half, half, DataFormat::NZ> einsum_0_n_fp16_nz;
        // swizzleDir[1] transA[0] transB[0] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0] DataFormatB[1]
        PpMatmulEinSum<1, false, false, half, half, DataFormat::NZ> einsum_1_n_fp16_nz;
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();

        if (tilingData.swizzlDirect == 0) {
            einsum_0_n_fp16_nz.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_0_n_fp16_nz.Process();
        } else if (tilingData.swizzlDirect == 1) {
            einsum_1_n_fp16_nz.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_1_n_fp16_nz.Process();
        }
    } else if (TILING_KEY_IS(21836)) { // 0b1'01'01'01'0'1'0'0'1'100
        // swizzleDir[0] transA[0] transB[1] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0] DataFormatB[1]
        PpMatmulEinSum<0, false, true, half, half, DataFormat::NZ> einsum_0_t_fp16_nz;
        // swizzleDir[1] transA[0] transB[1] DtypeA[001] DtypeB[001] DtypeC[001] DataFormatA[0] DataFormatB[1]    
        PpMatmulEinSum<1, false, true, half, half, DataFormat::NZ> einsum_1_t_fp16_nz;
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();

        if (tilingData.swizzlDirect == 0) {
            einsum_0_t_fp16_nz.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_0_t_fp16_nz.Process();
        } else if (tilingData.swizzlDirect == 1) {
            einsum_1_t_fp16_nz.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_1_t_fp16_nz.Process();
        }
    }
#endif

#if defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2) && defined(ORIG_DTYPE_Y) && \
defined(FORMAT_X1) && defined(FORMAT_X2) && defined(FORMAT_Y) && \
((ORIG_DTYPE_X1 == DT_BF16) && (ORIG_DTYPE_X2 == DT_BF16) && (ORIG_DTYPE_Y == DT_BF16) && \
(FORMAT_X1 == FORMAT_ND) && (FORMAT_X2 == FORMAT_FRACTAL_NZ) && (FORMAT_Y == FORMAT_ND))
    if (TILING_KEY_IS(27200)) { // 0b1'10'10'10'0'1'0'0'0'000
        PpMatmulBF16NDNZND<0, false, false> ppmatmul_bf16_nd_nz_nd_000; // swizzleDir[0] transA[0] transB[0]
        PpMatmulBF16NDNZND<1, false, false> ppmatmul_bf16_nd_nz_nd_100; // swizzleDir[1] transA[0] transB[0]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();

        if (tilingData.swizzlDirect == 0) {
            ppmatmul_bf16_nd_nz_nd_000.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_nd_nz_nd_000.run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_bf16_nd_nz_nd_100.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_nd_nz_nd_100.run();
        }
    } else if (TILING_KEY_IS(27208)) { // 0b1'10'10'10'0'1'0'0'1'000
        PpMatmulBF16NDNZND<0, false, true> ppmatmul_bf16_nd_nz_nd_001; // swizzleDir[0] transA[0] transB[1]
        PpMatmulBF16NDNZND<1, false, true> ppmatmul_bf16_nd_nz_nd_101; // swizzleDir[1] transA[0] transB[1]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();

        if (tilingData.swizzlDirect == 0) {
            ppmatmul_bf16_nd_nz_nd_001.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_nd_nz_nd_001.run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_bf16_nd_nz_nd_101.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_nd_nz_nd_101.run();
        }
    } else if (TILING_KEY_IS(27216)) { // 0b1'10'10'10'0'1'0'1'0'000
        PpMatmulBF16NDNZND<0, true, false> ppmatmul_bf16_nd_nz_nd_010; // swizzleDir[0] transA[1] transB[0]
        PpMatmulBF16NDNZND<1, true, false> ppmatmul_bf16_nd_nz_nd_110; // swizzleDir[1] transA[1] transB[0]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();

        if (tilingData.swizzlDirect == 0) {
            ppmatmul_bf16_nd_nz_nd_010.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_nd_nz_nd_010.run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_bf16_nd_nz_nd_110.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_nd_nz_nd_110.run();
        }
    } else if (TILING_KEY_IS(27224)) { // 0b1'10'10'10'0'1'0'1'1'000
        PpMatmulBF16NDNZND<0, true, true> ppmatmul_bf16_nd_nz_nd_011; // swizzleDir[0] transA[1] transB[1]
        PpMatmulBF16NDNZND<1, true, true> ppmatmul_bf16_nd_nz_nd_111; // swizzleDir[1] transA[1] transB[1]
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();

        if (tilingData.swizzlDirect == 0) {
            ppmatmul_bf16_nd_nz_nd_011.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_nd_nz_nd_011.run();
        } else if (tilingData.swizzlDirect == 1) {
            ppmatmul_bf16_nd_nz_nd_111.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            ppmatmul_bf16_nd_nz_nd_111.run();
        }
    } else if (TILING_KEY_IS(27204)) { // 0b1'10'10'10'0'1'0'0'0'100
        // swizzleDir[0] transA[0] transB[0] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0] DataFormatB[1]
        PpMatmulEinSum<0, false, false, __bf16, __bf16, DataFormat::NZ> einsum_0_n_bf16_nz;
        // swizzleDir[1] transA[0] transB[0] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0] DataFormatB[1]
        PpMatmulEinSum<1, false, false, __bf16, __bf16, DataFormat::NZ> einsum_1_n_bf16_nz;
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();

        if (tilingData.swizzlDirect == 0) {
            einsum_0_n_bf16_nz.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_0_n_bf16_nz.Process();
        } else if (tilingData.swizzlDirect == 1) {
            einsum_1_n_bf16_nz.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_1_n_bf16_nz.Process();
        }
    } else if (TILING_KEY_IS(27212)) { // 0b1'10'10'10'0'1'0'0'1'100
        // swizzleDir[0] transA[0] transB[1] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0] DataFormatB[1]
        PpMatmulEinSum<0, false, true, __bf16, __bf16, DataFormat::NZ> einsum_0_t_bf16_nz;
        // swizzleDir[1] transA[0] transB[1] DtypeA[010] DtypeB[010] DtypeC[010] DataFormatA[0] DataFormatB[1]
        PpMatmulEinSum<1, false, true, __bf16, __bf16, DataFormat::NZ> einsum_1_t_bf16_nz;
        SetPadding<uint64_t>((uint64_t)0);
        SetNdpara(1, 0, 0);
        SetAtomicnone();

        if (tilingData.swizzlDirect == 0) {
            einsum_0_t_bf16_nz.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_0_t_bf16_nz.Process();
        } else if (tilingData.swizzlDirect == 1) {
            einsum_1_t_bf16_nz.Init(gm_a, gm_b, gm_c, &tilingData);
            einsum_1_t_bf16_nz.Process();
        }
    }
#endif
#endif

#ifdef __DAV_M200__
#if defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_X2) && defined(ORIG_DTYPE_Y) && \
defined(FORMAT_X1) && defined(FORMAT_X2) && defined(FORMAT_Y) && \
((ORIG_DTYPE_X1 == DT_FLOAT16) && (ORIG_DTYPE_X2 == DT_FLOAT16) && (ORIG_DTYPE_Y == DT_FLOAT16) && \
(FORMAT_X1 == FORMAT_FRACTAL_NZ) && (FORMAT_X2 == FORMAT_FRACTAL_NZ) && (FORMAT_Y == FORMAT_FRACTAL_NZ))
    if (TILING_KEY_IS(5600)) { // 0b0'01'01'01'1'1'1'0'0'000
        PpMatmulF16NZ<0, false, false, half, half, float, false> matmul_0000;
        PpMatmulF16NZ<1, false, false, half, half, float, false> matmul_1000;
        PpMatmulF16NZ<0, false, false, half, half, float, true> matmul_0001;
        PpMatmulF16NZ<1, false, false, half, half, float, true> matmul_1001;
        SetPadding<uint64_t>(0);
        SetMasknorm();
        SetVectorMask<uint8_t>((uint64_t)-1, (uint64_t)-1);
        if (tilingData.swizzlDirect == 0 && tilingData.coreLoop <= tilingData.blockDim) {
            matmul_0000.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_0000.Run();
        } else if (tilingData.swizzlDirect == 1 && tilingData.coreLoop <= tilingData.blockDim) {
            matmul_1000.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_1000.Run();
        } else if (tilingData.swizzlDirect == 0 && tilingData.coreLoop > tilingData.blockDim) {
            matmul_0001.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_0001.Run();
        } else if (tilingData.swizzlDirect == 1 && tilingData.coreLoop > tilingData.blockDim) {
            matmul_1001.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_1001.Run();
        }
    } else if (TILING_KEY_IS(5608)) { // 0b0'01'01'01'1'1'1'0'1'000
        PpMatmulF16NZ<0, false, true, half, half, float, false> matmul_0010;
        PpMatmulF16NZ<1, false, true, half, half, float, false> matmul_1010;
        PpMatmulF16NZ<0, false, true, half, half, float, true> matmul_0011;
        PpMatmulF16NZ<1, false, true, half, half, float, true> matmul_1011;
        SetPadding<uint64_t>(0);
        SetMasknorm();
        SetVectorMask<uint8_t>((uint64_t)-1, (uint64_t)-1);
        if (tilingData.swizzlDirect == 0 && tilingData.coreLoop <= tilingData.blockDim) {
            matmul_0010.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_0010.Run();
        } else if (tilingData.swizzlDirect == 1 && tilingData.coreLoop <= tilingData.blockDim) {
            matmul_1010.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_1010.Run();
        } else if (tilingData.swizzlDirect == 0 && tilingData.coreLoop > tilingData.blockDim) {
            matmul_0011.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_0011.Run();
        } else if (tilingData.swizzlDirect == 1 && tilingData.coreLoop > tilingData.blockDim) {
            matmul_1011.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_1011.Run();
        }
    } else if (TILING_KEY_IS(5616)) { // 0b0'01'01'01'1'1'1'1'0'000
        PpMatmulF16NZ<0, true, false, half, half, float, false> matmul_0100;
        PpMatmulF16NZ<1, true, false, half, half, float, false> matmul_1100;
        PpMatmulF16NZ<0, true, false, half, half, float, true> matmul_0101;
        PpMatmulF16NZ<1, true, false, half, half, float, true> matmul_1101;
        SetPadding<uint64_t>(0);
        SetMasknorm();
        SetVectorMask<uint8_t>((uint64_t)-1, (uint64_t)-1);
        if (tilingData.swizzlDirect == 0 && tilingData.coreLoop <= tilingData.blockDim) {
            matmul_0100.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_0100.Run();
        } else if (tilingData.swizzlDirect == 1 && tilingData.coreLoop <= tilingData.blockDim) {
            matmul_1100.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_1100.Run();
        } else if (tilingData.swizzlDirect == 0 && tilingData.coreLoop > tilingData.blockDim) {
            matmul_0101.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_0101.Run();
        } else if (tilingData.swizzlDirect == 1 && tilingData.coreLoop > tilingData.blockDim) {
            matmul_1101.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_1101.Run();
        }

    } else if (TILING_KEY_IS(5624)) { // 0b0'01'01'01'1'1'1'1'1'000
        PpMatmulF16NZ<0, true, true, half, half, float, false> matmul_0110;
        PpMatmulF16NZ<1, true, true, half, half, float, false> matmul_1110;
        PpMatmulF16NZ<0, true, true, half, half, float, true> matmul_0111;
        PpMatmulF16NZ<1, true, true, half, half, float, true> matmul_1111;
        SetPadding<uint64_t>(0);
        SetMasknorm();
        SetVectorMask<uint8_t>((uint64_t)-1, (uint64_t)-1);
        if (tilingData.swizzlDirect == 0 && tilingData.coreLoop <= tilingData.blockDim) {
            matmul_0110.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_0110.Run();
        } else if (tilingData.swizzlDirect == 1 && tilingData.coreLoop <= tilingData.blockDim) {
            matmul_1110.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_1110.Run();
        } else if (tilingData.swizzlDirect == 0 && tilingData.coreLoop > tilingData.blockDim) {
            matmul_0111.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_0111.Run();
        } else if (tilingData.swizzlDirect == 1 && tilingData.coreLoop > tilingData.blockDim) {
            matmul_1111.SetArgs(gm_a, gm_b, gm_c, &tilingData);
            matmul_1111.Run();
        }
    }
#endif
#endif
}