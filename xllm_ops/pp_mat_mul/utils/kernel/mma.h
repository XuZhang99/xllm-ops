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
 * \file mma.h
 * \brief
 */

#ifndef INCLUDE_MMA_H
#define INCLUDE_MMA_H

#include "hardware.h"
#include "kernel_tensor.h"

namespace PpMatMulNS {
template <ArchType ArchTag, typename ElementA, typename ElementB, typename AccDTypeC, bool IsTransposeA>
struct mmad {
    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor,
                    AscendC::LocalTensor<ElementA> l0aTensor,
                    AscendC::LocalTensor<ElementB> l0bTensor,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool initC,
                    uint8_t unitFlag = 0) {};

    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor,
                    AscendC::LocalTensor<ElementA> l0aTensor,
                    AscendC::LocalTensor<ElementB> l0bTensor,
                    uint64_t biasBt,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool initC,
                    uint8_t unitFlag = 0) {};
};

// Partial specialization for V220, int8_t, not_vector_A, not TransposeA
template <ArchType ArchTag, typename AccDTypeC, typename ElementA, typename ElementB>
struct mmad<ArchTag, ElementA, ElementB, AccDTypeC, false> {
    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor,
                    AscendC::LocalTensor<ElementA> l0aTensor,
                    AscendC::LocalTensor<ElementB> l0bTensor,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool initC,
                    uint8_t unitFlag = 0)
    {
        AscendC::Mmad(l0cTensor,                       // C
                      l0aTensor,                       // A
                      l0bTensor,                       // B
                      AscendC::MmadParams(mTileActual, // m
                                          nTileActual, // n
                                          kPartActual, // k
                                          unitFlag,    // unitFlag
                                          false,       // cmatrixSource
                                          initC));     // cmatrixInitVal
    };

    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor,
                    AscendC::LocalTensor<ElementA> l0aTensor,
                    AscendC::LocalTensor<ElementB> l0bTensor,
                    uint64_t biasBt,
                    uint32_t mTileActual,
                    uint32_t nTileActual,
                    uint32_t kPartActual,
                    bool initC,
                    uint8_t unitFlag = 0)
    {
        AscendC::LocalTensor<AccDTypeC> biasTensor;
        biasTensor.InitBuffer(biasBt, nTileActual);
        biasTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2);
        AscendC::Mmad(l0cTensor,                       // C
                      l0aTensor,                       // A
                      l0bTensor,                       // B
                      biasTensor,                      // bt
                      AscendC::MmadParams(mTileActual, // m
                                          nTileActual, // n
                                          kPartActual, // k
                                          unitFlag,    // unitFlag
                                          true,        // cmatrixSource
                                          false));     // cmatrixInitVal
    };
};

}
#endif