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
 * \file l1_to_bt_iterator.h
 * \brief
 */
#ifndef L1_TO_BT_ITERATOR_H
#define L1_TO_BT_ITERATOR_H

/////////////////////////////////////////////////////
// l1_to_bt
/////////////////////////////////////////////////////

// Partial specialization for V220
template <typename DataType>
struct l1_to_bt<ArchType::ASCEND_V220, DataType> {
    __aicore__ l1_to_bt(uint64_t dst,
                        const AscendC::LocalTensor<DataType> &src,
                        uint16_t convControl,
                        uint16_t nBurst,
                        uint16_t lenBurst,
                        uint16_t srcGap,
                        uint16_t dstGap)
    {
        AscendC::LocalTensor<DataType> dstTensor;
        dstTensor.InitBuffer(dst, nBurst * lenBurst);
        dstTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2);
        AscendC::DataCopy(dstTensor,
                          src,
                          AscendC::DataCopyParams(nBurst,   // nBurst
                                                  lenBurst, // lenBurst
                                                  srcGap,   // srcGap
                                                  dstGap)); // dstGap
    }
};

#endif