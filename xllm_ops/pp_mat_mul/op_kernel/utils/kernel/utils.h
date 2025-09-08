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
 * \file utils.h
 * \brief
 */

#ifndef INCLUDE_UTILS_H
#define INCLUDE_UTILS_H

namespace PpMatMulNS {
template <typename IN_DTYPE>
__aicore__ inline void CreateCaMatrix(const AscendC::LocalTensor<IN_DTYPE> &dst,
                                      const uint16_t repeats,
                                      const uint16_t blockNum,
                                      const uint16_t dstGap,
                                      const IN_DTYPE initValue)
{
    AscendC::InitConstValue<IN_DTYPE>(dst,
                                      AscendC::InitConstValueParams<IN_DTYPE>(repeats, blockNum, dstGap, initValue));
}
__aicore__ inline void SetFftsBaseAddr(uint64_t config)
{
    AscendC::SetSyncBaseAddr(config);
}
template <typename IN_DTYPE>
__aicore__ inline void SetPadding(IN_DTYPE padValue)
{
    AscendC::SetLoadDataPaddingValue<IN_DTYPE>(padValue);
}
__aicore__ inline void SetAtomicnone()
{
    AscendC::SetAtomicNone();
}
__aicore__ inline void SetMasknorm()
{
#if __CCE_AICORE__ == 100
    return;
#endif
    AscendC::SetMaskNorm();
}
__aicore__ inline void SetNdpara(uint16_t ndNum, uint16_t srcNdStride, uint16_t dstNdStride)
{
    AscendC::SetFixpipeNz2ndFlag(ndNum, srcNdStride, dstNdStride);
}
template <typename IN_DTYPE>
__aicore__ inline void SetVectorMask(const uint64_t maskHigh, const uint64_t maskLow)
{
    AscendC::SetVectorMask<IN_DTYPE>(maskHigh, maskLow);
}
__aicore__ inline int64_t GetSubBlockidx()
{
    return AscendC::GetSubBlockIdx();
}
__aicore__ inline void WaitFlagDev(uint16_t flagId)
{
    AscendC::WaitEvent(flagId);
}
template <pipe_t pipe, uint8_t mode>
__aicore__ inline void FftsCrossCoreSync(uint16_t flagId)
{
    AscendC::CrossCoreSetFlag<mode, pipe>(flagId);
}
template <typename IN_DTYPE, bool setRelu = false>
__aicore__ inline void SetFpc(const AscendC::LocalTensor<IN_DTYPE> &preTensor, bool isUnitFlag = false)
{
    AscendC::SetFixPipeConfig<IN_DTYPE, setRelu>(preTensor, isUnitFlag);
}

}
#endif