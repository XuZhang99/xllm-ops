/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>

namespace AtbOps {
namespace Utils {
template<class T>
class Compare {
public:
    static bool IsEqual(const T &lh, const T &rh)
    {
        return lh == rh;
    }
};

template<>
class Compare<float> {
public:
    static bool IsEqual(const float &lh, const float &rh)
    {
        return std::abs(lh - rh) < 0.000001; // float类型精度0.000001
    }
};

template<>
class Compare<double> {
public:
    static bool IsEqual(const double &lh, const double &rh)
    {
        return std::abs(lh - rh) < 0.000000001; // double类型精度0.000000001
    }
};
} // namespace AtbOps
} //Utils
