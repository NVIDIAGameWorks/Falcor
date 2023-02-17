/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Testing/UnitTest.h"
#include "Utils/Math/Matrix/Matrix.h"

#include <fmt/format.h>
#include <iostream>

namespace Falcor
{

CPU_TEST(Matrix_FloatFormatter)
{
    rmcv::mat3x3 test0({1.1f, 1.2f, 1.3f, 2.1f, 2.2f, 2.3f, 3.1f, 3.2f, 3.3f});

    EXPECT_EQ(fmt::format("{}", test0), "{{1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}, {3.1, 3.2, 3.3}}");
    EXPECT_EQ(
        fmt::format("{:e}", test0),
        "{{1.100000e+00, 1.200000e+00, 1.300000e+00}, "
        "{2.100000e+00, 2.200000e+00, 2.300000e+00}, "
        "{3.100000e+00, 3.200000e+00, 3.300000e+00}}"
    );
    EXPECT_EQ(fmt::format("{:g}", test0), "{{1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}, {3.1, 3.2, 3.3}}");
    EXPECT_EQ(fmt::format("{:.1}", test0), "{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}");
    EXPECT_EQ(fmt::format("{:.2f}", test0), "{{1.10, 1.20, 1.30}, {2.10, 2.20, 2.30}, {3.10, 3.20, 3.30}}");
}

} // namespace Falcor
