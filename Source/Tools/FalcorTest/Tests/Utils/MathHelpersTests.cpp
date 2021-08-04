/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace Falcor
{
    GPU_TEST(SphericalCoordinates)
    {
        ctx.createProgram("Tests/Utils/MathHelpersTests.cs.slang", "testSphericalCoordinates");
        constexpr int32_t n = 1024 * 1024;
        ctx.allocateStructuredBuffer("result", n);
        // The shader runs threadgroups of 1024 threads.
        ctx.runProgram(n);

        // The shader generates a bunch of random vectors, converts them to
        // spherical coordinates and back, and computes the dot product with
        // the original vector.  Here, we'll check that the dot product is
        // pretty close to one.
        const float* r = ctx.mapBuffer<const float>("result");
        for (int32_t i = 0; i < n; ++i)
        {
            EXPECT_GT(r[i], .999f) << "i = " << i;
            EXPECT_LT(r[i], 1.001f) << "i = " << i;
        }
        ctx.unmapBuffer("result");
    }
}
