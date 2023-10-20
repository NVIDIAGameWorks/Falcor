/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/Debug/WarpProfiler.h"

namespace Falcor
{
GPU_TEST(WarpProfiler, Device::Type::D3D12)
{
    WarpProfiler profiler(ctx.getDevice(), 4);

    ProgramDesc desc;
    desc.addShaderLibrary("Tests/Utils/Debug/WarpProfilerTests.cs.slang").csEntry("main");
    desc.setShaderModel(ShaderModel::SM6_5); // Minimum required shader model.
    ctx.createProgram(desc);

    auto var = ctx.vars().getRootVar();
    profiler.bindShaderData(var);
    profiler.begin(ctx.getRenderContext());

    ctx.runProgram(256, 256, 16); // Launch 2^20 threads = 32768 warps.

    profiler.end(ctx.getRenderContext());

    // Utilization
    {
        auto histogram = profiler.getWarpHistogram(0);
        EXPECT_EQ(histogram.size(), WarpProfiler::kWarpSize);

        size_t warpCount = 0;
        for (auto h : histogram)
        {
            warpCount += h;
        }
        EXPECT_EQ(histogram[31], 32768);
        EXPECT_EQ(warpCount, 32768);
    }

    {
        auto histogram = profiler.getWarpHistogram(1);
        EXPECT_EQ(histogram.size(), WarpProfiler::kWarpSize);

        size_t warpCount = 0;
        for (auto h : histogram)
        {
            warpCount += h;
        }
        EXPECT_EQ(histogram[7], 16384);
        EXPECT_EQ(warpCount, 16384);
    }

    {
        auto histogram = profiler.getWarpHistogram(0, 2);
        EXPECT_EQ(histogram.size(), WarpProfiler::kWarpSize);

        EXPECT_EQ(histogram[7], 16384);
        EXPECT_EQ(histogram[31], 32768);
    }

    // Divergence
    {
        auto histogram = profiler.getWarpHistogram(2);
        EXPECT_EQ(histogram.size(), WarpProfiler::kWarpSize);

        size_t warpCount = 0;
        for (auto h : histogram)
        {
            warpCount += h;
        }
        EXPECT_EQ(histogram[3], 32768);
        EXPECT_EQ(warpCount, 32768);
    }

    {
        auto histogram = profiler.getWarpHistogram(3);
        EXPECT_EQ(histogram.size(), WarpProfiler::kWarpSize);

        size_t warpCount = 0;
        for (auto h : histogram)
        {
            warpCount += h;
        }
        EXPECT_EQ(histogram[7], 8192);
        EXPECT_EQ(warpCount, 8192);
    }
}
} // namespace Falcor
