/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

namespace Falcor
{
    namespace
    {
        const uint32_t kNumElems = 32 * 128;
        std::mt19937 rng;

        std::vector<uint32_t> generateMatchData(size_t numElems, size_t numUnique = 20)
        {
            std::vector<uint32_t> elems(numUnique);
            for (auto& e : elems) e = rng();

            std::uniform_int_distribution<size_t> select(0, numUnique-1);
            std::vector<uint32_t> data(numElems);
            for (auto& d : data) d = elems[select(rng)];

            return data;
        }

        std::vector<uint4> computeMatchResult(const std::vector<uint32_t>& data, uint32_t laneCount)
        {
            assert(laneCount >= 4 && laneCount <= 128);
            std::vector<uint4> masks(data.size(), uint4(0));

            for (size_t i = 0; i < data.size(); i++)
            {
                size_t firstLane = (i / laneCount) * laneCount;
                uint32_t currentLaneValue = data[i];
                uint4& mask = masks[i];

                for (uint32_t j = 0; j < laneCount; j++)
                {
                    if (data[firstLane + j] == currentLaneValue)
                    {
                        mask[j >> 5] |= (1u << (j & 0x1f));
                    }
                }
            }

            return masks;
        }
    }

    GPU_TEST(WaveGetLaneCount)
    {
        ctx.createProgram("Tests/Slang/WaveOps.cs.slang", "testWaveGetLaneCount", Program::DefineList(), Shader::CompilerFlags::None, "6_0");

        auto var = ctx.vars().getRootVar();
        uint32_t zero = 0;
        auto pLaneCount = Buffer::createTyped<uint32_t>(1, ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, &zero);
        var["laneCount"] = pLaneCount;

        ctx.runProgram(1, 1, 1);

        const uint32_t laneCount = *(const uint32_t*)pLaneCount->map(Buffer::MapType::Read);
        pLaneCount->unmap();
        EXPECT_GE(laneCount, 4u);
        EXPECT_LE(laneCount, 128u);
    }

    GPU_TEST(WaveMatch, "Requires shader model 6.5")
    {
        ctx.createProgram("Tests/Slang/WaveOps.cs.slang", "testWaveMatch", Program::DefineList(), Shader::CompilerFlags::None, "6_5");
        ctx.allocateStructuredBuffer("result", kNumElems);

        auto var = ctx.vars().getRootVar();
        uint32_t zero = 0;
        auto pLaneCount = Buffer::createTyped<uint32_t>(1, ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, &zero);
        var["laneCount"] = pLaneCount;

        std::vector<uint32_t> matchData = generateMatchData(kNumElems);
        assert(matchData.size() == kNumElems);
        var["testData"] = Buffer::createTyped<uint32_t>(kNumElems, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, matchData.data());

        ctx.runProgram(kNumElems, 1, 1);

        // Get the lane count. We abort the test if it is an unsupported count.
        const uint32_t laneCount = *(const uint32_t*)pLaneCount->map(Buffer::MapType::Read);
        pLaneCount->unmap();
        if (laneCount < 4 || laneCount > 128) throw std::exception("Unsupported wave lane count");

        // Verify results of wave match.
        std::vector<uint4> expectedResult = computeMatchResult(matchData, laneCount);
        assert(expectedResult.size() == matchData.size());

        const uint4* result = ctx.mapBuffer<const uint4>("result");
        for (size_t i = 0; i < matchData.size(); i++)
        {
            EXPECT_EQ(result[i].x, expectedResult[i].x) << "i = " << i;
            EXPECT_EQ(result[i].y, expectedResult[i].y) << "i = " << i;
            EXPECT_EQ(result[i].z, expectedResult[i].z) << "i = " << i;
            EXPECT_EQ(result[i].w, expectedResult[i].w) << "i = " << i;
        }
        ctx.unmapBuffer("result");
    }
}
