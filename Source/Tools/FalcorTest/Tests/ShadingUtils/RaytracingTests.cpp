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
#include <random>
#include <glm/gtx/io.hpp>

namespace Falcor
{
    namespace
    {
        int float_as_int(float f) { return *reinterpret_cast<int*>(&f); }
        float int_as_float(int i) { return *reinterpret_cast<float*>(&i); }

        /** Unmodified reference code from Ray Tracing Gems, Chapter 6.
        */
        constexpr float origin() { return 1.0f / 32.0f; }
        constexpr float float_scale() { return 1.0f / 65536.0f; }
        constexpr float int_scale() { return 256.0f; }

        // Normal points outward for rays exiting the surface, else is flipped.
        float3 offset_ray(const float3 p, const float3 n)
        {
            int3 of_i(int_scale() * n.x, int_scale() * n.y, int_scale() * n.z);

            float3 p_i(
                int_as_float(float_as_int(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                int_as_float(float_as_int(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                int_as_float(float_as_int(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

            return float3(fabsf(p.x) < origin() ? p.x + float_scale()*n.x : p_i.x,
                fabsf(p.y) < origin() ? p.y + float_scale()*n.y : p_i.y,
                fabsf(p.z) < origin() ? p.z + float_scale()*n.z : p_i.z);
        }
    }

    GPU_TEST(ComputeRayOrigin)
    {
        const uint32_t n = 1 << 16;

        std::mt19937 rng;
        auto dist = std::uniform_real_distribution<float>(-1.f, 1.f);
        auto r = [&]() -> float { return dist(rng); };

        // Create random test data.
        std::vector<float3> testPositions(n);
        std::vector<float3> testNormals(n);
        for (uint32_t i = 0; i < n; i++)
        {
            float scale = std::pow(10.f, (float)i / n * 60.f - 30.f); // 1e-30..1e30
            testPositions[i] = float3(r(), r(), r()) * scale;
            testNormals[i] = glm::normalize(float3(r(), r(), r()));
        }

        // Setup and run GPU test.
        ctx.createProgram("Tests/ShadingUtils/RaytracingTests.cs.slang", "testComputeRayOrigin", {{"SCENE_MATERIAL_COUNT", "1"}, {"SCENE_GRID_COUNT", "0"}});
        ctx.allocateStructuredBuffer("result", n);
        // TODO: Cleanup when !122 is merged
        //ctx.allocateStructuredBuffer("pos", n, testPositions, testPositions.size() * sizeof(float3));
        //ctx.allocateStructuredBuffer("normal", n, testNormals.size() * sizeof(float3));
        auto pPos = Buffer::createStructured(ctx.getProgram(), "pos", n);
        pPos->setBlob(testPositions.data(), 0, testPositions.size() * sizeof(float3));
        ctx["pos"] = pPos;
        auto pNormal = Buffer::createStructured(ctx.getProgram(), "normal", n);
        pNormal->setBlob(testNormals.data(), 0, testNormals.size() * sizeof(float3));
        ctx["normal"] = pNormal;

        ctx["CB"]["n"] = n;
        ctx.runProgram(n);

        // Verify results.
        const float3* result = ctx.mapBuffer<const float3>("result");
        for (uint32_t i = 0; i < n; i++)
        {
            float3 ref = offset_ray(testPositions[i], testNormals[i]);
            EXPECT_EQ(result[i], ref) << "i = " << i;
        }
        ctx.unmapBuffer("result");
    }
}
