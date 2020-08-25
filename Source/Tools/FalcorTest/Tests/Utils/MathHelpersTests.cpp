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
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace Falcor
{
    namespace
    {
        struct BBoxTestCase
        {
            float3 origin, aabbMin, aabbMax;
            float angle;
        };

        void runBBoxTestComputeShader(GPUUnitTestContext& ctx, const BBoxTestCase* testCases, int nTests, const char* entrypoint)
        {
            Buffer::SharedPtr pOriginBuffer = Buffer::createTyped<float3>(nTests);
            Buffer::SharedPtr pAABBMinBuffer = Buffer::createTyped<float3>(nTests);
            Buffer::SharedPtr pAABBMaxBuffer = Buffer::createTyped<float3>(nTests);

            for (int i = 0; i < nTests; ++i)
            {
                pOriginBuffer->setElement(i, testCases[i].origin);
                pAABBMinBuffer->setElement(i, testCases[i].aabbMin);
                pAABBMaxBuffer->setElement(i, testCases[i].aabbMax);
            }

            // Setup and run GPU test.
            ctx.createProgram("Tests/Utils/MathHelpersTests.cs.slang", entrypoint);
            ctx["origin"] = pOriginBuffer;
            ctx["aabbMin"] = pAABBMinBuffer;
            ctx["aabbMax"] = pAABBMaxBuffer;
            ctx.allocateStructuredBuffer("sinTheta", nTests);
            ctx.allocateStructuredBuffer("cosTheta", nTests);
            ctx.allocateStructuredBuffer("coneDir", nTests);
            ctx.runProgram(nTests);
        }

        void testKnownBBoxes(GPUUnitTestContext& ctx, const char* entrypoint)
        {
            // Generate test data...
            BBoxTestCase testCases[] =
            {
                 // Unit box centered at z=2 -> then tan(theta) is the ratio of
                 // half of the length of the diagonal of a box face (aka
                 // sqrt(2) / 2), divided by the distance from |origin| to the
                 // box (aka 2).
                 { float3(0., 0., 0.), float3(-0.5f, -0.5f, 2.f), float3(0.5f, 0.5f, 3.f), std::atan2(std::sqrt(2.f) / 2.f, 2.f) },
                 // Point is inside the box
                 { float3(0.5f, 10.f, -20.f), float3(-.25f, 5.f, -22.f), float3(3.f, 17.f, 29.f), static_cast<float>(M_PI) }
            };
            int nTests = sizeof(testCases) / sizeof(testCases[0]);

            runBBoxTestComputeShader(ctx, testCases, nTests, entrypoint);

            const float* sinTheta = ctx.mapBuffer<const float>("sinTheta");
            const float* cosTheta = ctx.mapBuffer<const float>("cosTheta");

            for (int i = 0; i < nTests; ++i)
            {
                const BBoxTestCase& tc = testCases[i];
                if (tc.angle == static_cast<float>(M_PI))
                {
                    // Expect to get it exact for points inside the box.
                    EXPECT_EQ(sinTheta[i], 0.f);
                    EXPECT_EQ(cosTheta[i], -1.f);
                }
                else
                {
                    const float eps = 1e-4f;
                    EXPECT(std::sin(testCases[i].angle) > (1.f - eps) * sinTheta[i] &&
                           std::sin(testCases[i].angle) < (1.f + eps) * sinTheta[i]) <<
                      "BBoxTestCase " << i << ", expected sin(theta) = " << std::sin(testCases[i].angle) << ", got " << sinTheta[i];
                    EXPECT(std::cos(testCases[i].angle) > (1.f - eps) * cosTheta[i] &&
                           std::cos(testCases[i].angle) < (1.f + eps) * cosTheta[i]) <<
                      "BBoxTestCase " << i << ", expected cos(theta) = " << std::cos(testCases[i].angle) << ", got " << cosTheta[i];
                }
            }

            ctx.unmapBuffer("cosTheta");
            ctx.unmapBuffer("sinTheta");
        }

        void testRandomBBoxes(GPUUnitTestContext& ctx, const char* entrypoint)
        {
            // Generate test data.
            std::vector<BBoxTestCase> testCases;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> posAndNegDist(-100.f, 100.f);
            std::uniform_real_distribution<> posDist(1e-4f, 100.f);
            for (int i = 0; i < 256; ++i)
            {
                // Random origins and bounding boxes.
                float3 origin(posAndNegDist(gen), posAndNegDist(gen), posAndNegDist(gen));
                float3 aabbMin(posAndNegDist(gen), posAndNegDist(gen), posAndNegDist(gen));
                float3 aabbMax = aabbMin + float3(posDist(gen), posDist(gen), posDist(gen));

                testCases.push_back({origin, aabbMin, aabbMax, 0.f}); // angle is unused in this test.
            }

            runBBoxTestComputeShader(ctx, testCases.data(), static_cast<int>(testCases.size()), entrypoint);

            const float* sinTheta = ctx.mapBuffer<const float>("sinTheta");
            const float* cosTheta = ctx.mapBuffer<const float>("cosTheta");
            const float3* coneDir = ctx.mapBuffer<const float3>("coneDir");

            for (size_t i = 0; i < testCases.size(); ++i)
            {
                const BBoxTestCase &b = testCases[i];
                bool inside = (b.origin.x >= b.aabbMin.x && b.origin.x <= b.aabbMax.x &&
                               b.origin.y >= b.aabbMin.y && b.origin.y <= b.aabbMax.y &&
                               b.origin.z >= b.aabbMin.z && b.origin.z <= b.aabbMax.z);
                if (inside)
                {
                    // Expect to get it exact for points inside the box.
                    EXPECT_EQ(sinTheta[i], 0.f);
                    EXPECT_EQ(cosTheta[i], -1.f);
                }
                else
                {
                  float minCosTheta = 1.f;
                  for (int j = 0; j < 8; ++j)
                    {
                      // Make sure that the vector to AABB corner is inside the cone.
                      float3 corner = float3((j & 1) ? b.aabbMin.x : b.aabbMax.x,
                                             (j & 2) ? b.aabbMin.y : b.aabbMax.y,
                                             (j & 4) ? b.aabbMin.z : b.aabbMax.z);
                      float3 v = normalize(corner - b.origin);
                      float ct = dot(v, normalize(coneDir[i]));
                      EXPECT_GT(ct, (cosTheta[i] > 0 ? (0.99f * cosTheta[i]) : (1.01f * cosTheta[i])));
                      minCosTheta = std::min(minCosTheta, ct);
                    }
                  // Make sure that the maximum angle between a vector to an AABB corner and the
                  // cone axis isn't much bigger than the reported cone axis.
                  EXPECT_LT(minCosTheta, (cosTheta[i] > 0 ? (1.01f * cosTheta[i]) : (0.99f * cosTheta[i])));
                }
            }

            ctx.unmapBuffer("cosTheta");
            ctx.unmapBuffer("sinTheta");
        }
    }

    GPU_TEST(BoxSubtendedConeAngleCenter)
    {
        testKnownBBoxes(ctx, "testBoundingConeAngleCenter");
    }

    GPU_TEST(BoxSubtendedConeAngleAverage)
    {
        testKnownBBoxes(ctx, "testBoundingConeAngleAverage");
    }

    // Disable this test for now: it turns out that this bounding method returns
    // cos(theta) = -1 for points that are close enough to the bounding box that
    // their cos(theta) value is < 0.
    GPU_TEST(BoxSubtendedConeAngleCenterRandoms, "Disabled as bounding cone is over-conservative (#699)")
    {
        testRandomBBoxes(ctx, "testBoundingConeAngleCenter");
    }

    GPU_TEST(BoxSubtendedConeAngleAverageRandoms)
    {
        testRandomBBoxes(ctx, "testBoundingConeAngleAverage");
    }

    GPU_TEST(SphereSubtendedAngle)
    {
        // Generate test data...
        struct TestCase
        {
            float3 origin;
            float radius;
            float angle;
        };
        TestCase testCases[] =
        {
           // sin(theta) between a vector to the center of the sphere and a
           // vector that is tangent to the sphere is the sphere radius
           // divided by the distance from the starting point to the center
           // of the sphere.
           { float3(0.f, 0.f, 2.f), 1.f, std::asin(1.f / 2.f) },
           { float3(10.f, -5.f, 2.f), 0.1f, std::asin(0.1f / std::sqrt(10.f*10.f + 5.f*5.f + 2.f*2.f)) },
           // Point inside the sphere.
           { float3(0.5f, 0.f, 0.f), 0.51f, static_cast<float>(M_PI) }
        };
        int nTests = sizeof(testCases) / sizeof(testCases[0]);

        Buffer::SharedPtr pTestCaseBuffer = Buffer::createTyped<float4>(nTests);

        for (int i = 0; i < nTests; ++i)
        {
            // Pack sphere origins and radii into float4s.
            pTestCaseBuffer->setElement(i, float4(testCases[i].origin, testCases[i].radius));
        }

        // Setup and run GPU test.
        ctx.createProgram("Tests/Utils/MathHelpersTests.cs.slang", "testBoundSphereAngle");
        ctx["spheres"] = pTestCaseBuffer;
        ctx.allocateStructuredBuffer("sinTheta", nTests);
        ctx.allocateStructuredBuffer("cosTheta", nTests);
        ctx.runProgram(nTests);

        const float* sinTheta = ctx.mapBuffer<const float>("sinTheta");
        const float* cosTheta = ctx.mapBuffer<const float>("cosTheta");

        for (int i = 0; i < nTests; ++i)
        {
            const TestCase& tc = testCases[i];
            if (tc.angle == static_cast<float>(M_PI))
            {
                // Expect to get it exact for points inside the sphere.
                EXPECT_EQ(sinTheta[i], 0.f);
                EXPECT_EQ(cosTheta[i], -1.f);
            }
            else
            {
                const float eps = 1e-4f;
                EXPECT(std::sin(testCases[i].angle) > (1.f - eps) * sinTheta[i] &&
                       std::sin(testCases[i].angle) < (1.1 + eps) * sinTheta[i]) <<
                  "Expected sin(theta) = " << std::sin(testCases[i].angle) << ", got " << sinTheta[i] <<
                  ", for sphere at (" << tc.origin.x << ", " << tc.origin.y << ", " << tc.origin.z <<
                  "), radius " << tc.radius;
                EXPECT(std::cos(testCases[i].angle) > (1.f - eps) * cosTheta[i] &&
                       std::cos(testCases[i].angle) < (1.f + eps) * cosTheta[i]) <<
                  "Expected cos(theta) = " << std::cos(testCases[i].angle) << ", got " << cosTheta[i] <<
                  ", for sphere at (" << tc.origin.x << ", " << tc.origin.y << ", " << tc.origin.z <<
                  "), radius " << tc.radius;
            }
        }

        ctx.unmapBuffer("cosTheta");
        ctx.unmapBuffer("sinTheta");
    }
}
