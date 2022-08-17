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
#include "Utils/HostDeviceShared.slangh"
#include "Utils/Math/Common.h"
#include <glm/gtx/io.hpp>
#include <random>
#include <cmath>

namespace Falcor
{
    namespace
    {
        const char kShaderFilename[] = "Tests/Utils/GeometryHelpersTests.cs.slang";

        /** Modified reference code from Ray Tracing Gems, Chapter 6.

            Modified after discussion with Carsten Waechter, see comment in the implementation
         */
        constexpr float origin() { return 1.0f / 16.0f; }
        constexpr float float_scale() { return 3.0f / 65536.0f; }
        constexpr float int_scale() { return 3 * 256.0f; }

        // Normal points outward for rays exiting the surface, else is flipped.
        float3 offset_ray(const float3 p, const float3 n)
        {
            int3 of_i(int_scale() * n.x, int_scale() * n.y, int_scale() * n.z);

            float3 p_i(
                asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
                asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
                asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));

            return float3(fabsf(p.x) < origin() ? p.x + float_scale()*n.x : p_i.x,
                fabsf(p.y) < origin() ? p.y + float_scale()*n.y : p_i.y,
                fabsf(p.z) < origin() ? p.z + float_scale()*n.z : p_i.z);
        }

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
            ctx.createProgram(kShaderFilename, entrypoint);
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
            const int nTests = 1 << 16;

            // Generate test data.
            std::vector<BBoxTestCase> testCases;
            std::mt19937 gen;
            std::uniform_real_distribution<> posAndNegDist(-100.f, 100.f);
            std::uniform_real_distribution<> posDist(1e-4f, 100.f);

            for (int i = 0; i < nTests; ++i)
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

    GPU_TEST(ComputeRayOrigin)
    {
        const uint32_t nTests = 1 << 16;

        std::mt19937 rng;
        auto dist = std::uniform_real_distribution<float>(-1.f, 1.f);
        auto r = [&]() -> float { return dist(rng); };

        // Create random test data.
        std::vector<float3> testPositions(nTests);
        std::vector<float3> testNormals(nTests);
        for (uint32_t i = 0; i < nTests; i++)
        {
            float scale = std::pow(10.f, (float)i / nTests * 60.f - 30.f); // 1e-30..1e30
            testPositions[i] = float3(r(), r(), r()) * scale;
            testNormals[i] = glm::normalize(float3(r(), r(), r()));
        }

        // Setup and run GPU test.
        ctx.createProgram(kShaderFilename, "testComputeRayOrigin", Program::DefineList(), Shader::CompilerFlags::FloatingPointModePrecise);
        ctx.allocateStructuredBuffer("result", nTests);
        ctx.allocateStructuredBuffer("pos", nTests, testPositions.data(), testPositions.size() * sizeof(float3));
        ctx.allocateStructuredBuffer("normal", nTests, testNormals.data(), testNormals.size() * sizeof(float3));
        ctx["CB"]["n"] = nTests;
        ctx.runProgram(nTests);

        // Verify results.
        const float3* result = ctx.mapBuffer<const float3>("result");
        for (uint32_t i = 0; i < nTests; i++)
        {
            float3 ref = offset_ray(testPositions[i], testNormals[i]);
            EXPECT_EQ(result[i], ref) << "i = " << i;
        }
        ctx.unmapBuffer("result");
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
        ctx.createProgram(kShaderFilename, "testBoundSphereAngle");
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

    GPU_TEST(ComputeClippedTriangleArea2D)
    {
        struct TestCase
        {
            float2 p[3];
            float area; // Expected clipped area.

            // Returns signed triangle area. Positive for CCW winding.
            float triangleArea() const
            {
                float A = -p[1].y * p[2].x + p[0].y * (p[2].x - p[1].x) + p[0].x * (p[1].y - p[2].y) + p[1].x * p[2].y;
                return 0.5f * A;
            }

            // Returns true if point 'v' is inside the triangle (irrespective of winding).
            bool isInside(const float2& v) const
            {
                float s = p[0].y * p[2].x - p[0].x * p[2].y + (p[2].y - p[0].y) * v.x + (p[0].x - p[2].x) * v.y;
                float t = p[0].x * p[1].y - p[0].y * p[1].x + (p[0].y - p[1].y) * v.x + (p[1].x - p[0].x) * v.y;
                if ((s < 0.f) != (t < 0.f)) return false;

                float A = -p[1].y * p[2].x + p[0].y * (p[2].x - p[1].x) + p[0].x * (p[1].y - p[2].y) + p[1].x * p[2].y;
                return A < 0 ?
                    (s <= 0.f && s + t >= A) :
                    (s >= 0.f && s + t <= A);
            }
        };

        struct AABB2D
        {
            float2 minPos;
            float2 maxPos;

            float area() const
            {
                float2 extent = maxPos - minPos;
                return extent.x * extent.y;
            }
        };

        std::vector<TestCase> testsFixed
        {
            // The box we test against is assumed to be at x=[1,2] and y=[2,3].
            // All coordinates and areas below are exactly representable in floating-point numbers.

            // Degenerate triangles (point)
            {{ float2(1.50f, 2.50f), float2(1.50f, 2.50f), float2(1.50f, 2.50f) }, 0.f},        // Inside box
            {{ float2(2.00f, 2.50f), float2(2.00f, 2.50f), float2(2.00f, 2.50f) }, 0.f},        // On edge
            {{ float2(3.00f, 2.50f), float2(3.00f, 2.50f), float2(3.00f, 2.50f) }, 0.f},        // Outside box

            // Degenerate triangles (lines)
            {{ float2(1.25f, 2.75f), float2(1.50f, 2.25f), float2(1.25f, 2.75f) }, 0.f},        // Inside box
            {{ float2(1.50f, 2.75f), float2(1.50f, 2.25f), float2(1.50f, 2.50f) }, 0.f},        // Inside box
            {{ float2(1.00f, 2.75f), float2(2.00f, 2.50f), float2(1.00f, 2.75f) }, 0.f},        // Touching edge
            {{ float2(1.25f, 3.00f), float2(1.50f, 3.00f), float2(1.75f, 3.00f) }, 0.f},        // Along edge
            {{ float2(0.50f, 3.00f), float2(2.00f, 1.00f), float2(0.50f, 3.00f) }, 0.f},        // Intersecting box
            {{ float2(0.50f, 3.00f), float2(2.00f, 4.00f), float2(0.50f, 3.00f) }, 0.f},        // Outside box

            // Clockwise triangles
            {{ float2(0.00f, 0.00f), float2(1.00f, 1.50f), float2(2.00f, 1.00f) }, 0.f},        // Outside box
            {{ float2(2.25f, 2.50f), float2(2.75f, 2.25f), float2(2.50f, 2.00f) }, 0.f},        // Outside box
            {{ float2(1.00f, 3.00f), float2(1.50f, 3.25f), float2(2.00f, 3.00f) }, 0.f},        // Outside box, touching edge
            {{ float2(1.75f, 1.75f), float2(2.25f, 2.25f), float2(2.25f, 1.75f) }, 0.f},        // Outside box, touching edge
            {{ float2(1.25f, 2.25f), float2(1.25f, 2.75f), float2(1.75f, 2.25f) }, -0.125f},    // Inside box (area -1/8)
            {{ float2(1.25f, 2.75f), float2(1.75f, 2.50f), float2(1.50f, 2.25f) }, -0.09375f},  // Inside box (area -3/32)
            {{ float2(1.00f, 2.00f), float2(1.25f, 3.00f), float2(2.00f, 2.50f) }, -0.4375f},   // All vert on edge (area -7/16)
            {{ float2(1.25f, 3.00f), float2(1.50f, 3.00f), float2(1.75f, 2.00f) }, -0.125f},    // All vert on edge (area -1/8)
            {{ float2(2.00f, 2.00f), float2(1.00f, 2.00f), float2(1.00f, 3.00f) }, -0.5f},      // All vert on edge (area -1/2)
            {{ float2(1.00f, 2.25f), float2(1.00f, 2.75f), float2(1.50f, 2.50f) }, -0.125f},    // Two vert on edge, one inside (area -1/8)
            {{ float2(1.50f, 2.25f), float2(2.50f, 3.00f), float2(2.50f, 2.50f) }, -0.0625f},   // Clipped 3 verts (area -1/16)
            {{ float2(0.50f, 2.75f), float2(0.50f, 3.25f), float2(1.50f, 2.75f) }, -0.0625f},   // Clipped 3 verts (area -1/16)
            {{ float2(1.00f, 3.00f), float2(2.00f, 1.50f), float2(1.25f, 1.50f) }, -0.25f},     // Clipped 3 verts (area -1/4)
            {{ float2(1.50f, 2.25f), float2(1.50f, 2.75f), float2(2.50f, 2.25f) }, -0.1875f},   // Clipped 4 verts, two vert inside, one outside (area -3/16)
            {{ float2(1.25f, 2.25f), float2(1.75f, 2.75f), float2(1.75f, 1.75f) }, -0.21875f},  // Clipped 4 verts, two vert inside, one outside (area -7/32)
            {{ float2(1.75f, 1.75f), float2(1.75f, 2.75f), float2(2.25f, 1.75f) }, -0.125f},    // Clipped 4 verts, one vert inside, two outside (area -1/8)
            {{ float2(1.25f, 2.50f), float2(0.50f, 3.25f), float2(2.75f, 3.00f) }, -0.375f},    // Clipped 4 verts, one vert inside, two outside (area -3/8)
            {{ float2(1.00f, 2.50f), float2(2.50f, 3.50f), float2(1.50f, 2.00f) }, -0.5f},      // Clipped 4 verts, two vert on edge, one outside (area -1/2)
            {{ float2(1.00f, 2.50f), float2(2.50f, 3.50f), float2(3.00f, 3.00f) }, -0.1875f},   // Clipped 4 verts, one vert on edge, two outside (area -3/16)
            {{ float2(0.50f, 2.25f), float2(0.50f, 2.75f), float2(2.50f, 2.75f) }, -0.25f},     // Clipped 4 verts, all verts outside, tri intersects box (area -1/4)
            {{ float2(1.25f, 1.75f), float2(0.75f, 2.75f), float2(1.75f, 1.75f) }, -0.109375f}, // Clipped 4 verts, all verts outside, tri intersects box (area -7/64)
            {{ float2(0.00f, 0.00f), float2(1.00f, 8.00f), float2(3.00f, 2.00f) }, -1.0f},      // Clipped 4 verts, box fully enclosed in tri (area -1)
            {{ float2(1.25f, 2.50f), float2(0.50f, 3.25f), float2(2.75f, 3.00f) }, -0.375f},    // Clipped 5 verts (area -3/8)
            {{ float2(1.50f, 2.50f), float2(2.25f, 2.50f), float2(1.50f, 1.75f) }, -0.21875f},  // Clipped 5 verts (area -7/32)
            {{ float2(0.75f, 2.25f), float2(2.25f, 2.75f), float2(1.75f, 1.75f) }, -0.46875f},  // Clipped 6 verts (area -15/32)
            {{ float2(0.75f, 2.25f), float2(1.25f, 3.25f), float2(2.25f, 2.25f) }, -0.609375f}, // Clipped 6 verts (area -39/64)
            {{ float2(1.50f, 1.75f), float2(0.75f, 2.50f), float2(2.25f, 3.25f) }, -0.6875f},   // Clipped 7 verts (area -11/16)

            // Counter-clockwise triangles
            {{ float2(0.00f, 0.00f), float2(2.00f, 1.00f), float2(1.00f, 1.50f) }, 0.f},        // Outside box
            {{ float2(2.75f, 2.25f), float2(2.25f, 2.50f), float2(2.50f, 2.00f) }, 0.f},        // Outside box
            {{ float2(1.00f, 3.00f), float2(2.00f, 3.00f), float2(1.50f, 3.25f) }, 0.f},        // Outside box, touching edge
            {{ float2(2.25f, 2.25f), float2(1.75f, 1.75f), float2(2.25f, 1.75f) }, 0.f},        // Outside box, touching edge
            {{ float2(1.25f, 2.25f), float2(1.75f, 2.25f), float2(1.25f, 2.75f) }, 0.125f},     // Inside box (area 1/8)
            {{ float2(1.25f, 2.75f), float2(1.50f, 2.25f), float2(1.75f, 2.50f) }, 0.09375},    // Inside box (area 3/32)
            {{ float2(1.00f, 2.00f), float2(2.00f, 2.50f), float2(1.25f, 3.00f) }, 0.4375f},    // All vert on edge (area 7/16)
            {{ float2(1.50f, 3.00f), float2(1.25f, 3.00f), float2(1.75f, 2.00f) }, 0.125f},     // All vert on edge (area 1/8)
            {{ float2(1.00f, 2.00f), float2(2.00f, 2.00f), float2(1.00f, 3.00f) }, 0.5f},       // All vert on edge (area 1/2)
            {{ float2(1.00f, 2.75f), float2(1.00f, 2.25f), float2(1.50f, 2.50f) }, 0.125f},     // Two vert on edge, one inside (area 1/8)
            {{ float2(2.50f, 3.00f), float2(1.50f, 2.25f), float2(2.50f, 2.50f) }, 0.0625f},    // Clipped 3 verts (area 1/16)
            {{ float2(0.50f, 3.25f), float2(0.50f, 2.75f), float2(1.50f, 2.75f) }, 0.0625f},    // Clipped 3 verts (area 1/16)
            {{ float2(2.00f, 1.50f), float2(1.00f, 3.00f), float2(1.25f, 1.50f) }, 0.25f},      // Clipped 3 verts (area 1/4)
            {{ float2(1.50f, 2.25f), float2(2.50f, 2.25f), float2(1.50f, 2.75f) }, 0.1875f},    // Clipped 4 verts, two vert inside, one outside (area 3/16)
            {{ float2(1.25f, 2.25f), float2(1.75f, 1.75f), float2(1.75f, 2.75f) }, 0.21875f},   // Clipped 4 verts, two vert inside, one outside (area 7/32)
            {{ float2(1.75f, 2.75f), float2(1.75f, 1.75f), float2(2.25f, 1.75f) }, 0.125f},     // Clipped 4 verts, one vert inside, two outside (area 1/8)
            {{ float2(1.25f, 2.50f), float2(2.75f, 3.00f), float2(0.50f, 3.25f) }, 0.375f},     // Clipped 4 verts, one vert inside, two outside (area 3/8)
            {{ float2(2.50f, 3.50f), float2(1.00f, 2.50f), float2(1.50f, 2.00f) }, 0.5f},       // Clipped 4 verts, two vert on edge, one outside (area 1/2)
            {{ float2(2.50f, 3.50f), float2(1.00f, 2.50f), float2(3.00f, 3.00f) }, 0.1875f},    // Clipped 4 verts, one vert on edge, two outside (area 3/16)
            {{ float2(0.50f, 2.25f), float2(2.50f, 2.75f), float2(0.50f, 2.75f) }, 0.25f},      // Clipped 4 verts, all vert outside, tri intersects box (area 1/4)
            {{ float2(0.75f, 2.75f), float2(1.25f, 1.75f), float2(1.75f, 1.75f) }, 0.109375f},  // Clipped 4 verts, all vert outside, tri intersects box (area 7/64)
            {{ float2(0.00f, 0.00f), float2(3.00f, 2.00f), float2(1.00f, 8.00f) }, 1.0f},       // Clipped 4 verts, box fully enclosed in tri (area 1)
            {{ float2(0.50f, 3.25f), float2(1.25f, 2.50f), float2(2.75f, 3.00f) }, 0.375f},     // Clipped 5 verts (area 3/8)
            {{ float2(1.50f, 2.50f), float2(1.50f, 1.75f), float2(2.25f, 2.50f) }, 0.21875f},   // Clipped 5 verts (area 7/32)
            {{ float2(2.25f, 2.75f), float2(0.75f, 2.25f), float2(1.75f, 1.75f) }, 0.46875f},   // Clipped 6 verts (area 15/32)
            {{ float2(1.25f, 3.25f), float2(0.75f, 2.25f), float2(2.25f, 2.25f) }, 0.609375f},  // Clipped 6 verts (area 39/64)
            {{ float2(0.75f, 2.50f), float2(1.50f, 1.75f), float2(2.25f, 3.25f) }, 0.6875f},    // Clipped 7 verts (area 11/16)
        };

        auto createPosBuffer = [](const std::vector<TestCase>& tests)
        {
            std::vector<float3> pos;
            for (size_t i = 0; i < tests.size(); i++)
            {
                pos.push_back(float3(tests[i].p[0], 0.f));
                pos.push_back(float3(tests[i].p[1], 0.f));
                pos.push_back(float3(tests[i].p[2], 0.f));
            }
            FALCOR_ASSERT(pos.size() == 3 * tests.size());
            return Buffer::createStructured(sizeof(float3), (uint32_t)pos.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, pos.data(), false);
        };

        auto createAABBBuffer = [](const std::vector<AABB2D>& aabb)
        {
            return Buffer::createStructured(sizeof(AABB2D), (uint32_t)aabb.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, aabb.data(), false);
        };

        // Test manually created fixed test cases.
        // The exact expected clipped area is specified in the table above.

        uint32_t nTests = (uint32_t)testsFixed.size();
        std::vector<AABB2D> aabb;
        aabb.resize(nTests, { float2(1.f, 2.f), float2(2.f, 3.f) });

        // Setup and run GPU test.
        ctx.createProgram(kShaderFilename, "testComputeClippedTriangleArea2D");
        ctx.allocateStructuredBuffer("result", nTests);
        ctx["pos"] = createPosBuffer(testsFixed);
        ctx["aabb"] = createAABBBuffer(aabb);
        ctx["CB"]["n"] = nTests;
        ctx.runProgram(nTests);

        // Verify results.
        auto verify = [&ctx](const std::vector<AABB2D>& aabb, const std::vector<TestCase>& tests, float threshold, const std::string& desc)
        {
            const float3* result = ctx.mapBuffer<const float3>("result");
            for (size_t i = 0; i < tests.size(); i++)
            {
                float expectedArea = tests[i].area;
                float returnedArea = result[i].x;
                EXPECT(!std::isnan(returnedArea) && !std::isinf(returnedArea)) << " (i = " << i << ")";

                float absErr = std::abs(returnedArea - expectedArea);
                float maxErr = aabb[i].area() * threshold;
                EXPECT_LE(absErr, maxErr) << "returned " << returnedArea << ", expected " << expectedArea << " (" << desc << " i=" << i << ")";
            }
            ctx.unmapBuffer("result");
        };

        verify(aabb, testsFixed, 1e-6f, "Fixed test");

        // Test randomly created triangles and bounding boxes.
        // The triangles are numerically integrated to get an estimation of the expected clipped area.

        nTests = 10000;
        std::vector<TestCase> testsRandom(nTests);
        aabb.resize(nTests);

        std::mt19937 r;
        std::normal_distribution<> d{ 0.f, 1.f }; // mean 0, stddev 1
        auto f = [&]() { return float2(d(r), d(r)); };

        for (size_t i = 0; i < nTests; i++)
        {
            // Create a random bounding box.
            auto& b = aabb[i];
            b.minPos = f();
            b.maxPos = f();
            if (b.minPos.x > b.maxPos.x) std::swap(b.minPos.x, b.maxPos.x);
            if (b.minPos.y > b.maxPos.y) std::swap(b.minPos.y, b.maxPos.y);

            // Create a random triangle.
            auto& t = testsRandom[i];
            for (int j = 0; j < 3; j++) t.p[j] = f();

            // Estimate the clipped triangle area by numerical integration.
            // Using a uniform grid of samples over the bounding box to check how many are inside the triangle.
            const uint32_t m = 100; // 100x100 grid
            uint32_t hits = 0;

            for (int j = 0; j < m; j++)
            {
                float v = (j + 0.5f) / m;
                float y = lerp(b.minPos.y, b.maxPos.y, v);

                for (int k = 0; k < m; k++)
                {
                    float u = (k + 0.5f) / m;
                    float x = lerp(b.minPos.x, b.maxPos.x, u);

                    if (t.isInside(float2(x, y))) hits++;
                }
            }

            // Compute estimed clipped area based on fraction of hits.
            float sign = t.triangleArea() >= 0.f ? 1.f : -1.f;
            t.area = sign * b.area() * ((float)hits / (m * m));
        }

        // Run GPU test.
        ctx.allocateStructuredBuffer("result", nTests);
        ctx["pos"] = createPosBuffer(testsRandom);
        ctx["aabb"] = createAABBBuffer(aabb);
        ctx["CB"]["n"] = nTests;
        ctx.runProgram(nTests);

        // Verify results. Using a larger test threshold to account for errors in area estimation above.
        verify(aabb, testsRandom, 0.01f, "Random test");
    }
}
