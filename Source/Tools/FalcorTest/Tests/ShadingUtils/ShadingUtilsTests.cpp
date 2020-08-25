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

namespace Falcor
{
    namespace
    {
        using float3 = float3;

        enum class RayOriginLocation
        {
            None = 0x0,
            Outside = 0x1,
            Inside = 0x2
        };

        float3 getHitPoint(float radius, float3 center)
        {
            std::mt19937 rng;
            auto dist = std::uniform_real_distribution<float>(-1, 1);
            auto r = [&]() -> float { return dist(rng); };

            float x1, x2;
            do
            {
                x1 = r();
                x2 = r();
            } while (x1 * x1 + x2 * x2 >= 1);

            float3 point;
            point.x = radius * 2 * x1 * std::sqrt(1 - x1 * x1 - x2 * x2) + center.x;
            point.y = radius * 2 * x2 * std::sqrt(1 - x1 * x1 - x2 * x2) + center.y;
            point.z = radius * (1 - 2 * (x1 * x1 + x2 * x2)) + center.z;
            return point;
        }

        float3 getRayOrigin(RayOriginLocation loc, float radius, float3 center, float3 hit, bool tangential)
        {
            std::mt19937 rng;
            auto distOut = std::uniform_real_distribution<float>(-20.f, 20.f);
            auto rOut = [&]() -> float { return distOut(rng); };
            auto distIn = std::uniform_real_distribution<float>(-radius, radius);
            auto rIn = [&]() -> float { return distIn(rng); };

            float3 normal = center - hit;
            float3 shiftedHit = hit - center;
            float x, y, z;

            switch (loc)
            {
            case RayOriginLocation::Outside:
                if (tangential)
                {
                    x = rOut();
                    y = rOut();
                    z = (glm::dot(normal, hit) - normal.x * x - normal.y * y) / normal.z;
                    return float3(x, y, z);
                }

                do
                {
                    x = rOut();
                    y = rOut();
                    z = rOut();
                } while (x * x + y * y + z * z <= radius * radius || glm::dot(normal, float3(x, y, z) - shiftedHit) >= 0);
                break;
            case RayOriginLocation::Inside:
                do
                {
                    x = rIn();
                    y = rIn();
                    z = rIn();
                } while (x * x + y * y + z * z >= radius * radius);
                break;
            default:
                should_not_get_here();
            }

            return float3(x, y, z) + center;
        }

        float3 getRayDir(bool hasIntersection, float3 origin, float3 hit, bool normalized)
        {
            if (hit == origin)
            {
                std::mt19937 rng;
                auto dist = std::uniform_real_distribution<float>(-5, 5);
                auto r = [&]() -> float { return dist(rng); };

                auto dir = float3(r(), r(), r());
                return (normalized) ? glm::normalize(dir) : dir;
            }

            float3 dir = hit - origin;
            if (hasIntersection)
            {
                return (normalized) ? glm::normalize(dir) : dir;
            }
            else
            {
                return (normalized) ? glm::normalize(-dir) : -dir;
            }
        }
    }

    // Just check the first four values.
    GPU_TEST(RadicalInverse)
    {
        ctx.createProgram("Tests/ShadingUtils/ShadingUtilsTests.cs.slang", "testRadicalInverse");
        ctx.allocateStructuredBuffer("result", 4);
        ctx["TestCB"]["resultSize"] = 4;
        ctx.runProgram();

        const float *s = ctx.mapBuffer<const float>("result");
        EXPECT_EQ(s[0], 0.f);
        EXPECT_EQ(s[1], 0.5f);
        EXPECT_EQ(s[2], 0.25f);
        EXPECT_EQ(s[3], 0.75f);
        ctx.unmapBuffer("result");
    }

    GPU_TEST(Random)
    {
        ctx.createProgram("Tests/ShadingUtils/ShadingUtilsTests.cs.slang", "testRand");
        const int32_t n = 4 * 1024 * 1024;
        ctx.allocateStructuredBuffer("result", n);
        ctx["TestCB"]["resultSize"] = n;
        ctx.runProgram();

        // A fairly crude test: bucket the range [0,1] into nBuckets buckets
        // and make sure that all of them have more or less 1/nBuckets of the
        // total values.  This doesn't really test the quality of the PRNG very
        // well, but will at least detect if it's totally borked.
        const float* r = ctx.mapBuffer<const float>("result");
        constexpr int32_t nBuckets = 64;
        int32_t counts[nBuckets] = { 0 };
        for (int32_t i = 0; i < n; ++i)
        {
            EXPECT(r[i] >= 0 && r[i] < 1.f) << r[i];
            ++counts[int32_t(r[i] * nBuckets)];
        }
        ctx.unmapBuffer("result");

        for (int32_t i = 0; i < nBuckets; ++i)
        {
            EXPECT_GT(counts[i], .98 * n / nBuckets);
            EXPECT_LT(counts[i], 1.02 * n / nBuckets);
        }
    }

    GPU_TEST(SphericalCoordinates)
    {
        ctx.createProgram("Tests/ShadingUtils/ShadingUtilsTests.cs.slang", "testSphericalCoordinates");
        constexpr int32_t n = 1024 * 1024;
        ctx.allocateStructuredBuffer("result", n);
        ctx["TestCB"]["resultSize"] = n;
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

    GPU_TEST(RaySphereIntersection)
    {
        std::mt19937 rng;
        auto dist = std::uniform_real_distribution<float>(-10.f, 10.f);
        auto r = [&]() -> float { return dist(rng); };

        std::vector<float3> testSphereCenters(12);
        std::vector<float> testSphereRadii(12);
        std::vector<float3> refIsects(12);
        std::vector<float3> testRayOrigins(12);
        std::vector<float3> testRayDirs(12);

        for (int32_t i = 0; i < 12; i++)
        {
            testSphereCenters[i] = float3(r(), r(), r());
            testSphereRadii[i] = abs(r());
            refIsects[i] = getHitPoint(testSphereRadii[i], testSphereCenters[i]);
            switch (i)
            {
            case 0:
            case 1:
                testRayOrigins[i] = getRayOrigin(RayOriginLocation::Outside, testSphereRadii[i], testSphereCenters[i], refIsects[i], false);
                testRayDirs[i] = getRayDir(true, testRayOrigins[i], refIsects[i], (i % 2 == 0) ? true : false);
                break;
            case 2:
            case 3:
                testRayOrigins[i] = getRayOrigin(RayOriginLocation::Outside, testSphereRadii[i], testSphereCenters[i], refIsects[i], true);
                testRayDirs[i] = getRayDir(true, testRayOrigins[i], refIsects[i], (i % 2 == 0) ? true : false);
                break;
            case 4:
            case 5:
                testRayOrigins[i] = getRayOrigin(RayOriginLocation::Inside, testSphereRadii[i], testSphereCenters[i], refIsects[i], false);
                testRayDirs[i] = getRayDir(true, testRayOrigins[i], refIsects[i], (i % 2 == 0) ? true : false);
                break;
            case 6:
            case 7:
                testRayOrigins[i] = getRayOrigin(RayOriginLocation::Outside, testSphereRadii[i], testSphereCenters[i], refIsects[i], false);
                testRayDirs[i] = getRayDir(false, testRayOrigins[i], refIsects[i], (i % 2 == 0) ? true : false);
                break;
            case 8:
            case 9:
                testRayOrigins[i] = getRayOrigin(RayOriginLocation::Outside, testSphereRadii[i], testSphereCenters[i], refIsects[i], false);
                testRayDirs[i] = getRayDir(false, testRayOrigins[i], refIsects[i], (i % 2 == 0) ? true : false);
                break;
            case 10:
            case 11:
                testRayOrigins[i] = refIsects[i];
                testRayDirs[i] = getRayDir(true, testRayOrigins[i], refIsects[i], (i % 2 == 0) ? true : false);
                break;
            }
        }

        ctx.createProgram("Tests/ShadingUtils/ShadingUtilsTests.cs.slang", "testRaySphereIntersection");
        ctx.allocateStructuredBuffer("sphereCenter", 12, testSphereCenters.data(), testSphereCenters.size() * sizeof(float3));
        ctx.allocateStructuredBuffer("sphereRadius", 12, testSphereRadii.data());
        ctx.allocateStructuredBuffer("rayOrigin", 12, testRayOrigins.data(), testRayOrigins.size() * sizeof(float3));
        ctx.allocateStructuredBuffer("rayDir", 12, testRayDirs.data(), testRayDirs.size() * sizeof(float3));
        ctx.allocateStructuredBuffer("isectResult", 12);
        ctx.allocateStructuredBuffer("isectLoc", 12);
        ctx["TestCB"]["resultSize"] = 12;

        ctx.runProgram();

        const uint32_t* result = ctx.mapBuffer<const uint32_t>("isectResult");
        const float3* isectLoc = ctx.mapBuffer<const float3>("isectLoc");
        for (int32_t i = 0; i < 12; i++)
        {
            switch (i)
            {
            case 6:
            case 7:
            case 8:
            case 9:
                EXPECT_EQ(result[i], 0u);
                break;
            default:
                const float eps = 5e-4f;
                EXPECT_EQ(result[i], 1u) << "RaySphereTestCase" << i << ", expected " << 1 << ", got " << result[i];
                EXPECT(abs(isectLoc[i].x - refIsects[i].x) <= eps * (abs(isectLoc[i].x) + abs(refIsects[i].x) + 1.0f)) << "RaySphereTestCase" << i << ", expected " << refIsects[i].x << ", got " << isectLoc[i].x;
                EXPECT(abs(isectLoc[i].y - refIsects[i].y) <= eps * (abs(isectLoc[i].y) + abs(refIsects[i].y) + 1.0f)) << "RaySphereTestCase" << i << ", expected " << refIsects[i].y << ", got " << isectLoc[i].y;
                EXPECT(abs(isectLoc[i].z - refIsects[i].z) <= eps * (abs(isectLoc[i].z) + abs(refIsects[i].z) + 1.0f)) << "RaySphereTestCase" << i << ", expected " << refIsects[i].z << ", got " << isectLoc[i].z;
            }

            // << "RaySphereTestCase" << i << ", expected (" << refIsects[i].x << ", " << refIsects[i].y << ", " << refIsects[i].z << "), got (" << isectLoc[i].x << ", " << isectLoc[i].y << ", " << isectLoc[i].z << ")";
        }
        ctx.unmapBuffer("isectResult");
        ctx.unmapBuffer("isectLoc");
    }

}  // namespace Falcor
