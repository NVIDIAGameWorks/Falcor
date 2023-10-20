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
#include "Utils/Math/Matrix.h"

#include <fmt/format.h>
#include <iostream>

namespace Falcor
{

template<typename T, int N>
bool almostEqual(math::vector<T, N> a, math::vector<T, N> b, T epsilon = T(1e-5))
{
    return all(abs(a - b) < math::vector<T, N>(epsilon));
}

template<typename T>
bool almostEqual(math::quat<T> a, math::quat<T> b, T epsilon = T(1e-5))
{
    return all(math::vector<T, 4>(b.x - a.x, b.y - a.y, b.z - a.z, b.w - a.w) < math::vector<T, 4>(epsilon));
}

#define EXPECT_ALMOST_EQ(a, b) EXPECT_TRUE(almostEqual(a, b)) << fmt::format("{} != {}", a, b)

CPU_TEST(Matrix_Constructor)
{
    // Default constructor
    {
        float4x4 m;
        EXPECT_EQ(m[0], float4(1, 0, 0, 0));
        EXPECT_EQ(m[1], float4(0, 1, 0, 0));
        EXPECT_EQ(m[2], float4(0, 0, 1, 0));
        EXPECT_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Initializer list constructor
    {
        float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        EXPECT_EQ(m[0], float4(1, 2, 3, 4));
        EXPECT_EQ(m[1], float4(5, 6, 7, 8));
        EXPECT_EQ(m[2], float4(9, 10, 11, 12));
        EXPECT_EQ(m[3], float4(13, 14, 15, 16));
    }

    // Identity
    {
        float4x4 m = float4x4::identity();
        EXPECT_EQ(m[0], float4(1, 0, 0, 0));
        EXPECT_EQ(m[1], float4(0, 1, 0, 0));
        EXPECT_EQ(m[2], float4(0, 0, 1, 0));
        EXPECT_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Zeros
    {
        float4x4 m = float4x4::zeros();
        EXPECT_EQ(m[0], float4(0, 0, 0, 0));
        EXPECT_EQ(m[1], float4(0, 0, 0, 0));
        EXPECT_EQ(m[2], float4(0, 0, 0, 0));
        EXPECT_EQ(m[3], float4(0, 0, 0, 0));
    }
}

CPU_TEST(Matrix_multilply)
{
    // Scalar multiplication
    {
        float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        float4x4 m2 = m * 2.f;
        EXPECT_EQ(m2[0], float4(2, 4, 6, 8));
        EXPECT_EQ(m2[1], float4(10, 12, 14, 16));
        EXPECT_EQ(m2[2], float4(18, 20, 22, 24));
        EXPECT_EQ(m2[3], float4(26, 28, 30, 32));
    }

    // Matrix/matrix multiplication
    {
        float4x4 m1({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        float4x4 m2({-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16});
        float4x4 m3 = mul(m1, m2);
        EXPECT_EQ(m3[0], float4(-90, -100, -110, -120));
        EXPECT_EQ(m3[1], float4(-202, -228, -254, -280));
        EXPECT_EQ(m3[2], float4(-314, -356, -398, -440));
        EXPECT_EQ(m3[3], float4(-426, -484, -542, -600));
    }

    // Matrix/vector multiplication
    {
        float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        float4 v(1, 2, 3, 4);
        float4 v2 = mul(m, v);
        EXPECT_EQ(v2, float4(30, 70, 110, 150));
    }

    // Vector/matrix multiplication
    {
        float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        float4 v(1, 2, 3, 4);
        float4 v2 = mul(v, m);
        EXPECT_EQ(v2, float4(90, 100, 110, 120));
    }
}

CPU_TEST(Matrix_transformPoint)
{
    float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    float3 v(1, 2, 3);
    float3 v2 = transformPoint(m, v);
    EXPECT_EQ(v2, float3(18, 46, 74));
}

CPU_TEST(Matrix_transformVector)
{
    // 3x3
    {
        float3x3 m({1, 2, 3, 4, 5, 6, 7, 8, 9});
        float3 v(1, 2, 3);
        float3 v2 = transformVector(m, v);
        EXPECT_EQ(v2, float3(14, 32, 50));
    }

    // 4x4
    {
        float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        float3 v(1, 2, 3);
        float3 v2 = transformVector(m, v);
        EXPECT_EQ(v2, float3(14, 38, 62));
    }
}

CPU_TEST(Matrix_transpose)
{
    // 3x3
    {
        float3x3 m({1, 2, 3, 4, 5, 6, 7, 8, 9});
        float3x3 m2 = transpose(m);
        EXPECT_EQ(m2[0], float3(1, 4, 7));
        EXPECT_EQ(m2[1], float3(2, 5, 8));
        EXPECT_EQ(m2[2], float3(3, 6, 9));
    }

    // 4x4
    {
        float4x4 m({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        float4x4 m2 = transpose(m);
        EXPECT_EQ(m2[0], float4(1, 5, 9, 13));
        EXPECT_EQ(m2[1], float4(2, 6, 10, 14));
        EXPECT_EQ(m2[2], float4(3, 7, 11, 15));
        EXPECT_EQ(m2[3], float4(4, 8, 12, 16));
    }
}

CPU_TEST(Matrix_translate)
{
    float4x4 m({
        // clang-format off
        1,  0, 0, 10,
        0, -1, 0, 20,
        0,  0, 1, 30,
        0,  0, 0, 1
        // clang-format on
    });
    float4x4 m2 = translate(m, float3(1, 2, 3));
    EXPECT_ALMOST_EQ(m2[0], float4(1, 0, 0, 11));
    EXPECT_ALMOST_EQ(m2[1], float4(0, -1, 0, 18));
    EXPECT_ALMOST_EQ(m2[2], float4(0, 0, 1, 33));
    EXPECT_ALMOST_EQ(m2[3], float4(0, 0, 0, 1));
}

CPU_TEST(Matrix_rotate)
{
    float4x4 m({
        // clang-format off
        1,  0, 0, 10,
        0, -1, 0, 20,
        0,  0, 1, 30,
        0,  0, 0, 1
        // clang-format on
    });
    float4x4 m2 = rotate(m, math::radians(90.f), float3(0, 1, 0));
    EXPECT_ALMOST_EQ(m2[0], float4(0, 0, 1, 10));
    EXPECT_ALMOST_EQ(m2[1], float4(0, -1, 0, 20));
    EXPECT_ALMOST_EQ(m2[2], float4(-1, 0, 0, 30));
    EXPECT_ALMOST_EQ(m2[3], float4(0, 0, 0, 1));
}

CPU_TEST(Matrix_scale)
{
    float4x4 m({
        // clang-format off
        1,  0, 0, 10,
        0, -1, 0, 20,
        0,  0, 1, 30,
        0,  0, 0, 1
        // clang-format on
    });
    float4x4 m2 = scale(m, float3(2, 3, 4));
    EXPECT_ALMOST_EQ(m2[0], float4(2, 0, 0, 10));
    EXPECT_ALMOST_EQ(m2[1], float4(0, -3, 0, 20));
    EXPECT_ALMOST_EQ(m2[2], float4(0, 0, 4, 30));
    EXPECT_ALMOST_EQ(m2[3], float4(0, 0, 0, 1));
}

CPU_TEST(Matrix_determinant)
{
    // 2x2
    {
        float2x2 m1 = float2x2({1, 2, 1, 2});
        EXPECT_EQ(math::determinant(m1), 0);
        float2x2 m2 = float2x2({1, 2, 3, 4});
        EXPECT_EQ(math::determinant(m2), -2);
    }

    // 3x3
    {
        float3x3 m1 = float3x3({1, 2, 3, 4, 5, 6, 7, 8, 9});
        EXPECT_EQ(math::determinant(m1), 0);
        float3x3 m2 = float3x3({1, 2, 3, 6, 5, 4, 8, 7, 9});
        EXPECT_EQ(math::determinant(m2), -21);
    }

    // 4x4
    {
        float4x4 m1 = float4x4({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        EXPECT_EQ(math::determinant(m1), 0);
        float4x4 m2 = float4x4({1, 2, 3, 4, 8, 7, 6, 5, 9, 10, 12, 11, 15, 16, 13, 14});
        EXPECT_EQ(math::determinant(m2), 72);
    }
}

CPU_TEST(Matrix_inverse)
{
    // 2x2
    {
        float2x2 m = inverse(float2x2({1, 2, 3, 4}));
        EXPECT_ALMOST_EQ(m[0], float2(-2, 1));
        EXPECT_ALMOST_EQ(m[1], float2(1.5f, -0.5f));
    }

    // 3x3
    {
        float3x3 m = inverse(float3x3({1, 2, 3, 6, 5, 4, 8, 7, 9}));
        EXPECT_ALMOST_EQ(m[0], float3(-0.809523f, -0.142857f, 0.333333f));
        EXPECT_ALMOST_EQ(m[1], float3(1.047619f, 0.714285f, -0.666666f));
        EXPECT_ALMOST_EQ(m[2], float3(-0.095238f, -0.428571f, 0.333333f));
    }

    // 4x4
    {
        float4x4 m = inverse(float4x4({1, 2, 3, 4, 8, 7, 6, 5, 9, 10, 12, 11, 15, 16, 13, 14}));
        EXPECT_ALMOST_EQ(m[0], float4(1.125f, 1.25f, -0.5f, -0.375f));
        EXPECT_ALMOST_EQ(m[1], float4(-1.652777f, -1.527777f, 0.5f, 0.625f));
        EXPECT_ALMOST_EQ(m[2], float4(-0.625f, -0.25f, 0.5f, -0.125f));
        EXPECT_ALMOST_EQ(m[3], float4(1.263888f, 0.638888f, -0.5f, -0.125f));
    }
}

CPU_TEST(Matrix_extractEulerAngleXYZ)
{
    {
        // float4x4 m = math::matrixFromRotationXYZ(math::radians(45.f), math::radians(45.f), math::radians(45.f));
        float4x4 m = float4x4({
            // clang-format off
            0.5f, -0.5f, 0.707107f, 0.f,
            0.853553f, 0.146446f, -0.5f, 0.f,
            0.146446f, 0.853553f, 0.5f, 0.f,
            0.f, 0.f, 0.f, 1.f
            // clang-format on
        });
        float3 angles;
        math::extractEulerAngleXYZ(m, angles.x, angles.y, angles.z);
        EXPECT_ALMOST_EQ(angles, math::radians(float3(45.f, 45.f, 45.f)));
    }

    {
        // float4x4 m = math::matrixFromRotationXYZ(math::radians(20.f), math::radians(40.f), math::radians(60.f));
        float4x4 m = float4x4({
            // clang-format off
            0.383022f, -0.663414f, 0.642787f, 0.f,
            0.923720f, 0.279453f, -0.262002f, 0.f,
            -0.005813f, 0.694109f, 0.719846f, 0.f,
            0.f, 0.f, 0.f, 1.f
            // clang-format on
        });
        float3 angles;
        math::extractEulerAngleXYZ(m, angles.x, angles.y, angles.z);
        EXPECT_ALMOST_EQ(angles, math::radians(float3(20.f, 40.f, 60.f)));
    }
}

CPU_TEST(Matrix_decompose)
{
    const auto testDecompose = [&](float4x4 m,
                                   float3 expectedScale,
                                   quatf expectedOrientation,
                                   float3 expectedTranslation,
                                   float3 expectedSkew,
                                   float4 expectedPerspective,
                                   bool expectedResult = true)
    {
        float3 scale;
        quatf orientation;
        float3 translation;
        float3 skew;
        float4 perspective;
        bool result = math::decompose(m, scale, orientation, translation, skew, perspective);
        if (expectedResult)
        {
            EXPECT_ALMOST_EQ(scale, expectedScale);
            EXPECT_ALMOST_EQ(orientation, expectedOrientation);
            EXPECT_ALMOST_EQ(translation, expectedTranslation);
            EXPECT_ALMOST_EQ(skew, expectedSkew);
            EXPECT_ALMOST_EQ(perspective, expectedPerspective);
        }
        EXPECT(result == expectedResult);
    };

    // Zero matrix
    testDecompose(
        float4x4::zeros(), // matrix
        float3(),          // scale
        quatf(),           // orientation
        float3(),          // translation
        float3(),          // skew
        float4(),          // perspective
        false              // result
    );

    // Identity matrix
    testDecompose(
        float4x4::identity(),      // matrix
        float3(1.f, 1.f, 1.f),     // scale
        quatf::identity(),         // orientation
        float3(0.f, 0.f, 0.f),     // translation
        float3(0.f, 0.f, 0.f),     // skew
        float4(0.f, 0.f, 0.f, 1.f) // perspective
    );

    // Scale only
    testDecompose(
        float4x4({
            // clang-format off
            2.f, 0.f, 0.f, 0.f,
            0.f, 3.f, 0.f, 0.f,
            0.f, 0.f, 4.f, 0.f,
            0.f, 0.f, 0.f, 1.f
            // clang-format on
        }),
        float3(2.f, 3.f, 4.f),     // scale
        quatf::identity(),         // orientation
        float3(0.f, 0.f, 0.f),     // translation
        float3(0.f, 0.f, 0.f),     // skew
        float4(0.f, 0.f, 0.f, 1.f) // perspective
    );

    // Orientation only
    // float4x4 m = math::matrixFromRotationX(math::radians(45.f));
    testDecompose(
        float4x4({
            // clang-format off
            1.f, 0.f, 0.f, 0.f,
            0.f, 0.707107f, -0.707107f, 0.f,
            0.f, 0.707107f, 0.707107f, 0.f,
            0.f, 0.f, 0.f, 1.f
            // clang-format on
        }),
        float3(1.f, 1.f, 1.f),                // scale
        quatf(0.382683f, 0.f, 0.f, 0.92388f), // orientation
        float3(0.f, 0.f, 0.f),                // translation
        float3(0.f, 0.f, 0.f),                // skew
        float4(0.f, 0.f, 0.f, 1.f)            // perspective
    );

    // Translation only
    testDecompose(
        float4x4({
            // clang-format off
            1.f, 0.f, 0.f, 1.f,
            0.f, 1.f, 0.f, 2.f,
            0.f, 0.f, 1.f, 3.f,
            0.f, 0.f, 0.f, 1.f
            // clang-format off
        }),
        float3(1.f, 1.f, 1.f),     // scale
        quatf::identity(),         // orientation
        float3(1.f, 2.f, 3.f),     // translation
        float3(0.f, 0.f, 0.f),     // skew
        float4(0.f, 0.f, 0.f, 1.f) // perspective
    );

    // Skew only
    testDecompose(
        float4x4({
            // clang-format off
            1.f, 2.f, 3.f, 0.f,
            0.f, 1.f, 4.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f
            // clang-format on
        }),
        float3(1.f, 1.f, 1.f),     // scale
        quatf::identity(),         // orientation
        float3(0.f, 0.f, 0.f),     // translation
        float3(4.f, 3.f, 2.f),     // skew
        float4(0.f, 0.f, 0.f, 1.f) // perspective
    );

    // Perspective only
    testDecompose(
        float4x4({
            // clang-format off
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.1f, 0.2f, 0.3f, 1.f
            // clang-format on
        }),
        float3(1.f, 1.f, 1.f),        // scale
        quatf::identity(),            // orientation
        float3(0.f, 0.f, 0.f),        // translation
        float3(0.f, 0.f, 0.f),        // skew
        float4(0.1f, 0.2f, 0.3f, 1.f) // perspective
    );

    // Affine transform
    float4x4 m = float4x4::identity();
    m = mul(math::matrixFromScaling(float3(2.f, 3.f, 4.f)), m);
    m = mul(math::matrixFromRotationX(math::radians(45.f)), m);
    m = mul(math::matrixFromTranslation(float3(1.f, 2.f, 3.f)), m);
    testDecompose(
        float4x4({
            // clang-format off
            2.f, 0.f, 0.f, 1.f,
            0.f, 2.12132f, -2.82843f, 2.f,
            0.f, 2.12132f, 2.82843f, 3.f,
            0.f, 0.f, 0.f, 1.f
            // clang-format on
        }),
        float3(2.f, 3.f, 4.f),                // scale
        quatf(0.382683f, 0.f, 0.f, 0.92388f), // orientation
        float3(1.f, 2.f, 3.f),                // translation
        float3(0.f, 0.f, 0.f),                // skew
        float4(0.f, 0.f, 0.f, 1.f)            // perspective
    );
}

CPU_TEST(Matrix_matrixFromCoefficients)
{
    // 3x3
    {
        const float values[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        float3x3 m = math::matrixFromCoefficients<float, 3, 3>(values);
        EXPECT_EQ(m[0], float3(1, 2, 3));
        EXPECT_EQ(m[1], float3(4, 5, 6));
        EXPECT_EQ(m[2], float3(7, 8, 9));
    }

    // 4x4
    {
        const float values[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        float4x4 m = math::matrixFromCoefficients<float, 4, 4>(values);
        EXPECT_EQ(m[0], float4(1, 2, 3, 4));
        EXPECT_EQ(m[1], float4(5, 6, 7, 8));
        EXPECT_EQ(m[2], float4(9, 10, 11, 12));
        EXPECT_EQ(m[3], float4(13, 14, 15, 16));
    }
}

CPU_TEST(Matrix_matrixFromColumns)
{
    // 2x4
    {
        math::matrix<float, 2, 4> m = math::matrixFromColumns(float2(1, 2), float2(3, 4), float2(5, 6), float2(7, 8));
        EXPECT_EQ(m[0], float4(1, 3, 5, 7));
        EXPECT_EQ(m[1], float4(2, 4, 6, 8));
    }

    // 4x2
    {
        math::matrix<float, 4, 2> m = math::matrixFromColumns(float4(1, 2, 3, 4), float4(5, 6, 7, 8));
        EXPECT_EQ(m[0], float2(1, 5));
        EXPECT_EQ(m[1], float2(2, 6));
        EXPECT_EQ(m[2], float2(3, 7));
        EXPECT_EQ(m[3], float2(4, 8));
    }

    // 4x4
    {
        float4x4 m = math::matrixFromColumns(float4(1, 2, 3, 4), float4(5, 6, 7, 8), float4(9, 10, 11, 12), float4(13, 14, 15, 16));
        EXPECT_EQ(m[0], float4(1, 5, 9, 13));
        EXPECT_EQ(m[1], float4(2, 6, 10, 14));
        EXPECT_EQ(m[2], float4(3, 7, 11, 15));
        EXPECT_EQ(m[3], float4(4, 8, 12, 16));
    }
}

CPU_TEST(Matrix_matrixFromDiagonal)
{
    // 3x3
    {
        float3x3 m = math::matrixFromDiagonal(float3(1, 2, 3));
        EXPECT_EQ(m[0], float3(1, 0, 0));
        EXPECT_EQ(m[1], float3(0, 2, 0));
        EXPECT_EQ(m[2], float3(0, 0, 3));
    }

    // 4x4
    {
        float4x4 m = math::matrixFromDiagonal(float4(1, 2, 3, 4));
        EXPECT_EQ(m[0], float4(1, 0, 0, 0));
        EXPECT_EQ(m[1], float4(0, 2, 0, 0));
        EXPECT_EQ(m[2], float4(0, 0, 3, 0));
        EXPECT_EQ(m[3], float4(0, 0, 0, 4));
    }
}

CPU_TEST(Matrix_perspective)
{
    float4x4 m = math::perspective(math::radians(45.f), 2.f, 0.1f, 1000.f);
    EXPECT_ALMOST_EQ(m[0], float4(1.207107f, 0, 0, 0));
    EXPECT_ALMOST_EQ(m[1], float4(0, 2.414213f, 0, 0));
    EXPECT_ALMOST_EQ(m[2], float4(0, 0, -1.0001f, -0.1f));
    EXPECT_ALMOST_EQ(m[3], float4(0, 0, -1.f, 0));
}

CPU_TEST(Matrix_ortho)
{
    float4x4 m = math::ortho(-10.f, 10.f, -10.f, 10.f, 0.1f, 1000.f);
    EXPECT_ALMOST_EQ(m[0], float4(0.1f, 0, 0, 0));
    EXPECT_ALMOST_EQ(m[1], float4(0, 0.1f, 0, 0));
    EXPECT_ALMOST_EQ(m[2], float4(0, 0, -0.001f, -0.0001f));
    EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1.0f));
}

CPU_TEST(Matrix_matrixFromTranslation)
{
    float4x4 m = math::matrixFromTranslation(float3(1, 2, 3));
    EXPECT_EQ(m[0], float4(1, 0, 0, 1));
    EXPECT_EQ(m[1], float4(0, 1, 0, 2));
    EXPECT_EQ(m[2], float4(0, 0, 1, 3));
    EXPECT_EQ(m[3], float4(0, 0, 0, 1));
}

CPU_TEST(Matrix_matrixFromRotation)
{
    // Rotation around X-axis by 90 degrees
    {
        float4x4 m = math::matrixFromRotation(math::radians(90.f), float3(1, 0, 0));
        EXPECT_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0, 0, -1, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0, 1, 0, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around X-axis by -45 degrees
    {
        float4x4 m = math::matrixFromRotation(math::radians(-45.f), float3(1, 0, 0));
        EXPECT_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0, 0.707106f, 0.707106f, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0, -0.707106f, 0.707106f, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Y-axis by 90 degrees
    {
        float4x4 m = math::matrixFromRotation(math::radians(90.f), float3(0, 1, 0));
        EXPECT_ALMOST_EQ(m[0], float4(0, 0, 1, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        EXPECT_ALMOST_EQ(m[2], float4(-1, 0, 0, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Y-axis by -45 degrees
    {
        float4x4 m = math::matrixFromRotation(math::radians(-45.f), float3(0, 1, 0));
        EXPECT_ALMOST_EQ(m[0], float4(0.707106f, 0, -0.707106f, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0.707106f, 0, 0.707106f, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Z-axis by 90 degrees
    {
        float4x4 m = math::matrixFromRotation(math::radians(90.f), float3(0, 0, 1));
        EXPECT_ALMOST_EQ(m[0], float4(0, -1, 0, 0));
        EXPECT_ALMOST_EQ(m[1], float4(1, 0, 0, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Z-axis by -45 degrees
    {
        float4x4 m = math::matrixFromRotation(math::radians(-45.f), float3(0, 0, 1));
        EXPECT_ALMOST_EQ(m[0], float4(0.707106f, 0.707106f, 0, 0));
        EXPECT_ALMOST_EQ(m[1], float4(-0.707106f, 0.707106f, 0, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around oblique axis
    {
        float4x4 m = math::matrixFromRotation(math::radians(60.f), normalize(float3(1, 1, 1)));
        EXPECT_ALMOST_EQ(m[0], float4(0.666666f, -0.333333f, 0.666666f, 0.f));
        EXPECT_ALMOST_EQ(m[1], float4(0.666666f, 0.666666f, -0.333333f, 0.f));
        EXPECT_ALMOST_EQ(m[2], float4(-0.333333f, 0.666666f, 0.666666f, 0.f));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }
}

CPU_TEST(Matrix_matrixFromRotationXYZ)
{
    // Rotation around X-axis by 90 degrees
    {
        float4x4 m = math::matrixFromRotationX(math::radians(90.f));
        EXPECT_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0, 0, -1, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0, 1, 0, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around X-axis by 90 degrees
    {
        float4x4 m = math::matrixFromRotationXYZ(math::radians(90.f), 0.f, 0.f);
        EXPECT_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0, 0, -1, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0, 1, 0, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around X-axis by -45 degrees
    {
        float4x4 m = math::matrixFromRotationX(math::radians(-45.f));
        EXPECT_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0, 0.707106f, 0.707106f, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0, -0.707106f, 0.707106f, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around X-axis by -45 degrees
    {
        float4x4 m = math::matrixFromRotationXYZ(math::radians(-45.f), 0.f, 0.f);
        EXPECT_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0, 0.707106f, 0.707106f, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0, -0.707106f, 0.707106f, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Y-axis by 90 degrees
    {
        float4x4 m = math::matrixFromRotationY(math::radians(90.f));
        EXPECT_ALMOST_EQ(m[0], float4(0, 0, 1, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        EXPECT_ALMOST_EQ(m[2], float4(-1, 0, 0, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Y-axis by 90 degrees
    {
        float4x4 m = math::matrixFromRotationXYZ(0.f, math::radians(90.f), 0.f);
        EXPECT_ALMOST_EQ(m[0], float4(0, 0, 1, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        EXPECT_ALMOST_EQ(m[2], float4(-1, 0, 0, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Y-axis by -45 degrees
    {
        float4x4 m = math::matrixFromRotationY(math::radians(-45.f));
        EXPECT_ALMOST_EQ(m[0], float4(0.707106f, 0, -0.707106f, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0.707106f, 0, 0.707106f, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Y-axis by -45 degrees
    {
        float4x4 m = math::matrixFromRotationXYZ(0.f, math::radians(-45.f), 0.f);
        EXPECT_ALMOST_EQ(m[0], float4(0.707106f, 0, -0.707106f, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0.707106f, 0, 0.707106f, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Z-axis by 90 degrees
    {
        float4x4 m = math::matrixFromRotationZ(math::radians(90.f));
        EXPECT_ALMOST_EQ(m[0], float4(0, -1, 0, 0));
        EXPECT_ALMOST_EQ(m[1], float4(1, 0, 0, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Z-axis by 90 degrees
    {
        float4x4 m = math::matrixFromRotationXYZ(0.f, 0.f, math::radians(90.f));
        EXPECT_ALMOST_EQ(m[0], float4(0, -1, 0, 0));
        EXPECT_ALMOST_EQ(m[1], float4(1, 0, 0, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Z-axis by -45 degrees
    {
        float4x4 m = math::matrixFromRotationZ(math::radians(-45.f));
        EXPECT_ALMOST_EQ(m[0], float4(0.707106f, 0.707106f, 0, 0));
        EXPECT_ALMOST_EQ(m[1], float4(-0.707106f, 0.707106f, 0, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around Z-axis by -45 degrees
    {
        float4x4 m = math::matrixFromRotationXYZ(0.f, 0.f, math::radians(-45.f));
        EXPECT_ALMOST_EQ(m[0], float4(0.707106f, 0.707106f, 0, 0));
        EXPECT_ALMOST_EQ(m[1], float4(-0.707106f, 0.707106f, 0, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around XYZ by 45 degrees
    {
        float4x4 m = math::matrixFromRotationXYZ(math::radians(45.f), math::radians(45.f), math::radians(45.f));
        EXPECT_ALMOST_EQ(m[0], float4(0.5f, -0.5f, 0.707107f, 0.f));
        EXPECT_ALMOST_EQ(m[1], float4(0.853553f, 0.146446f, -0.5f, 0.f));
        EXPECT_ALMOST_EQ(m[2], float4(0.146446f, 0.853553f, 0.5f, 0.f));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around XYZ by 20, 40, 60 degrees
    {
        float4x4 m = math::matrixFromRotationXYZ(math::radians(20.f), math::radians(40.f), math::radians(60.f));
        EXPECT_ALMOST_EQ(m[0], float4(0.383022f, -0.663414f, 0.642787f, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0.923720f, 0.279453f, -0.262002f, 0));
        EXPECT_ALMOST_EQ(m[2], float4(-0.005813f, 0.694109f, 0.719846f, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }
}

CPU_TEST(Matrix_matrixFromScaling)
{
    float4x4 m = math::matrixFromScaling(float3(2.f, 3.f, 4.f));
    EXPECT_ALMOST_EQ(m[0], float4(2, 0, 0, 0));
    EXPECT_ALMOST_EQ(m[1], float4(0, 3, 0, 0));
    EXPECT_ALMOST_EQ(m[2], float4(0, 0, 4, 0));
    EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
}

CPU_TEST(Matrix_matrixFromLookAt)
{
    // Right handed
    {
        float4x4 m = math::matrixFromLookAt(float3(10, 5, 0), float3(0, -5, 0), float3(0, 1, 0), math::Handedness::RightHanded);
        EXPECT_ALMOST_EQ(m[0], float4(0, 0, -1, 0));
        EXPECT_ALMOST_EQ(m[1], float4(-0.707107f, 0.707107f, 0, 3.535535f));
        EXPECT_ALMOST_EQ(m[2], float4(0.707107f, 0.707107f, 0, -10.606603f));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Left handed
    {
        float4x4 m = math::matrixFromLookAt(float3(10, 5, 0), float3(0, -5, 0), float3(0, 1, 0), math::Handedness::LeftHanded);
        EXPECT_ALMOST_EQ(m[0], float4(0, 0, 1, 0));
        EXPECT_ALMOST_EQ(m[1], float4(-0.707107f, 0.707107f, 0, 3.535535f));
        EXPECT_ALMOST_EQ(m[2], float4(-0.707107f, -0.707107f, 0, 10.606603f));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }
}

CPU_TEST(Matrix_matrixFromQuat)
{
    // Identity quaternion
    {
        float4x4 m = math::matrixFromQuat(quatf::identity());
        EXPECT_ALMOST_EQ(m[0], float4(1, 0, 0, 0));
        EXPECT_ALMOST_EQ(m[1], float4(0, 1, 0, 0));
        EXPECT_ALMOST_EQ(m[2], float4(0, 0, 1, 0));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }

    // Rotation around oblique axis
    {
        quatf q = math::quatFromAngleAxis(math::radians(60.f), normalize(float3(1, 1, 1)));
        float4x4 m = math::matrixFromQuat(q);
        EXPECT_ALMOST_EQ(m[0], float4(0.666666f, -0.333333f, 0.666666f, 0.f));
        EXPECT_ALMOST_EQ(m[1], float4(0.666666f, 0.666666f, -0.333333f, 0.f));
        EXPECT_ALMOST_EQ(m[2], float4(-0.333333f, 0.666666f, 0.666666f, 0.f));
        EXPECT_ALMOST_EQ(m[3], float4(0, 0, 0, 1));
    }
}

CPU_TEST(Matrix_FloatFormatter)
{
    float3x3 test0({1.1f, 1.2f, 1.3f, 2.1f, 2.2f, 2.3f, 3.1f, 3.2f, 3.3f});

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
