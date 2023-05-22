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
#include "Utils/Math/Quaternion.h"

namespace Falcor
{

template<typename T>
bool almostEqual(T a, T b, T epsilon = T(1e-5))
{
    return std::abs(a - b) < epsilon;
}

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

template<typename T>
bool almostEqualOrientation(math::quat<T> a, math::quat<T> b, T epsilon = T(1e-5))
{
    if (dot(a, b) < T(0))
        b = -b;
    return almostEqual(a, b, epsilon);
}

#define EXPECT_ALMOST_EQ(a, b) EXPECT_TRUE(almostEqual(a, b)) << fmt::format("{} != {}", a, b)
#define EXPECT_ALMOST_EQ_EPS(a, b, eps) EXPECT_TRUE(almostEqual(a, b, eps)) << fmt::format("{} != {}", a, b)
#define EXPECT_ALMOST_EQ_ORIENTATION(a, b) EXPECT_TRUE(almostEqualOrientation(a, b)) << fmt::format("{} != {}", a, b)

CPU_TEST(Quaternion_Constructor)
{
    // Default constructor
    {
        quatf q;
        EXPECT_EQ(q.x, 0.f);
        EXPECT_EQ(q.y, 0.f);
        EXPECT_EQ(q.z, 0.f);
        EXPECT_EQ(q.w, 1.f);
    }

    // Constructor with 4 floats
    {
        quatf q(1.f, 2.f, 3.f, 4.f);
        EXPECT_EQ(q.x, 1.f);
        EXPECT_EQ(q.y, 2.f);
        EXPECT_EQ(q.z, 3.f);
        EXPECT_EQ(q.w, 4.f);
    }

    // Constructor with vector + scalar
    {
        quatf q(float3(1.f, 2.f, 3.f), 4.f);
        EXPECT_EQ(q.x, 1.f);
        EXPECT_EQ(q.y, 2.f);
        EXPECT_EQ(q.z, 3.f);
        EXPECT_EQ(q.w, 4.f);
    }

    // Identity
    {
        quatf q = quatf::identity();
        EXPECT_EQ(q.x, 0.f);
        EXPECT_EQ(q.y, 0.f);
        EXPECT_EQ(q.z, 0.f);
        EXPECT_EQ(q.w, 1.f);
    }
}

CPU_TEST(Quaternion_Access)
{
    {
        quatf q(1.f, 2.f, 3.f, 4.f);
        EXPECT_EQ(q[0], 1.f);
        EXPECT_EQ(q[1], 2.f);
        EXPECT_EQ(q[2], 3.f);
        EXPECT_EQ(q[3], 4.f);
    }
}

CPU_TEST(Quaternion_Operator)
{
    // Unary + operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = +q1;
        EXPECT_EQ(q2, quatf(1.f, 2.f, 3.f, 4.f));
    }

    // Unary - operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = -q1;
        EXPECT_EQ(q2, quatf(-1.f, -2.f, -3.f, -4.f));
    }

    // Binary + operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(2.f, 3.f, 4.f, 5.f);
        quatf q3 = q1 + q2;
        EXPECT_EQ(q3, quatf(3.f, 5.f, 7.f, 9.f));
        quatf q4 = q1 + 2.f;
        EXPECT_EQ(q4, quatf(3.f, 4.f, 5.f, 6.f));
        quatf q5 = 2.f + q1;
        EXPECT_EQ(q5, quatf(3.f, 4.f, 5.f, 6.f));
    }

    // Binary - operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(2.f, 3.f, 4.f, 5.f);
        quatf q3 = q1 - q2;
        EXPECT_EQ(q3, quatf(-1.f, -1.f, -1.f, -1.f));
        quatf q4 = q1 - 2.f;
        EXPECT_EQ(q4, quatf(-1.f, 0.f, 1.f, 2.f));
        quatf q5 = 2.f - q1;
        EXPECT_EQ(q5, quatf(1.f, 0.f, -1.f, -2.f));
    }

    // Binary * operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = q1 * 2.f;
        EXPECT_EQ(q2, quatf(2.f, 4.f, 6.f, 8.f));
        quatf q3 = 3.f * q1;
        EXPECT_EQ(q3, quatf(3.f, 6.f, 9.f, 12.f));
    }

    // Binary / operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = q1 / 2.f;
        EXPECT_EQ(q2, quatf(0.5f, 1.f, 1.5f, 2.f));
    }

    // Binary == operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(1.f, 2.f, 3.f, 4.f);
        EXPECT_EQ(q1 == q2, bool4(true, true, true, true));
        quatf q3(1.f, 2.f, 3.f, 5.f);
        EXPECT_EQ(q1 == q3, bool4(true, true, true, false));
    }

    // Binary != operator
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(1.f, 2.f, 3.f, 4.f);
        EXPECT_EQ(q1 != q2, bool4(false, false, false, false));
        quatf q3(1.f, 2.f, 3.f, 5.f);
        EXPECT_EQ(q1 != q3, bool4(false, false, false, true));
    }
}

CPU_TEST(Quaternion_Multiply)
{
    // Quaternion / quaternion multiplication
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(2.f, 3.f, 4.f, 5.f);
        quatf q3 = mul(q1, q2);
        EXPECT_EQ(q3, quatf(12.f, 24.f, 30.f, 0.f));
    }

    // Quaternion / vector multiplication
    {
        quatf q1(2.f, 3.f, 4.f, 5.f);
        float3 v1(2.f, 3.f, 4.f);
        float3 v2 = mul(q1, v1);
        EXPECT_EQ(v2, float3(2.f, 3.f, 4.f));
    }
}

CPU_TEST(Quaternion_FloatChecks)
{
    // isfinite
    {
        quatf q1(0.f, 0.f, 0.f, 0.f);
        EXPECT_EQ(isfinite(q1), bool4(true, true, true, true));
        quatf q2(std::numeric_limits<float>::infinity(), 0.f, 0.f, std::numeric_limits<float>::infinity());
        EXPECT_EQ(isfinite(q2), bool4(false, true, true, false));
    }
    // isinf
    {
        quatf q1(0.f, 0.f, 0.f, 0.f);
        EXPECT_EQ(isinf(q1), bool4(false, false, false, false));
        quatf q2(std::numeric_limits<float>::infinity(), 0.f, 0.f, std::numeric_limits<float>::infinity());
        EXPECT_EQ(isinf(q2), bool4(true, false, false, true));
    }
    // isnan
    {
        quatf q1(0.f, 0.f, 0.f, 0.f);
        EXPECT_EQ(isnan(q1), bool4(false, false, false, false));
        quatf q2(std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f, std::numeric_limits<float>::quiet_NaN());
        EXPECT_EQ(isnan(q2), bool4(true, false, false, true));
    }
}

CPU_TEST(Quaternion_Functions)
{
    // dot product
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(2.f, 3.f, 4.f, 5.f);
        float d = dot(q1, q2);
        EXPECT_EQ(d, 40.f);
    }

    // cross product
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(2.f, 3.f, 4.f, 5.f);
        quatf q3 = cross(q1, q2);
        EXPECT_EQ(q3, quatf(12.f, 24.f, 30.f, 0.f));
    }

    // length
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        float l = length(q1);
        EXPECT_EQ(l, sqrtf(30.f));
    }

    // normalize
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = normalize(q1);
        EXPECT_EQ(q2, q1 * (1.f / sqrtf(30.f)));
    }

    // conjugate
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = conjugate(q1);
        EXPECT_EQ(q2, quatf(-1.f, -2.f, -3.f, 4.f));
    }

    // inverse
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2 = inverse(q1);
        EXPECT_EQ(q2, quatf(-1.f / 30.f, -2.f / 30.f, -3.f / 30.f, 4.f / 30.f));
    }

    // lerp
    {
        quatf q1(1.f, 2.f, 3.f, 4.f);
        quatf q2(2.f, 3.f, 4.f, 5.f);
        quatf q3 = lerp(q1, q2, 0.5f);
        EXPECT_EQ(q3, quatf(1.5f, 2.5f, 3.5f, 4.5f));
    }
}

CPU_TEST(Quaternion_slerp)
{
    auto test = [&](float angle, float t, float expectedAngle)
    {
        quatf from = quatf::identity();
        quatf to = math::quatFromAngleAxis(math::radians(angle), float3(0, 1, 0));
        quatf expected = math::quatFromAngleAxis(math::radians(expectedAngle), float3(0, 1, 0));

        quatf q1 = slerp(from, to, t);
        EXPECT_ALMOST_EQ_ORIENTATION(q1, expected);

        quatf q2 = slerp(to, from, 1.f - t);
        EXPECT_ALMOST_EQ_ORIENTATION(q2, expected);
    };

    // Basic
    test(+160, 0.375f, +60);
    test(-160, 0.375f, -60);

    // Shorting
    test(+320, 0.375f, -15); // Mathematically, should be +120
    test(-320, 0.375f, +15); // Mathematically, should be -120

    // Lengthening short way
    test(320, 1.5f, -60); // Mathematically, should be 480 (ie -240)

    // Lengthening
    test(+70, 3, +210);
    test(-70, 3, -210);

    // Edge case that often causes NaNs
    test(0, .5f, 0);

    // This edge case is ill-defined for "intuitive" slerp and can't be tested.
    // test(180, .25f, 45);

    // Conversely, this edge case is well-defined for "intuitive" slerp.
    // For mathematical slerp, the axis is ill-defined and can take many values.
    test(360, .25f, 0);
}

CPU_TEST(Quaternion_Euler)
{
    // pitch 90 degrees
    {
        // quatf q = math::quatFromAngleAxis(math::radians(90.f), float3(1.f, 0.f, 0.f));
        quatf q = quatf(std::sqrt(0.5f), 0.f, 0.f, std::sqrt(0.5f));
        EXPECT_ALMOST_EQ(pitch(q), math::radians(90.f));
    }

    // pitch -60 degrees
    {
        // quatf q = math::quatFromAngleAxis(math::radians(-60.f), float3(1.f, 0.f, 0.f));
        quatf q = quatf(-0.5f, 0.f, 0.f, std::sqrt(0.75f));
        EXPECT_ALMOST_EQ(pitch(q), math::radians(-60.f));
    }

    // yaw 90 degrees
    {
        // quatf q = math::quatFromAngleAxis(math::radians(90.f), float3(0.f, 1.f, 0.f));
        quatf q = quatf(0.f, std::sqrt(0.5f), 0.f, std::sqrt(0.5f));
        EXPECT_ALMOST_EQ_EPS(yaw(q), math::radians(90.f), 1e-3f);
    }

    // yaw -60 degrees
    {
        // quatf q = math::quatFromAngleAxis(math::radians(-60.f), float3(0.f, 1.f, 0.f));
        quatf q = quatf(0.f, -0.5f, 0.f, std::sqrt(0.75f));
        EXPECT_ALMOST_EQ(yaw(q), math::radians(-60.f));
    }

    // roll 90 degrees
    {
        // quatf q = math::quatFromAngleAxis(math::radians(90.f), float3(0.f, 0.f, 1.f));
        quatf q = quatf(0.f, 0.f, std::sqrt(0.5f), std::sqrt(0.5f));
        EXPECT_ALMOST_EQ(roll(q), math::radians(90.f));
    }

    // roll -60 degrees
    {
        // quatf q = math::quatFromAngleAxis(math::radians(-60.f), float3(0.f, 0.f, 1.f));
        quatf q = quatf(0.f, 0.f, -0.5f, std::sqrt(0.75f));
        EXPECT_ALMOST_EQ(roll(q), math::radians(-60.f));
    }

    // eulerAngles
    {
        // quatf q1 = math::quatFromAngleAxis(math::radians(-60.f), float3(1.f, 0.f, 0.f));
        quatf q1 = quatf(-0.5f, 0.f, 0.f, std::sqrt(0.75f));
        EXPECT_ALMOST_EQ(eulerAngles(q1), float3(math::radians(-60.f), 0.f, 0.f));

        // quatf q2 = math::quatFromAngleAxis(math::radians(-60.f), float3(0.f, 1.f, 0.f));
        quatf q2 = quatf(0.f, -0.5f, 0.f, std::sqrt(0.75f));
        EXPECT_ALMOST_EQ(eulerAngles(q2), float3(0.f, math::radians(-60.f), 0.f));

        // quatf q3 = math::quatFromAngleAxis(math::radians(-60.f), float3(0.f, 0.f, 1.f));
        quatf q3 = quatf(0.f, 0.f, -0.5f, std::sqrt(0.75f));
        EXPECT_ALMOST_EQ(eulerAngles(q3), float3(0.f, 0.f, math::radians(-60.f)));

        // quatf q4 = mul(mul(q3, q2), q1);
        quatf q4 = quatf(-0.591506362f, -0.158493653f, -0.591506362f, 0.524519026f);
        EXPECT_ALMOST_EQ(eulerAngles(q4), float3(math::radians(-60.f), math::radians(-60.f), math::radians(-60.f)));
    }
}

CPU_TEST(Quaternion_quatFromAngleAxis)
{
    // 90 degrees X-axis
    {
        quatf q1 = math::quatFromAngleAxis(math::radians(90.f), float3(1.f, 0.f, 0.f));
        quatf q2 = quatf(std::sqrt(0.5f), 0.f, 0.f, std::sqrt(0.5f));
        EXPECT_ALMOST_EQ(q1, q2);
    }

    // -60 degrees X-axis
    {
        quatf q1 = math::quatFromAngleAxis(math::radians(-60.f), float3(1.f, 0.f, 0.f));
        quatf q2 = quatf(-0.5f, 0.f, 0.f, std::sqrt(0.75f));
        EXPECT_ALMOST_EQ(q1, q2);
    }

    // 90 degrees Y-axis
    {
        quatf q1 = math::quatFromAngleAxis(math::radians(90.f), float3(0.f, 1.f, 0.f));
        quatf q2 = quatf(0.f, std::sqrt(0.5f), 0.f, std::sqrt(0.5f));
        EXPECT_ALMOST_EQ(q1, q2);
    }

    // -60 degrees Y-axis
    {
        quatf q1 = math::quatFromAngleAxis(math::radians(-60.f), float3(0.f, 1.f, 0.f));
        quatf q2 = quatf(0.f, -0.5f, 0.f, std::sqrt(0.75f));
        EXPECT_ALMOST_EQ(q1, q2);
    }

    // 90 degrees Z-axis
    {
        quatf q1 = math::quatFromAngleAxis(math::radians(90.f), float3(0.f, 0.f, 1.f));
        quatf q2 = quatf(0.f, 0.f, std::sqrt(0.5f), std::sqrt(0.5f));
        EXPECT_ALMOST_EQ(q1, q2);
    }

    // -60 degrees Z-axis
    {
        quatf q1 = math::quatFromAngleAxis(math::radians(-60.f), float3(0.f, 0.f, 1.f));
        quatf q2 = quatf(0.f, 0.f, -0.5f, std::sqrt(0.75f));
        EXPECT_ALMOST_EQ(q1, q2);
    }
}

CPU_TEST(Quaternion_quatFromRotationBetweenVectors)
{
    auto test = [&](float3 v1, float3 axis, float angle)
    {
        quatf q1 = math::quatFromAngleAxis(math::radians(angle), axis);
        float3 v2 = transformVector(q1, v1);
        quatf q2 = math::quatFromRotationBetweenVectors(v1, v2);
        float3 v3 = transformVector(q2, v1);
        EXPECT_ALMOST_EQ(v2, v3);
    };

    test(float3(0.f, 1.f, 0.f), float3(1.f, 0.f, 0.f), 90);
    test(float3(0.f, 0.f, 1.f), float3(1.f, 0.f, 0.f), -135);

    test(float3(1.f, 0.f, 0.f), float3(0.f, 1.f, 0.f), 90);
    test(float3(0.f, 0.f, 1.f), float3(0.f, 1.f, 0.f), -135);

    test(float3(1.f, 0.f, 0.f), float3(0.f, 0.f, 1.f), 90);
    test(float3(0.f, 1.f, 0.f), float3(0.f, 0.f, 1.f), -135);

    test(float3(1.f, 0.f, 0.f), normalize(float3(1.f, 1.f, 1.f)), 45);
    test(float3(0.f, 1.f, 0.f), normalize(float3(1.f, 1.f, 1.f)), 90);
    test(float3(0.f, 0.f, 1.f), normalize(float3(1.f, 1.f, 1.f)), 135);
}

CPU_TEST(Quaternion_quatFromEulerAngles)
{
    // 90 degrees X-axis
    {
        quatf q1 = math::quatFromEulerAngles(float3(math::radians(90.f), 0.f, 0.f));
        quatf q2 = quatf(std::sqrt(0.5f), 0.f, 0.f, std::sqrt(0.5f));
        EXPECT_ALMOST_EQ(q1, q2);
    }

    // -60 degrees X-axis
    {
        quatf q1 = math::quatFromEulerAngles(float3(math::radians(-60.f), 0.f, 0.f));
        quatf q2 = quatf(-0.5f, 0.f, 0.f, std::sqrt(0.75f));
        EXPECT_ALMOST_EQ(q1, q2);
    }

    // 90 degrees Y-axis
    {
        quatf q1 = math::quatFromEulerAngles(float3(0.f, math::radians(90.f), 0.f));
        quatf q2 = quatf(0.f, std::sqrt(0.5f), 0.f, std::sqrt(0.5f));
        EXPECT_ALMOST_EQ(q1, q2);
    }

    // -60 degrees Y-axis
    {
        quatf q1 = math::quatFromEulerAngles(float3(0.f, math::radians(-60.f), 0.f));
        quatf q2 = quatf(0.f, -0.5f, 0.f, std::sqrt(0.75f));
        EXPECT_ALMOST_EQ(q1, q2);
    }

    // 90 degrees Z-axis
    {
        quatf q1 = math::quatFromEulerAngles(float3(0.f, 0.f, math::radians(90.f)));
        quatf q2 = quatf(0.f, 0.f, std::sqrt(0.5f), std::sqrt(0.5f));
        EXPECT_ALMOST_EQ(q1, q2);
    }

    // -60 degrees Z-axis
    {
        quatf q1 = math::quatFromEulerAngles(float3(0.f, 0.f, math::radians(-60.f)));
        quatf q2 = quatf(0.f, 0.f, -0.5f, std::sqrt(0.75f));
        EXPECT_ALMOST_EQ(q1, q2);
    }
}

CPU_TEST(Quaternion_quatFromMatrix)
{
    auto test = [&](int x, int y, int z)
    {
        float3 eulerAngles = radians(float3(x, y, z));
        quatf q1 = math::quatFromEulerAngles(eulerAngles);
        float3x3 m = math::matrixFromQuat(q1);
        quatf q2 = math::quatFromMatrix(m);
        EXPECT_ALMOST_EQ(q1, q2);
        EXPECT_ALMOST_EQ_EPS(eulerAngles, math::eulerAngles(q2), 1e-3f);
    };

    test(0, 0, 0);
    test(90, 0, 0);
    test(0, 90, 0);
    test(0, 0, 90);
    test(-45, 0, 0);
    test(0, -45, 0);
    test(0, 0, -45);
    test(10, 20, 30);
    test(-30, -20, -10);
}

} // namespace Falcor
