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
#pragma once
#include "Utils/Math/MathConstants.slangh"

 /** Utility functions for interval arithmetic.
     All interval functions operate on float2 where interval.x is the minimum value and interval.y is the maximum value.
 */

/** Combine two intervals into one.
*/
float2 ivlCombine(float2 a, float2 b)
{
    return float2(min(a.x, b.x), max(a.y, b.y));
}

/** Checks if the interval is valid.
*/
bool ivlValid(float2 interval)
{
    return interval.x <= interval.y;
}

/** Checks if the interval contains zero.
*/
bool ivlContainsZero(float2 interval)
{
    return interval.x <= 0.0f && interval.y >= 0.0f;
}

/** Calculates the min interval of an interval and a scalar.
*/
float2 ivlMin(float2 interval, float scalar)
{
    return min(interval, scalar);
}

/** Calculates the min interval of two intervals.
*/
float2 ivlMin(float2 intervalA, float2 intervalB)
{
    return min(intervalA, intervalB);
}

/** Calculates the max interval of an interval and a scalar.
*/
float2 ivlMax(float2 interval, float scalar)
{
    return max(interval, scalar);
}

/** Calculates the max interval of two intervals.
*/
float2 ivlMax(float2 intervalA, float2 intervalB)
{
    return max(intervalA, intervalB);
}

/** Clamps the interval between two scalars.
*/
float2 ivlClamp(float2 interval, float scalarMin, float scalarMax)
{
    return clamp(interval, scalarMin, scalarMax);
}

/** Clamps the interval between 0 and 1.
*/
float2 ivlSaturate(float2 interval)
{
    return clamp(interval, 0.0f, 1.0f);
}

/** Calculates the interval sum of an interval and a scalar.
*/
float2 ivlAdd(float2 interval, float scalar)
{
    return interval + scalar;
}

/** Calculates the interval sum of two intervals.
*/
float2 ivlAdd(float2 intervalA, float2 intervalB)
{
    return intervalA + intervalB;
}

/** Calculates the interval that results when subtracting a scalar from an interval.
*/
float2 ivlSub(float2 interval, float scalar)
{
    return interval - scalar;
}

/** Calculates the interval that results when subtracting intervalB from intervalA.
*/
float2 ivlSub(float2 intervalA, float2 intervalB)
{
    return intervalA - intervalB.yx;
}

/** Calculates the product of an interval and a scalar.
*/
float2 ivlMul(float2 interval, float scalar)
{
    float2 prod = interval * scalar;
    return float2(min(prod.x, prod.y), max(prod.x, prod.y));
}

/** Calculates the product of an interval and a scalar, the interval must be >= 0.
*/
float2 ivlPosMul(float2 interval, float scalar)
{
    return interval * scalar;
}

/** Calculates the product of an interval and a scalar, the interval must be <= 0.
*/
float2 ivlNegMul(float2 interval, float scalar)
{
    return interval.yx * scalar;
}

/** Calculates the product of two intervals.
*/
float2 ivlMul(float2 intervalA, float2 intervalB)
{
    float4 prod = float4(intervalA * intervalB, intervalA.yx * intervalB);
    return float2(min(min(min(prod.x, prod.y), prod.z), prod.w), max(max(max(prod.x, prod.y), prod.z), prod.w));
}

/** Calculates the product of two intervals, intervalA must be >= 0.
*/
float2 ivlPosMul(float2 intervalA, float2 intervalB)
{
    float4 prod = float4(intervalA * intervalB, intervalA.yx * intervalB);
    return float2(min(prod.x, prod.z), max(prod.y, prod.w));
}

/** Calculates the product of two intervals, intervalA must be <= 0.
*/
float2 ivlNegMul(float2 intervalA, float2 intervalB)
{
    float4 prod = float4(intervalA * intervalB, intervalA.yx * intervalB);
    return float2(min(prod.y, prod.w), max(prod.x, prod.z));
}

/** Calculates the product of two intervals, intervalA and intervalB must be >= 0.
*/
float2 ivlPosMulPos(float2 intervalA, float2 intervalB)
{
    return intervalA * intervalB;
}

/** Calculates the product of two intervals, intervalA and intervalB must be <= 0.
*/
float2 ivlNegMulNeg(float2 intervalA, float2 intervalB)
{
    return (intervalA * intervalB).yx;
}

/** Calculates the product of two intervals, intervalA must be >= 0 and intervalB must be <= 0.
*/
float2 ivlPosMulNeg(float2 intervalA, float2 intervalB)
{
    return intervalA.yx * intervalB;
}

/** Calculates the product of two intervals, intervalA must be <= 0 and intervalB must be >= 0.
*/
float2 ivlNegMulPos(float2 intervalA, float2 intervalB)
{
    return intervalA * intervalB.yx;
}

/** Calculates the interval that results when dividing an interval with a scalar.
*/
float2 ivlDiv(float2 interval, float scalar)
{
    float2 fract = interval / scalar;
    return float2(min(fract.x, fract.y), max(fract.x, fract.y));
}

/** Calculates the interval that results when dividing intervalA with intervalB.
*/
float2 ivlDiv(float2 intervalA, float2 intervalB)
{
    bool bContainsZero = ivlContainsZero(intervalB);

    float2 denom = bContainsZero ? float2(-FLT_MAX, FLT_MAX) : 1.0f / intervalB;
    return ivlMul(intervalA, denom);
}

/** Calculates the absolute value of an interval.
*/
float2 ivlAbs(float2 interval)
{
    return float2(max(max(interval.x, -interval.y), 0.0f), max(-interval.x, interval.y));
}

/** Calculates the absolute value of an interval, the interval must be <= 0.
*/
float2 ivlNegAbs(float2 interval)
{
    return -interval.yx;
}

/** Calculates the square an interval.
*/
float2 ivlSquare(float2 interval)
{
    interval = ivlAbs(interval);
    return interval * interval;
}

/** Calculates the square an interval, the interval must be >= 0.
*/
float2 ivlPosSquare(float2 interval)
{
    return interval * interval;
}

/** Calculates the square an interval, the interval must be <= 0.
*/
float2 ivlNegSquare(float2 interval)
{
    return interval.yx * interval.yx;
}

/** Calculates the square an interval, the interval must be <= 0.
*/
float2 ivlSqrt(float2 interval)
{
    return sqrt(interval);
}

/** Calculates the square an interval, the interval must be <= 0.
*/
float2 ivlNegate(float2 interval)
{
    return -interval.yx;
}

/** Calculates the length of a 2D vector with interval components.
*/
float2 ivlLength(float2 xInterval, float2 yInterval)
{
    return ivlSqrt( // sqrt((x * x) + (y * y))
                ivlAdd( // (x * x) + (y * y)
                    ivlSquare(xInterval),   // x * x
                    ivlSquare(yInterval)));  // y * y
}

/** Calculates the length of a 3D vector with interval components.
*/
float2 ivlLength(float2 xInterval, float2 yInterval, float2 zInterval)
{
    return ivlSqrt( // sqrt((x * x) + (y * y) + (z * z))
                ivlAdd( // (x * x) + (y * y) + (z * z)
                    ivlAdd(
                        ivlSquare(xInterval),   // x * x
                        ivlSquare(yInterval)),  // y * y
                        ivlSquare(zInterval))); // z * z
}

/** Calculates the length of a 3D vector with interval components.
*/
float2 ivlLength(float2 xInterval, float2 yInterval, float2 zInterval, float2 wInterval)
{
    return ivlSqrt( // sqrt((x * x) + (y * y) + (z * z) + (w * w))
                ivlAdd( // (x * x) + (y * y) + (z * z) + (w * w)
                ivlAdd(
                ivlAdd(
                        ivlSquare(xInterval),   // x * x
                        ivlSquare(yInterval)),  // y * y
                        ivlSquare(zInterval)),  // z * z
                        ivlSquare(wInterval))); // w * w
}
