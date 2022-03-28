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
import Utils.Math.IntervalArithmetic;

 /** Blending of two "objects" based on a radius. Equation 17.57 in RTR4, and also in the Graphics Codex by Morgan McGuire.
 */
float sdfBlend(float distance1, float distance2, float radius)
{
    float h = saturate(0.5f + 0.5f * (distance2 - distance1) / radius);
    float d = lerp(distance2, distance1, h) - radius * h * (1.0f - h);
    return d;
}

float2 sdfIntervalBlend(float2 distance1, float2 distance2, float radius)
{
    float2 h = ivlSaturate(ivlAdd(ivlMul(ivlDiv(ivlSub(distance2, distance1), radius), 0.5f), 0.5f));
    float2 d = ivlAdd(ivlMul(ivlAdd(ivlNegate(h), 1.0f), distance2), ivlSub(ivlMul(h, distance1), ivlMul(ivlMul(h, ivlAdd(ivlNegate(h), 1.0f)), radius)));
    return d;
}

/** smin() is used by sdfSmoothUnion() below.
    http://iquilezles.org/www/articles/smin/smin.htm
*/
float smin(float a, float b, float k)
{
    float h = max(k - abs(a - b), 0.0f);
    return min(a, b) - h * h * 0.25f / k;
}

float2 intervalSMin(float2 a, float2 b, float k)
{
    float2 h = ivlMax(ivlAdd(ivlNegate(ivlAbs(ivlSub(a, b))), k), 0.0f);
    h = ivlDiv(ivlMul(ivlPosSquare(h), 0.25f), k);

    return ivlSub(ivlMin(a, b), h);
}

/** smax() is used by sdfSmoothIntersection() below.
    http://iquilezles.org/www/articles/smin/smin.htm
*/
float smax(float a, float b, float k)
{
    float h = max(k - abs(a - b), 0.0f);
    return max(a, b) + h * h * 0.25f / k;
}

float2 intervalSMax(float2 a, float2 b, float k)
{
    float2 h = ivlMax(ivlAdd(ivlNegate(ivlAbs(ivlSub(a, b))), k), 0.0f);
    h = ivlDiv(ivlMul(ivlPosSquare(h), 0.25f), k);

    return ivlAdd(ivlMax(a, b), h);
}

/** Union, intersection, and subtraction operators.
*/
float sdfUnion(float a, float b)
{
    return min(a, b);
}

float sdfIntersection(float a, float b)
{
    return max(a, b);
}

float sdfSubtraction(float a, float b)
{
    return max(a, -b);
}

float2 sdfIntervalUnion(float2 a, float2 b)
{
    return ivlMin(a, b);
}

float2 sdfIntervalIntersection(float2 a, float2 b)
{
    return ivlMax(a, b);
}

float2 sdfIntervalSubtraction(float2 a, float2 b)
{
    return ivlMax(a, ivlNegate(b));
}

/** Smooth union.
*/
float sdfSmoothUnion(float a, float b, float k)
{
    return smin(a, b, k);
}

/** Smooth intersection.
*/
float sdfSmoothIntersection(float a, float b, float k)
{
    return smax(a, b, k);
}

/** Smooth subtraction.
*/
float sdfSmoothSubtraction(float a, float b, float k)
{
    return smax(a, -b, k);
}

/** Interval Smooth union.
*/
float2 sdfIntervalSmoothUnion(float2 a, float2 b, float k)
{
    return intervalSMin(a, b, k);
}

/** Interval Smooth intersection.
*/
float2 sdfIntervalSmoothIntersection(float2 a, float2 b, float k)
{
    return intervalSMax(a, b, k);
}

/** Interval Smooth subtraction.
*/
float2 sdfIntervalSmoothSubtraction(float2 a, float2 b, float k)
{
    return intervalSMax(a, ivlNegate(b), k);
}
