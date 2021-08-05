/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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

/** Blending of two "objects" based on a radius. Equation 17.57 in RTR4, and also in the Graphics Codex by Morgan McGuire.
 */
float sdfBlend(float distance1, float distance2, float radius)
{
    float h = saturate(0.5f + 0.5f * (distance2 - distance1) / radius);
    float d = (1.0f - h) * distance2 + h * distance1 - radius * h * (1.0f - h);
    return d;
}

/** smin() is used by sdfSmoothUnion() below.
*/
// http://iquilezles.org/www/articles/smin/smin.htm
float smin(float a, float b, float k)
{
    float h = max(k - abs(a - b), 0.0f);
    return min(a, b) - h * h * 0.25f / k;
}


/** smax() is used by sdfSmoothIntersection() below.
*/
// http://iquilezles.org/www/articles/smin/smin.htm
float smax(float a, float b, float k)
{
    float h = max(k - abs(a - b), 0.0f);
    return max(a, b) + h * h * 0.25f / k;
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
