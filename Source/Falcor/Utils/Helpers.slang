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
#include "Utils/Math/MathConstants.slangh"
#include "Scene/Material/MaterialDefines.slangh"

__exported import Utils.Color.ColorHelpers;

/*******************************************************************
    Spherical map sampling
*******************************************************************/
float2 dirToSphericalCrd(float3 direction)
{
    float3 p = normalize(direction);
    float2 uv;
    uv.x = (1 + atan2(-p.z, p.x) / M_PI) * 0.5;
    uv.y = acos(p.y) / M_PI;
    return uv;
}

float3 sphericalCrdToDir(float2 uv)
{
    float phi = M_PI * uv.y;
    float theta = M_2PI * uv.x - (M_PI / 2.0f);

    float3 dir;
    dir.x = sin(phi) * sin(theta);
    dir.y = cos(phi);
    dir.z = sin(phi) * cos(theta);

    return normalize(dir);
}

/*******************************************************************
    Sample Patterns
*******************************************************************/

float radicalInverse(uint i)
{
    i = (i & 0x55555555) << 1 | (i & 0xAAAAAAAA) >> 1;
    i = (i & 0x33333333) << 2 | (i & 0xCCCCCCCC) >> 2;
    i = (i & 0x0F0F0F0F) << 4 | (i & 0xF0F0F0F0) >> 4;
    i = (i & 0x00FF00FF) << 8 | (i & 0xFF00FF00) >> 8;
    i = (i << 16) | (i >> 16);
    return float(i) * 2.3283064365386963e-10f;
}

float2 getHammersley(uint i, uint N)
{
    return float2(float(i) / float(N), radicalInverse(i));
}

/*******************************************************************
    Ray tracing
*******************************************************************/

/** Computes new ray origin based on hit position to avoid self-intersections.
    The function assumes that the hit position has been computed by barycentric
    interpolation, and not from the ray t which is less accurate.

    The method is described in Ray Tracing Gems, Chapter 6, "A Fast and Robust
    Method for Avoiding Self-Intersection" by Carsten Wächter and Nikolaus Binder.

    \param[in] pos Ray hit position.
    \param[in] normal Face normal of hit surface (normalized). The offset will be in the positive direction.
    \return Ray origin of the new ray.
*/
float3 computeRayOrigin(float3 pos, float3 normal)
{
    const float origin = 1.f / 32.f;
    const float fScale = 1.f / 65536.f;
    const float iScale = 256.f;

    // Per-component integer offset to bit representation of fp32 position.
    int3 iOff = int3(normal * iScale);
    float3 iPos = asfloat(asint(pos) + (pos < 0.f ? -iOff : iOff));

    // Select per-component between small fixed offset or above variable offset depending on distance to origin.
    float3 fOff = normal * fScale;
    return abs(pos) < origin ? pos + fOff : iPos;
}

/** Ray-sphere intersection.
    This function implements the standard analytic test with improvements to floating-point precision
    and returns the closest hit.
    \param[in] rayOrigin Ray origin.
    \param[in] rayDir Ray direction (does not have to be normalized).
    \param[in] center Sphere center.
    \param[in] radius Sphere radius.
    \param[in] intersectionPos Position on the sphere for the closest intersection (if any).
    \return True if the ray intersects the sphere.
*/
bool intersectRaySphere(float3 rayOrigin, float3 rayDir, float3 sphereCenter, float sphereRadius, out float3 intersectionPos)
{
    // Implementation is taken from Chapter 7 of Ray-Tracing Gems
    float3 f = rayOrigin - sphereCenter;
    float a = dot(rayDir, rayDir);
    float b = dot(-f, rayDir);
    float discriminant = sphereRadius * sphereRadius - dot(f + b / a * rayDir, f + b / a * rayDir);

    // Negative discriminant means ray missed sphere.
    if (discriminant < 0.f) return false;

    // If b and discriminant are both 0, then the ray's origin lies on the sphere
    if (b == 0 && discriminant == 0)
    {
        intersectionPos = rayOrigin;
        return true;
    }

    // There are two solutions t0 and t1, but one or both may be negative.
    float c = dot(f, f) - sphereRadius * sphereRadius;
    float signB = (b < 0) ? -1 : 1;
    float q = b + signB * sqrt(a * discriminant);
    float t0 = c / q;
    float t1 = q / a;

    float tc = t0 < 0.f ? t1 : t0; // tc is the closest hit we care about
    if (tc < 0.f) return false;

    intersectionPos = rayOrigin + tc * rayDir;
    return true;
}

/*******************************************************************
                    Shading
*******************************************************************/

float4 applyAmbientOcclusion(float4 color, Texture2D aoTex, SamplerState s, float2 texC)
{
    float aoFactor = aoTex.SampleLevel(s, texC, 0).r;
    return float4(color.rgb * aoFactor, color.a);
}

// TODO: this function is broken an may return negative values.
float getMetallic(float3 diffuse, float3 spec)
{
    // This is based on the way that UE4 and Substance Painter 2 converts base+metallic+specular level to diffuse/spec colors
    // We don't have the specular level information, so the assumption is that it is equal to 0.5 (based on the UE4 documentation)
    // Note that I'm using the luminance here instead of the actual colors. The reason is that there's no guaraentee that all RGB channels will end up with the same metallic value
    float d = luminance(diffuse);
    float s = luminance(spec);
    if (s == 0) return 0;
    float a = 0.04;
    float b = s + d - 0.08;
    float c = 0.04 - s;
    float root = sqrt(b*b - 0.16*c);
    float m = (root - b) * 12.5;
    return m;
}

/*******************************************************************
                    Motion vectors
*******************************************************************/

/** Calculate screen-space motion vector.
    \param[in] pixelCrd Sample in current frame expressed in pixel coordinates with origin in the top-left corner.
    \param[in] prevPosH Sample in previous frame expressed in homogeneous clip space coordinates. Note that the definition differs between D3D12 and Vulkan.
    \param[in] renderTargetDim Render target dimension in pixels.
    \return Motion vector pointing from current to previous position expressed in sceen space [0,1] with origin in the top-left corner.
*/
float2 calcMotionVector(float2 pixelCrd, float4 prevPosH, float2 renderTargetDim)
{
    float2 prevCrd = prevPosH.xy / prevPosH.w;
#ifdef FALCOR_VK
    prevCrd *= float2(0.5, 0.5);
#else
    prevCrd *= float2(0.5, -0.5);
#endif
    prevCrd += 0.5f;
    float2 normalizedCrd = pixelCrd / renderTargetDim;
    return prevCrd - normalizedCrd;
}

/*******************************************************************
                    Alpha test
*******************************************************************/
// Evaluate alpha test and return true if point should be discarded
bool evalBasicAlphaTest(float alpha, float threshold)
{
    return alpha < threshold;
}
/*******************************************************************
                    Hashed Alpha Test
*******************************************************************/
// Evaluate alpha test and return true if point should be discarded
bool evalHashedAlphaTest(float alpha, float materialThreshold, float hashedThreshold)
{
    float compareTo = hashedThreshold <= 0 ? materialThreshold : saturate(hashedThreshold);
    return alpha < compareTo;
}

float sineHash(float2 p)
{
    return frac(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x))));
}

float sineHash3D(float3 p)
{
    return sineHash(float2(sineHash(p.xy), p.z));
}

float calculateHashedAlpha(float3 hashInputCoord, float hashScale, bool useAnisotropy)
{
    // Provide a decent default to our alpha threshold
    float alphaCompare = 0.5f;

    if (useAnisotropy)
    {
        //////  Anisotropic version

        // Find the discretized derivatives of our coordinates
        float3 anisoDeriv = max(abs(ddx(hashInputCoord)), abs(ddy(hashInputCoord)));
        float3 anisoScales = float3(0.707f / (hashScale * anisoDeriv.x),
                                    0.707f / (hashScale * anisoDeriv.y),
                                    0.707f / (hashScale * anisoDeriv.z));
        // Find log-discretized noise scales
        float3 scaleFlr = float3(exp2(floor(log2(anisoScales.x))),
                                 exp2(floor(log2(anisoScales.y))),
                                 exp2(floor(log2(anisoScales.z))));
        float3 scaleCeil = float3(exp2(ceil(log2(anisoScales.x))),
                                  exp2(ceil(log2(anisoScales.y))),
                                  exp2(ceil(log2(anisoScales.z))));
        // Compute alpha thresholds at our two noise scales
        float2 alpha = float2(sineHash3D(floor(scaleFlr * hashInputCoord)),
                              sineHash3D(floor(scaleCeil * hashInputCoord)));
        // Factor to linearly interpolate with
        float3 fractLoc = float3(frac(log2(anisoScales.x)),
                                 frac(log2(anisoScales.y)),
                                 frac(log2(anisoScales.z)));
        float2 toCorners = float2(length(fractLoc),
                                  length(float3(1.0f, 1.0f, 1.0f) - fractLoc));
        float lerpFactor = toCorners.x / (toCorners.x + toCorners.y);
        // Interpolate alpha threshold from noise at two scales
        float x = (1 - lerpFactor) * alpha.x + lerpFactor * alpha.y;
        // Pass into CDF to compute uniformly distrib threshold
        float a = min(lerpFactor, 1 - lerpFactor);
        float3 cases = float3(x * x / (2 * a * (1 - a)), (x - 0.5 * a) / (1 - a), 1.0 - ((1 - x) * (1 - x) / (2 * a * (1 - a))));
        // Find our final, uniformly distributed alpha threshold
        alphaCompare = (x < (1 - a)) ? ((x < a) ? cases.x : cases.y) : cases.z;
        alphaCompare = clamp(alphaCompare, 1.0e-6, 1.0f);
    }
    else
    {
        //////  Isotropic version

        // Find the discretized derivatives of our coordinates
        float maxDeriv = max(length(ddx(hashInputCoord)), length(ddy(hashInputCoord)));
        float pixScale = 1.0 / (hashScale * maxDeriv);
        // Find two nearest log-discretized noise scales
        float2 pixScales = float2(exp2(floor(log2(pixScale))), exp2(ceil(log2(pixScale))));
        // Compute alpha thresholds at our two noise scales
        float2 alpha = float2(sineHash3D(floor(pixScales.x * hashInputCoord)), sineHash3D(floor(pixScales.y * hashInputCoord)));
        // Factor to interpolate lerp with
        float lerpFactor = frac(log2(pixScale));
        // Interpolate alpha threshold from noise at two scales
        float x = (1 - lerpFactor) * alpha.x + lerpFactor * alpha.y;
        float a = min(lerpFactor, 1 - lerpFactor);
        // Pass into CDF to compute uniformly distrib threshold
        float3 cases = float3(x * x / (2 * a * (1 - a)), (x - 0.5 * a) / (1 - a), 1.0 - ((1 - x) * (1 - x) / (2 * a * (1 - a))));
        // Find our final, uniformly distributed alpha threshold
        alphaCompare = (x < (1 - a)) ? ((x < a) ? cases.x : cases.y) : cases.z;
        alphaCompare = clamp(alphaCompare, 1e-6f, 1.0f);
    }

    return alphaCompare;
}

/*******************************************************************
    Alpha test
*******************************************************************/

/** Evaluates alpha test, returning true if pixel should be discarded.
    \todo calculateHashedAlpha requires ddx/ddy, so cannot be used in RT mode.
*/
bool evalAlphaTest(uint matFlags, float alpha, float threshold, float3 posW)
{
    if (EXTRACT_ALPHA_MODE(matFlags) != AlphaModeMask) return false;

    float hashedAlphaScale = 1.0f;
#ifdef _HASHED_ALPHA_SCALE
    hashedAlphaScale = _HASHED_ALPHA_SCALE;
#endif

    /* Evaluate alpha test material modifier */
#ifdef _DEFAULT_ALPHA_TEST
    return evalBasicAlphaTest(alpha, threshold);
#elif defined(_HASHED_ALPHA_TEST_ANISOTROPIC)
    float hashedThreshold = calculateHashedAlpha(posW, hashedAlphaScale, true);
    return evalHashedAlphaTest(alpha, threshold, hashedThreshold);
#else
    // Default to isotropic hashed alpha test
    float hashedThreshold = calculateHashedAlpha(posW, hashedAlphaScale, false);
    return evalHashedAlphaTest(alpha, threshold, hashedThreshold);
#endif
}

/*******************************************************************
   	Random numbers based on Mersenne Twister
*******************************************************************/
uint rand_init(uint val0, uint val1, uint backoff = 16)
{
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0;

    for(uint n = 0; n < backoff; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1<<4)+0xa341316c)^(v1+s0)^((v1>>5)+0xc8013ea4);
        v1 += ((v0<<4)+0xad90777d)^(v0+s0)^((v0>>5)+0x7e95761e);
    }

    return v0;
}

float rand_next(inout uint s)
{
    uint LCG_A = 1664525u;
    uint LCG_C = 1013904223u;
    s = (LCG_A * s + LCG_C);
    return float(s & 0x00FFFFFF) / float(0x01000000);
}
