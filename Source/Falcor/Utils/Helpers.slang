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

import Utils.Math.MathHelpers;

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
    \param[out] t Distance to the closest intersection.
    \return True if the ray intersects the sphere.
*/
bool intersectRaySphere(float3 rayOrigin, float3 rayDir, float3 sphereCenter, float sphereRadius, out float t)
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
        t = 0.f;
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

    t = tc;
    return true;
}

/** Ray-AABB intersection.
    \param[in] rayOrigin Ray origin.
    \param[in] rayDir Ray direction (does not have to be normalized).
    \param[in] aabbMin AABB minimum.
    \param[in] aabbMax AABB maximum.
    \param[out] nearFar Returns intersection interval along ray.
    \return True if the ray intersects the AABB.
*/
bool intersectRayAABB(const float3 rayOrigin, const float3 rayDir, const float3 aabbMin, const float3 aabbMax, out float2 nearFar)
{
    const float3 invDir = 1.f / rayDir;
    const float3 lo = (aabbMin - rayOrigin) * invDir;
    const float3 hi = (aabbMax - rayOrigin) * invDir;
    const float3 tmin = min(lo, hi), tmax = max(lo, hi);
    nearFar.x = max(0.f, max(tmin.x, max(tmin.y, tmin.z)));
    nearFar.y = min(tmax.x, min(tmax.y, tmax.z));
    return nearFar.x <= nearFar.y;
}

/** Ray intersection against linear swept sphere based on [Han et al. 2019], Ray Tracing Generalized Tube Primitives: Method and Applications.
    Paper link: http://www.sci.utah.edu/publications/Han2019a/tubes-final.pdf
    \param[in] rayOrigin Ray origin position.
    \param[in] rayDir Unit ray direction vector.
    \param[in] sphereA Sphere (3D position + radius) at one end point.
    \param[in] sphereB Sphere at the other end point.
    \param[in] useSphereJoints Indicating whether we test ray-sphere intersection at curve joints or not.
    \param[out] result The closest intersection distance t, and a parameter u for linear interpolation (between 0 and 1).
    \return True if the ray intersects the linear swept sphere segment.
*/
bool intersectLinearSweptSphereHan19(float3 rayOrigin, float3 rayDir, float4 sphereA, float4 sphereB, bool useSphereJoints, out float2 result)
{
    result = float2(FLT_MAX);

    bool reversed = false;
    if (sphereA.w > sphereB.w)
    {
        float4 tmp = sphereA;
        sphereA = sphereB;
        sphereB = tmp;
        reversed = true;
    }

    const float3 P1 = sphereA.xyz;
    const float3 P2 = sphereB.xyz;
    const float r1 = sphereA.w;
    const float r2 = sphereB.w;

    // Perpendicular distance to closest of (p0, p1) minus the max radius.
    float t0 = min(dot(rayDir, P1 - rayOrigin), dot(rayDir, P2 - rayOrigin)) - max(r1, r2);
    t0 = max(0.f, t0);
    // For better numerical stability, push the ray to be as close as possible to the curve.
    rayOrigin += t0 * rayDir;

    if (useSphereJoints)
    {
        // Intersecting two sphere endcaps.
        float t;
        if (intersectRaySphere(rayOrigin, rayDir, P1, r1, t))
        {
            if (t < result.x)
            {
                result.x = t;
                result.y = (reversed ? 1.f : 0.f);
            }
        }
        if (intersectRaySphere(rayOrigin, rayDir, P2, r2, t))
        {
            if (t < result.x)
            {
                result.x = t;
                result.y = (reversed ? 0.f : 1.f);
            }
        }
    }

    // Intersecting cone.
    float3 C = P2 - P1;
    const float lengthC = length(C);
    C /= lengthC;

    const float p1 = lengthC * r1 / (r2 - r1);
    const float p2 = p1 * r2 / r1;
    const float3 A = P1 - p1 * C;
    const float z1 = p1 - r1 * r1 / p1;
    const float z2 = p2 - r2 * r2 / p2;
    const float w = p2 * r2 / sqrt(p2 * p2 - r2 * r2);

    const float3 vz = C;
    const float3 vx = perp_stark(vz);
    const float3 vy = cross(vz, vx);

    const float tmp1 = 1.f / z2;
    const float tmp2 = p2 / w;

    // Row-major matrix.
    float4x4 M =
    {
        tmp1 * tmp2 * float4(vx, -dot(vx, A)),
        tmp1 * tmp2 * float4(vy, -dot(vy, A)),
        tmp1 * float4(vz, -dot(vz, A)),
        float4(0, 0, 0, 1)
    };

    const float zCap = z1 * tmp1;
    const float3 rayOriginXf = mul(M, float4(rayOrigin, 1.f)).xyz;
    const float3 rayDirXf = mul((float3x3)M, rayDir);

    const float a = rayDirXf.x * rayDirXf.x + rayDirXf.y * rayDirXf.y - rayDirXf.z * rayDirXf.z;
    const float b = 2.f * (rayOriginXf.x * rayDirXf.x + rayOriginXf.y * rayDirXf.y - rayOriginXf.z * rayDirXf.z);
    const float c = rayOriginXf.x * rayOriginXf.x + rayOriginXf.y * rayOriginXf.y - rayOriginXf.z * rayOriginXf.z;
    const float disc = b * b - 4 * a * c;
    if (disc >= 0)
    {
        const float sqrtDisc = sqrt(disc);
        const float inv2a = 0.5f / a;

        [unroll]
        for (int i = 0; i < 2; i++)
        {
            float t = (-b + (i * 2.f - 1.f) * sqrtDisc) * inv2a;
            if (t >= 0 && t < result.x)
            {
                // Check if z is in the valid range.
                const float z = rayOriginXf.z + t * rayDirXf.z;
                if (z >= zCap && z <= 1.f)
                {
                    const float u = (z - zCap) / (1.f - zCap);
                    result.x = t;
                    result.y = (reversed ? 1.f - u : u);
                }
            }
        }
    }

    result.x += t0;
    return (result.x < FLT_MAX);
}

/*******************************************************************
                    Shading
*******************************************************************/

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
