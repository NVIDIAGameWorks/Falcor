/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/

#ifndef _FALCOR_SHADING_HELPERS_H_
#define _FALCOR_SHADING_HELPERS_H_

// Make sure we get the macros like `_fn` and `_ref`
// TODO: just eliminate these since we know this is pure Slang.
#include "HostDeviceData.h"

/*******************************************************************
					Sampling functions
*******************************************************************/

static const float M_PIf = 3.14159265359f;
static const float M_1_PIf = 0.31830988618379f;

_fn float2 sample_disk(float rnd1, float rnd2, float minRSq = 0.0f)
{
    float2 p;
    float r = sqrt(max(minRSq, rnd1));
    float phi = 2.0f * M_PIf * rnd2;
    p.x = r * cos(phi);
    p.y = r * sin(phi);
    return p;
}

_fn float3 sample_gauss(float rnd1, float rnd2)
{
    float3 p;
    float r = sqrt(-2.f * log(1.f - rnd1));
    float phi = 2.0f * M_PIf * rnd2;
    p.x = r * cos(phi);
    p.y = r * sin(phi);
    p.z = 0.0f;
    return p;
}

_fn inline float eval_gauss2_1d(float x2)
{
    return exp(-.5f * x2) / sqrt(2.0f * M_PIf);
}
_fn inline float eval_gauss2_2d(float x2)
{
    return exp(-.5f * x2) / (2.0f * M_PIf);
}
_fn inline float eval_gauss2_3d(float x2)
{
    return exp(-.5f * x2) / pow(2.0f * M_PIf, 1.5f);
}
// "band limit" (k sigma in frequency space): k / (2pi sigma)
_fn inline float eval_gauss_1d(float x)
{
    return eval_gauss2_1d(x * x);
}
_fn inline float eval_gauss_2d(float3 x)
{
    return eval_gauss2_2d(x.x * x.x + x.y * x.y);
}
_fn inline float eval_gauss_3d(float3 x)
{
    return eval_gauss2_3d(x.x * x.x + x.y * x.y + x.z * x.z);
}

// [0, 2] -> [1, 0]
// "band limit" (1st root in frequency space): 1/2
// Therefore x = t / step =F=> limit = 1/2 / step = 1/2 f_sampling
_fn inline float eval_cos2_wnd(float x)
{
    float c = abs(x) < 2.0f ? cos((M_PIf * 0.25f) * x) : 0.0f;
    return c * c;
}

_fn float3 cosine_sample_hemisphere(float rnd1, float rnd2)
{
    float3 p;
    p.xy = sample_disk(rnd1, rnd2);
    // Project up to hemisphere.
    p.z = sqrt(max(0.0f, 1.0f - p.x * p.x - p.y * p.y));
    return p;
}

_fn float3 uniform_sample_sphere(float rnd1, float rnd2)
{
    float3 p;
    p.z = 2.0f * rnd1 - 1.0f;
    float r = sqrt(max(0.0f, 1.0f - p.z * p.z));
    float phi = 2.0f * M_PIf * rnd2;
    p.x = r * cos(phi);
    p.y = r * sin(phi);
    return p;
}

_fn float3 uniform_sample_hemisphere(float rnd1, float rnd2)
{
    float3 p;
    p.z = rnd1;
    float r = sqrt(max(0.0f, 1.0f - p.z * p.z));
    float phi = 2.0f * M_PIf * rnd2;
    p.x = r * cos(phi);
    p.y = r * sin(phi);
    return p;
}
/**
	Random numbers based on Mersenne Twister
*/
_fn uint rand_init(uint val0, uint val1, uint backoff = 16)
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

_fn float rand_next(_ref(uint) s)
{
    uint LCG_A = 1664525u;
    uint LCG_C = 1013904223u;
    s = (LCG_A * s + LCG_C);
    return float(s & 0x00FFFFFF) / float(0x01000000);
}

/*******************************************************************
					Shading routines
*******************************************************************/

// Ideal specular refraction
_fn float3 refract(float3 I, float3 N, float eta)
{
    float dNI = dot(N, I);
    float k = 1.0f - eta * eta * (1.0f - dNI * dNI);
    float3 R;
    if (k < 0.0f) 
    {
        R = float3(0.0f, 0.0f, 0.0f);
    }
    else 
    {
        R = eta * I - (eta * dNI + sqrt(k)) * N;
    }
    return R;
}

/*******************************************************************
					Geometric routines
*******************************************************************/

void _fn createTangentFrame(float3 normal, _ref(float3) bitangent)
{
	if(abs(normal.x) > abs(normal.y))
		bitangent = float3(normal.z, 0.f, -normal.x) / length(float2(normal.x, normal.z));
	else
		bitangent = float3(0.f, normal.z, -normal.y) / length(float2(normal.y, normal.z));
}

void _fn reflectFrame(float3 n, float3 reflect, _ref(float3) t, _ref(float3) b)
{
    if(abs(dot(n, reflect)) > 0.999f)
        reflect = abs(n.z) < .8f ? float3(0.f, 0.f, 1.f) : float3(1.f, 0.f, 0.f);
    b = normalize(cross(n, reflect));
    t = cross(b, n);
}

/*******************************************************************
					Texturing routines
*******************************************************************/

float4 _fn sampleTexture(Texture2D t, SamplerState s, ShadingAttribs attr)
{
#ifndef _MS_USER_DERIVATIVES
    return t.SampleBias(s, attr.UV, attr.lodBias);
#else
	return t.SampleGrad(s, attr.UV, attr.DPDX, attr.DPDY);
#endif
}

float4 _fn sampleTexture(Texture2DArray t, SamplerState s, ShadingAttribs attr, int arrayIndex)
{
#ifndef _MS_USER_DERIVATIVES
    return t.SampleBias(s, float3(attr.UV, arrayIndex), attr.lodBias);
#else
    return t.SampleGrad(s, float3(attr.UV, arrayIndex), attr.DPDX, attr.DPDY);
#endif
}

float4 _fn evalTex(in uint32_t hasTexture, Texture2D tex, SamplerState s, ShadingAttribs attr, in float4 defaultValue)
{
#ifndef _MS_DISABLE_TEXTURES
	if(hasTexture != 0)
    {
        // MAT_CODE
        defaultValue = sampleTexture(tex, s, attr);
    }
#endif
	return defaultValue;
}

float4 _fn evalWithColor(in uint32_t hasTexture, Texture2D tex, SamplerState s, float4 color, ShadingAttribs attr)
{
	return evalTex(hasTexture, tex, s, attr, color);
}

/*******************************************************************
					Normal mapping
*******************************************************************/

float3 _fn normalToRGB(float3 normal)
{
	return normal * 0.5f + 0.5f;
}

float3 _fn RGBToNormal(float3 rgbval)
{
    return rgbval * 2.f - 1.f;
}

float3 _fn fromLocal(in float3 v, in float3 t, in float3 b, in float3 n)
{
    return t * v.x + b * v.y + n * v.z;
}

float3 _fn toLocal(in float3 v, in float3 t, in float3 b, in float3 n)
{
    return float3(dot(v, t), dot(v, b), dot(v, n));
}

void _fn applyNormalMap(in float3 texValue, _ref(float3) n, _ref(float3) t, _ref(float3) b)
{
	float3 normalMap = normalize(texValue);
	n = fromLocal(normalMap, normalize(t), normalize(b), normalize(n));
    // Orthogonalize tangent frame
    b = normalize(b - n * dot(b, n));
    t = normalize(cross(b, n));
}

#ifndef _MS_LEAN_MAPPING
void _fn perturbNormal(MaterialData mat, _ref(ShadingAttribs) attr, bool forceSample)
{
	if(forceSample || mat.desc.hasNormalMap != 0)
	{
		float3 texValue = sampleTexture(mat.textures.normalMap, mat.samplerState, attr).rgb;
        applyNormalMap(RGBToNormal(texValue), attr.N, attr.T, attr.B);
	}
}
#else
void applyLeanMap(in Texture2D leanMap, in SamplerState samplerState, inout ShadingAttribs shAttr)
{
    float4 t = sampleTexture(leanMap, samplerState, shAttr);
    // Reconstruct B 
    float2 B = 2 * t.xy - 1;
    float2 M = t.zw;

    // Apply normal
    float3 unnormalizedNormal = float3(B, 1);
    applyNormalMap(unnormalizedNormal, shAttr.N, shAttr.T, shAttr.B);
    // Reconstruct the diagonal covariance matrix
    float2 maxCov = max((0), M - B*B);     // Use only diagonal of covariance due to float2 aniso roughness

    [unroll]
    for (uint iLayer = 0; iLayer < MatMaxLayers; iLayer++)
    {
        if (shAttr.preparedMat.desc.layers[iLayer].type == MatNone) break;

        if (shAttr.preparedMat.desc.layers[iLayer].type != MatLambert)
        {
            float2 roughness = shAttr.preparedMat.values.layers[iLayer].roughness.xy;
            roughness = sqrt(roughness*roughness + maxCov); // Approximate convolution that works for all
            shAttr.preparedMat.values.layers[iLayer].roughness.rg = roughness;
        }
    }
}

void perturbNormal(MaterialData mat, inout ShadingAttribs shAttr, bool forceSample)
{
    if (mat.desc.hasNormalMap != 0)
    {
        applyLeanMap(gLeanMaps[mat.values.id], gLeanMapSampler, shAttr);
    }
}
#endif

// Note: explicit overload instead of default argument, at least until cross-compiler can handle defaults
void _fn perturbNormal(MaterialData mat, _ref(ShadingAttribs) attr)
{
	perturbNormal(mat, attr, false);
}


/*******************************************************************
					Alpha test
*******************************************************************/

bool _fn alphaTestEnabled(MaterialData mat)
{
    return mat.desc.hasAlphaMap != 0;
}

bool _fn alphaTestPassed(MaterialData mat, ShadingAttribs attr)
{
#ifndef _MS_DISABLE_ALPHA_TEST
    if(sampleTexture(mat.textures.alphaMap, mat.samplerState, attr).x < mat.values.alphaThreshold)
        return false;
#endif
    return true;
}

void _fn basicAlphaTest(MaterialData mat, ShadingAttribs attr)
{
#ifndef _MS_DISABLE_ALPHA_TEST
    if(alphaTestEnabled(mat))
    {
        if(!alphaTestPassed(mat, attr))
            discard;
    }
#endif
}

/*******************************************************************
                    Hashed Alpha Test
*******************************************************************/

bool _fn hashedAlphaTestPassed(MaterialData mat, ShadingAttribs attr, float alphaThreshold)
{
    float compareTo = alphaThreshold <= 0 ? mat.values.alphaThreshold : clamp(alphaThreshold, 0.0f, 1.0f);
#ifndef _MS_DISABLE_ALPHA_TEST
    if (sampleTexture(mat.textures.alphaMap, mat.samplerState, attr).x < compareTo)
        return false;
#endif
    return true;
}

void _fn applyHashedAlphaTest(MaterialData mat, ShadingAttribs attr, float alphaThreshold)
{
#ifndef _MS_DISABLE_ALPHA_TEST
    if (alphaTestEnabled(mat))
    {
        if (!hashedAlphaTestPassed(mat, attr, alphaThreshold))
            discard;
    }
#endif
}

inline float _fn sineHash(float2 p)
{
    return frac(1e4 * sin(17.0 * p.x + p.y * 0.1) * (0.1 + abs(sin(p.y * 13.0 + p.x))));
}

inline float _fn sineHash3D(float3 p)
{
    return sineHash(float2(sineHash(p.xy), p.z));
}

float _fn calculateHashedAlpha(float3 hashInputCoord, float hashScale, bool useAnisotropy)
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
    alpha test
*******************************************************************/    
void _fn applyAlphaTest(MaterialData material, ShadingAttribs shAttr, float3 posW)
{
    float hashedAlphaScale = 1.0f;
#ifdef _HASHED_ALPHA_SCALE
    hashedAlphaScale = _HASHED_ALPHA_SCALE;
#endif

    /* Evaluate alpha test material modifier */
#ifdef _DEFAULT_ALPHA_TEST
    basicAlphaTest(material, shAttr);
#elif defined(_HASHED_ALPHA_TEST_ANISOTROPIC)
    float alphaThreshold = calculateHashedAlpha(posW, hashedAlphaScale, true);
    applyHashedAlphaTest(material, shAttr, alphaThreshold);
#else
    // Default to isotropic hashed alpha test
    float alphaThreshold = calculateHashedAlpha(posW, hashedAlphaScale, false);
    applyHashedAlphaTest(material, shAttr, alphaThreshold);
#endif
}

/*******************************************************************
    Spherical map sampling
*******************************************************************/
float2 dirToSphericalCrd(float3 direction)
{
    static const float PI = 3.14159265f;
    float3 p = normalize(direction);
    float2 uv;
    uv.x = (1 + atan2(-p.z, p.x) / PI) * 0.5;
    uv.y = 1 - (-acos(p.y) / PI);
    return uv;
}

/*******************************************************************
					Color conversion
*******************************************************************/

float4 _fn applyAmbientOcclusion(float4 color, Texture2D aoTex, SamplerState s, float2 texC)
{
    float aoFactor = aoTex.SampleLevel(s, texC, 0).r;
    return float4(color.rgb * aoFactor, color.a);
}

#endif	// _FALCOR_SHADING_HELPERS_H_