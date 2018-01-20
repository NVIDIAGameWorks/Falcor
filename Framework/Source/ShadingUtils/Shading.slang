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
#ifndef _FALCOR_SHADING_SLANG_
#define _FALCOR_SHADING_SLANG_
#import "HostDeviceData.h"
__import DefaultVS;
__exported __import ShaderCommon;
__import Lights;
__import BRDF;
__import Helpers;

/** Load data from a texture
*/
float4 sampleTexture(Texture2D t, SamplerState s, float2 uv, float4 factor, uint mode)
{
    if(mode == ChannelTypeUnused) return 0;
    if(mode == ChannelTypeConst) return factor;
    // else mode == ChannelTypeTexture
    return t.Sample(s, uv);
}

/** Shading result struct
*/
struct ShadingResult
{
    float3 diffuseBrdf;             // The result of the diffuse BRDF
    float3 specularBrdf;            // The result of the specular BRDF
    float3 diffuse;                 // The diffuse component of the result
    float3 specular;                // The specular component of the result
    float4 color;                   // The final color. Alpha holds the opacity valua
};

/** Convert RGB to normal
*/
float3 RgbToNormal(float3 rgb)
{
    float3 n = rgb * 2 - 1;
    return normalize(n);
}

/** Convert RG to normal
*/
float3 RgToNormal(float2 rg)
{
    float3 n;
    n.xy = rg * 2 - 1;

    // Saturate because error from BC5 can break the sqrt
    n.z = saturate(dot(rg, rg)); // z = r*r + g*g
    n.z = sqrt(1 - n.z);
    return normalize(n);
}

/** Apply normal map
*/
void applyNormalMap(MaterialData m, inout HitPoint hitPt)
{
    uint mapType = EXTRACT_NORMAL_MAP_TYPE(m.flags);
    if(mapType == NormalMapUnused) return;

    float3 mapN = m.resources.normalMap.Sample(m.resources.samplerState, hitPt.uv).rgb;
    switch(mapType)
    {
    case NormalMapRGB:
        mapN = RgbToNormal(mapN);
        break;
    case NormalMapRG:
        mapN = RgToNormal(mapN.rg);
        break;
    default:
        return;
    }

    // Apply the transformation. Everything is normalized already
    hitPt.N = hitPt.T * mapN.x + hitPt.B * mapN.y + hitPt.N * mapN.z;
    hitPt.B = normalize(hitPt.B - hitPt.N * dot(hitPt.B, hitPt.N));
    hitPt.T = normalize(cross(hitPt.B, hitPt.N));
}

/** Apply alpha-test
*/
void applyAlphaTest(MaterialData m, float alpha)
{
    if(EXTRACT_ALPHA_MODE(m.flags) == AlphaModeMask)
    {
        if(alpha < m.alphaThreshold) discard;
    }
}

HitPoint initHitPoint()
{
    HitPoint hitPt;
    hitPt.posW = 0;
    hitPt.V = 0;
    hitPt.N = 0;
    hitPt.T = 0;
    hitPt.B = 0;
    hitPt.uv = 0;
    hitPt.NdotV = 0;
    hitPt.diffuse = 0;
    hitPt.opacity = 1;
    hitPt.specular = 0;
    hitPt.linearRoughness = 0;
    hitPt.roughness = 0;
    hitPt.emissive = 0;
    hitPt.occlusion = 0;
    hitPt.lightMap = 0;
    hitPt.height = 0;
    hitPt.IoR = 0;
    hitPt.doubleSidedMaterial = false;

    return hitPt;
}

/** Prepare the hit-point data
*/
HitPoint prepareHitPoint(VertexOut v, MaterialData m, float3 camPosW)
{
    HitPoint hitPt = initHitPoint();

    // Sample the diffuse texture and apply the alpha test
    float4 diffuse = sampleTexture(m.resources.diffuse, m.resources.samplerState, v.texC, m.diffuse, EXTRACT_DIFFUSE_TYPE(m.flags));
    hitPt.diffuse = diffuse.rgb;
    hitPt.opacity = m.diffuse.a;
    alphaTest(m.flags, diffuse.a, m.alphaThreshold, v.posW);

    hitPt.posW = v.posW;
    hitPt.uv = v.texC;
    hitPt.V = normalize(camPosW - v.posW);
    hitPt.N = normalize(v.normalW);

    hitPt.B = normalize(v.bitangentW - hitPt.N * (dot(v.bitangentW, hitPt.N)));
    hitPt.T = normalize(cross(hitPt.B, hitPt.N));

    // Sample the rest of the texture
    float4 spec = sampleTexture(m.resources.specular, m.resources.samplerState, v.texC, m.specular, EXTRACT_SPECULAR_TYPE(m.flags));
    hitPt.specular = spec.rgb;
    hitPt.linearRoughness = max(0.08, 1 - spec.a); // Clamp the roughness so that the BRDF won't explode
    hitPt.roughness = hitPt.linearRoughness * hitPt.linearRoughness;
    hitPt.emissive = sampleTexture(m.resources.emissive, m.resources.samplerState, v.texC, float4(m.emissive, 1), EXTRACT_EMISSIVE_TYPE(m.flags)).rgb;
    hitPt.IoR = m.IoR;
    hitPt.doubleSidedMaterial = EXTRACT_DOUBLE_SIDED(m.flags);

#define channel_type(extract) (extract(m.flags) ? ChannelTypeTexture : ChannelTypeUnused)
    hitPt.occlusion = sampleTexture(m.resources.occlusionMap, m.resources.samplerState, v.texC, 1, channel_type(EXTRACT_OCCLUSION_MAP));
    hitPt.lightMap = sampleTexture(m.resources.lightMap, m.resources.samplerState, v.lightmapC, 1, channel_type(EXTRACT_LIGHT_MAP)).rgb;
    hitPt.height = sampleTexture(m.resources.heightMap, m.resources.samplerState, v.texC, 1, channel_type(EXTRACT_HEIGHT_MAP)).xy;
    hitPt.height = hitPt.height * m.heightScaleOffset.x + m.heightScaleOffset.y;
#undef channel_type
    
    applyNormalMap(m, hitPt);
    hitPt.NdotV = dot(hitPt.N, hitPt.V);

    // Flip the normal if it's backfacing
    if(hitPt.NdotV <= 0 && hitPt.doubleSidedMaterial)
    {
        hitPt.N = -hitPt.N;
        hitPt.NdotV = -hitPt.NdotV;
    }

    return hitPt;
}

ShadingResult initShadingResult()
{
    ShadingResult sr;
    sr.diffuse = 0;
    sr.color.rgb = 0;
    sr.color.a = 1;
    sr.specular = 0;
    sr.diffuseBrdf = 0;
    sr.specularBrdf = 0;
    return sr;
}

ShadingResult evalMaterial(HitPoint hitPt, LightData light, float shadowFactor)
{
    ShadingResult sr = initShadingResult();
    LightSample ls = evalLight(light, hitPt);

    // If the light doesn't hit the surface or we are viewing the surface from the back, return
    if(ls.NdotL <= 0 || hitPt.NdotV <= 0) return sr;

    // Calculate the diffuse term
    sr.diffuseBrdf = evalDiffuseBrdf(hitPt, ls);
    sr.diffuse = ls.diffuse * sr.diffuseBrdf * ls.NdotL;
    sr.color.rgb = sr.diffuse;
    sr.color.a = hitPt.opacity;

    // Calculate the specular term
    sr.specularBrdf = evalSpecularBrdf(hitPt, ls);
    sr.specular = ls.specular * sr.specularBrdf * ls.NdotL;
    sr.color.rgb += sr.specular;

    // Apply the shadow factor
    sr.color.rgb *= shadowFactor;

    return sr;
};

ShadingResult evalMaterial(HitPoint hitPt, LightProbeData probe)
{
    ShadingResult sr = initShadingResult();
    LightSample ls = evalLightProbe(probe, hitPt);

    sr.diffuse = hitPt.diffuse.rgb * ls.diffuse;
    sr.color.rgb = sr.diffuse;
    sr.specular = saturate(evalGGX(hitPt.roughness, ls.NdotH)) * ls.specular * hitPt.specular;
    sr.color.rgb += sr.specular;
    return sr;
}

#endif	// _FALCOR_SHADING_SLANG_
