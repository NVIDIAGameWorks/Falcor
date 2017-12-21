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
    float3 diffuseIllumination;
    float3 specularIllumination;
    float3 color;
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
    n.z = dot(rg, rg); // z = r*r + g*g
    n.z = sqrt(1 - n.z);
    return n;
}

/** Apply normal map
*/
void applyNormalMap(MaterialData m, inout HitPoint hitPt)
{
    uint mapType = EXTRACT_NORMAL_MAP_TYPE(m.flags);
    if(mapType == NormalMapUnused) return;

    float3 mapN = m.textures.normalMap.Sample(m.samplerState, hitPt.uv).rgb;
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
    
    mapN = normalize(mapN);
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

/** Prepare the hit-point data
*/
HitPoint prepareHitPoint(VertexOut v, MaterialData m, float3 camPosW)
{
    HitPoint hitPt;
    // Sample the diffuse texture and apply the alpha test
    float4 diffuse = sampleTexture(m.textures.diffuse, m.samplerState, v.texC, m.diffuse, EXTRACT_DIFFUSE_TYPE(m.flags));
    hitPt.diffuse = diffuse.rgb;
    hitPt.opacity = diffuse.a;
    applyAlphaTest(m, diffuse.a);

    hitPt.posW = v.posW;
    hitPt.uv = v.texC;
    hitPt.V = normalize(camPosW - v.posW);
    hitPt.N = normalize(v.normalW);
    hitPt.B = normalize(v.bitangentW - hitPt.N * (dot(v.bitangentW, hitPt.N)));
    hitPt.T = normalize(cross(hitPt.B, hitPt.N));

    // Sample the rest of the texture
    float4 spec = sampleTexture(m.textures.specular, m.samplerState, v.texC, m.specular, EXTRACT_SPECULAR_TYPE(m.flags));
    hitPt.specular = spec.rgb;
    hitPt.linearRoughness = max(0.08, 1 - spec.a); // Clamp the roughness so that the BRDF won't explode
    hitPt.roughness = hitPt.linearRoughness * hitPt.linearRoughness;
    hitPt.emissive = sampleTexture(m.textures.emissive, m.samplerState, v.texC, float4(m.emissive, 1), EXTRACT_EMISSIVE_TYPE(m.flags)).rgb;
    hitPt.IoR = m.IoR;

#define channel_type(extract) (extract(m.flags) ? ChannelTypeTexture : ChannelTypeUnused)
    hitPt.occlusion = sampleTexture(m.textures.occlusionMap, m.samplerState, v.texC, 1, channel_type(EXTRACT_OCCLUSION_MAP));
    hitPt.reflection = sampleTexture(m.textures.reflectionMap, m.samplerState, v.texC, 1, channel_type(EXTRACT_REFLECTION_MAP));
    hitPt.lightMap = sampleTexture(m.textures.lightMap, m.samplerState, v.texC, 1, channel_type(EXTRACT_LIGHT_MAP));
    hitPt.height = sampleTexture(m.textures.heightMap, m.samplerState, v.texC, 1, channel_type(EXTRACT_HEIGHT_MAP)).xy;
    hitPt.height = hitPt.height * m.heightScaleOffset.x + m.heightScaleOffset.y;
#undef channel_type

    hitPt.NdotV = 0; // Need to initialize it otherwise the compiler complains about uninitialized variable when calling applyNormalMap()
    applyNormalMap(m, hitPt);
    hitPt.NdotV = abs(dot(hitPt.N, hitPt.V)) + 1e-5f;

    return hitPt;
}

ShadingResult evalMaterial(HitPoint hitPt, MaterialData mat, LightData light, float shadowFactor)
{
    LightSample ls = evalLight(light, hitPt, EXTRACT_DOUBLE_SIDED(mat.flags));

    ShadingResult sr;
    sr.diffuseIllumination = 0;
    sr.color = 0;
    sr.specularIllumination = 0;
    
    sr.diffuseIllumination = evalDiffuse(hitPt, mat, ls);
    sr.color = hitPt.diffuse.rgb * sr.diffuseIllumination * (1 / M_PI);

    sr.specularIllumination = evalSpecular(hitPt, mat, ls);
    sr.color += sr.specularIllumination;
    sr.color *= shadowFactor;

    return sr;
};

#endif	// _FALCOR_SHADING_SLANG_
