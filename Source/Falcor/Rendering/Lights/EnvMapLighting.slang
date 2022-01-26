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
import Scene.Scene;
import Scene.ShadingData;
import Rendering.Materials.IBSDF;
import Utils.Math.MathHelpers;

/** Implementation of image-based lighting using an environment map probe.

    The code assumes the material can be approximated by a diffuse lobe
    plus a GGX microfacet specular lobe. Only reflection is supported.
    Note that the actual material may be different so the result can be
    arbitrarily wrong.
*/
struct EnvMapLighting
{
    Texture2D dfgTexture;       ///< Texture containing shared pre-integrated (DFG) term. The u-axis is NdotV=cos(theta) and the v-axis is GGX alpha, both in [0,1].
    SamplerState dfgSampler;    ///< Sampler for the DFG texture: point filtering with clamp wrap mode.

    Texture2D diffuseTexture;   ///< Texture containing pre-integrated diffuse (LD) term
    Texture2D specularTexture;  ///< Texture containing pre-integrated specular (LD) term
    SamplerState sampler;

    float roughnessToLod(float roughness, float mipCount)
    {
        return roughness * (mipCount - 1.f);
    }

    float3 getDiffuseDominantDir(float3 N, float3 V, float ggxAlpha)
    {
        float a = 1.02341 * ggxAlpha - 1.51174;
        float b = -0.511705 * ggxAlpha + 0.755868;
        float factor = saturate((saturate(dot(N, V)) * a + b) * ggxAlpha);
        return normalize(lerp(N, V, factor));
    }

    float3 getSpecularDominantDir(float3 N, float3 R, float ggxAlpha)
    {
        float smoothness = 1.f - ggxAlpha;
        float factor = smoothness * (sqrt(smoothness) + ggxAlpha);
        return normalize(lerp(N, R, factor));
    }

    float3 evalDiffuse(const ShadingData sd, const BSDFProperties bsdfProperties)
    {
        float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
        float3 dominantDir = getDiffuseDominantDir(sd.N, sd.V, ggxAlpha);
        float2 uv = world_to_latlong_map(gScene.envMap.toLocal(dominantDir));

        float width, height, mipCount;
        diffuseTexture.GetDimensions(0, width, height, mipCount);

        float3 diffuseLighting = diffuseTexture.SampleLevel(sampler, uv, 0).rgb;
        float preintegratedDisneyBRDF = dfgTexture.SampleLevel(dfgSampler, float2(dot(sd.N, sd.V), ggxAlpha), 0).z;

        return diffuseLighting * preintegratedDisneyBRDF * bsdfProperties.diffuseReflectionAlbedo * gScene.envMap.getIntensity();
    }

    float3 evalSpecular(const ShadingData sd, const BSDFProperties bsdfProperties, const float3 L)
    {
        float dfgWidth, dfgHeight;
        dfgTexture.GetDimensions(dfgWidth, dfgHeight);

        float width, height, mipCount;
        specularTexture.GetDimensions(0, width, height, mipCount);

        float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
        float3 dominantDir = getSpecularDominantDir(sd.N, L, ggxAlpha);
        float2 uv = world_to_latlong_map(gScene.envMap.toLocal(dominantDir));

        float mipLevel = roughnessToLod(bsdfProperties.roughness, mipCount);

        float3 ld = specularTexture.SampleLevel(sampler, uv, mipLevel).rgb;
        float2 dfg = dfgTexture.SampleLevel(dfgSampler, float2(dot(sd.N, sd.V), ggxAlpha), 0).xy;

        // ld * (f0 * Gv * (1 - Fc)) + (f90 * Gv * Fc)
        return ld * (bsdfProperties.specularReflectance * dfg.x + dfg.y) * gScene.envMap.getIntensity();
    }
}
