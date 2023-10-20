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
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.ClothMaterialInstance;
import Scene.Material.BasicMaterialData;
import Scene.Material.ShadingUtils;
import Scene.Scene;

/** Implementation of a simple cloth material.
 */
struct ClothMaterial : MaterialBase, IMaterial
{
    BasicMaterialData data;

    // Specify that the associated `MaterialInstance` type is `ClothMaterialInstance`.
    typedef ClothMaterialInstance MaterialInstance;

    ClothMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());

        ClothBRDF brdf = {};

        // Sample base color.
        const float3 baseColor = ms.sampleTexture(data.texBaseColor, s, sd.uv, data.baseColor, lod).rgb;

        // Sample the specular texture. This encodes additional BSDF parameters.
        const float4 spec = ms.sampleTexture(data.texSpecular, s, sd.uv, data.specular, lod);

        // Calculate the specular reflectance for dielectrics from the IoR, as in the Disney BSDF [Burley 2015].
        // UE4 uses 0.08 multiplied by a default specular value of 0.5, hence F0=0.04 as default. The default IoR=1.5 gives the same result.
        float f = (header.getIoR() - 1.f) / (header.getIoR() + 1.f);
        float F0 = f * f;

        brdf.diffuseColor = baseColor;
        brdf.subsurfaceColor = float3(0.5f);    // TODO: Add material parameter for this.
        brdf.roughness = spec.g;                // TODO: Linear or squared roughness?
        brdf.f0 = float3(F0);                   // TODO: What's appropriate here?

#if defined(SCENE_DIFFUSE_ALBEDO_MULTIPLIER)
        // DEMO21
        if (SCENE_DIFFUSE_ALBEDO_MULTIPLIER != 1.f)
        {
            brdf.diffuseColor = saturate(brdf.diffuseColor * SCENE_DIFFUSE_ALBEDO_MULTIPLIER);
        }
#endif

        // Compute final shading frame.
        ShadingFrame sf = sd.frame;
        if (isNormalMappingEnabled(hints))
        {
            float3 encodedNormal = ms.sampleTexture(data.texNormalMap, s, sd.uv, float4(0.f), lod).rgb;
            sf = computeShadingFrameFromNormalMap(sd, data.getNormalMapType(), encodedNormal);
        }
        flipShadingNormal(sd, sf);
        if (isAdjustShadingNormalEnabled(hints))
        {
            adjustShadingNormal(sd, sf);
        }

        return ClothMaterialInstance(sf, brdf);
    }

    [Differentiable]
    ClothMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
