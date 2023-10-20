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
__exported import Rendering.Materials.MERLMixMaterialInstance;
import Scene.Material.MERLMixMaterialData;
import Scene.Scene;

/** Implementation of measured MERLMix material.
 */
struct MERLMixMaterial : MaterialBase, IMaterial
{
    MERLMixMaterialData data;

    // Specify that the associated `MaterialInstance` type is `MERLMixMaterialInstance`.
    typedef MERLMixMaterialInstance MaterialInstance;

    MERLMixMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        let explicitLod = ExplicitLodTextureSampler(0.f);

        // Compute final shading frame.
        ShadingFrame sf = sd.frame;
        if (isNormalMappingEnabled(hints))
        {
            SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());
            float3 encodedNormal = ms.sampleTexture(data.texNormalMap, s, sd.uv, float4(0.f), lod).rgb;
            sf = computeShadingFrameFromNormalMap(sd, data.getNormalMapType(), encodedNormal);
        }
        flipShadingNormal(sd, sf);
        if (isAdjustShadingNormalEnabled(hints))
        {
            adjustShadingNormal(sd, sf);
        }

        // Fetch the BRDF index to use at the shading point.
        // The index map is assumed to be in 8-bit unorm format and we use the red channel scaled by 255 as index.
        // The index wraps around modulo the actual BRDF count.
        uint brdfIndex = 0;
        {
            SamplerState s = ms.getTextureSampler(data.getIndexSamplerID());
            float indexUnorm = ms.sampleTexture(data.texIndexMap, s, sd.uv, float4(0.f), explicitLod).r;
            brdfIndex = (uint)(indexUnorm * 255.f) % data.brdfCount;
        }
        uint byteOffset = brdfIndex * data.byteStride;

        // Evaluate the albedo for the current incident (view) direction.
        float3 albedo = {};
        float cosTheta = dot(sf.N, sd.V);

        if (cosTheta >= kMinCosTheta)
        {
            // The albedo is precomputed in a lookup table, which we sample using piecewise linear interpolation.
            // The columns store precomputed albedo sampled at cosTheta = (1,2,...N)/N, where N is the LUT size.
            // We offset the texture coordinate by -0.5/N to get correct interpolation of the two nearest samples.
            // The rows store one lookup table per BRDF index. We sample the midpoint of the texels vertically.
            float u = cosTheta - 0.5f / MERLMixMaterialData::kAlbedoLUTSize;
            float v = (brdfIndex + 0.5f) / (float)data.brdfCount;

            SamplerState s = ms.getTextureSampler(data.getLUTSamplerID());
            albedo = ms.sampleTexture(data.texAlbedoLUT, s, float2(u, v), float4(0.5f), explicitLod).rgb;
        }

        // Load extra data for sampling.
        ByteAddressBuffer brdfData = gScene.materials.getBuffer(data.bufferID);
        uint extraDataByteOffset = data.extraDataOffset + brdfIndex * data.extraDataStride;
        DiffuseSpecularData extraData = brdfData.Load<DiffuseSpecularData>(extraDataByteOffset);

        return MERLMixMaterialInstance(sf, data.bufferID, byteOffset, albedo, brdfIndex, extraData);
    }

    [Differentiable]
    MERLMixMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
