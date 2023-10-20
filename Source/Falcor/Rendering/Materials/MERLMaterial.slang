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
__exported import Rendering.Materials.MERLMaterialInstance;
import Scene.Material.MERLMaterialData;

/** Implementation of measured MERL material.
 */
struct MERLMaterial : MaterialBase, IMaterial
{
    MERLMaterialData data;

    // Specify that the associated `MaterialInstance` type is `MERLMaterialInstance`.
    typedef MERLMaterialInstance MaterialInstance;

    MERLMaterialInstance setupMaterialInstance(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        // Compute final shading frame.
        ShadingFrame sf = sd.frame;
        flipShadingNormal(sd, sf);
        if (isAdjustShadingNormalEnabled(hints))
        {
            adjustShadingNormal(sd, sf);
        }

        // Evaluate the albedo for the current incident (view) direction.
        float3 albedo = {};
        float cosTheta = dot(sf.N, sd.V);

        if (cosTheta >= kMinCosTheta)
        {
            // The albedo is precomputed in a lookup table, which we sample using piecewise linear interpolation.
            // The precompued albedo values are sampled at cosTheta = (1,2,...N)/N, where N is the LUT size.
            // We offset the texture coordinate by -0.5/N to get correct interpolation of the two nearest samples.
            float u = cosTheta - 0.5f / MERLMaterialData::kAlbedoLUTSize;

            SamplerState s = ms.getTextureSampler(data.samplerID);
            let explicitLod = ExplicitLodTextureSampler(0.f);
            albedo = ms.sampleTexture(data.texAlbedoLUT, s, float2(u, 0.5f), float4(0.5f), explicitLod).rgb;
        }

        return MERLMaterialInstance(sf, data.bufferID, albedo, data.extraData);
    }

    [Differentiable]
    MERLMaterialInstance setupDiffMaterialInstance(out DiffMaterialData diffData, const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod, const uint hints)
    {
        diffData = DiffMaterialData();
        return no_diff setupMaterialInstance(ms, sd, lod, hints);
    }
};
