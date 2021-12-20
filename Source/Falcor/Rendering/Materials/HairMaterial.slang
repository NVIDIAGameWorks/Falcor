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
__exported import Rendering.Materials.IMaterial;
__exported import Rendering.Materials.HairBSDF;
import Scene.Material.BasicMaterialData;

/** Implementation of Falcor's standard hair/fur material.
*/
struct HairMaterial : IMaterial
{
    MaterialHeader header;
    BasicMaterialData data;

    // Specify that the associated `BSDF` type is `HairBSDF`.
    typedef HairBSDF BSDF;

    HairBSDF setupBSDF(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod)
    {
        SamplerState s = ms.getTextureSampler(header.getDefaultTextureSamplerID());

        HairBSDFData d = {};

        d.IoR = data.IoR;
        d.eta = sd.frontFacing ? (sd.IoR / data.IoR) : (data.IoR / sd.IoR);

        // Sample base color.
        const float3 baseColor = ms.sampleTexture(data.texBaseColor, s, sd.uv, data.baseColor, lod).rgb;

        // Sample the specular texture. This encodes additional BSDF parameters.
        const float4 spec = ms.sampleTexture(data.texSpecular, s, sd.uv, data.specular, lod);

        d.baseColor = baseColor;
        d.betaM = spec.x;
        d.betaN = spec.y;
        d.alpha = spec.z;

        return HairBSDF(d);
    }

    // Normal mapping is not supported by this material.
    void modifyTangentSpace(const MaterialSystem ms, const float4 tangentW, const ITextureSampler lod, inout ShadingData sd) {}

    // Alpha testing is not supported by this material.
    float evalOpacity(const MaterialSystem ms, const VertexData v, const ITextureSampler lod) { return 1.f; }
};
