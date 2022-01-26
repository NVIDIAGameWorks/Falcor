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

/** Helper functions for generating the G-buffer.
*/

import Scene.SceneTypes;
import Scene.ShadingData;
import Rendering.Materials.IBSDF;
import Utils.Math.MathHelpers;
import Utils.Math.FormatConversion;
__exported import Scene.Material.ShadingUtils;

struct GBufferData
{
    float4 posW;
    float4 normW;
    float4 tangentW;
    float4 faceNormalW;
    float2 texC;
    uint4  mtlData;

    // Legacy channels
    float4 diffuseOpacity;
    float4 specRough;
    float4 emissive;
};

/** Helper function to prepare G-buffer data.
*/
GBufferData prepareGBufferData(const ShadingData sd, const IBSDF bsdf)
{
    GBufferData gbuf;

    // We store the final normal and tangent in the G-buffer.
    // In order to reconstruct the bitangent later, we also need to store its handedness (sign).
    float3 B = cross(sd.N, sd.T);
    float bitangentSign = dot(sd.B, B) >= 0.f ? 1.f : -1.f;

    gbuf.posW           = float4(sd.posW, 1.f);
    gbuf.normW          = float4(sd.N, 0.f);
    gbuf.tangentW       = float4(sd.T, bitangentSign);
    gbuf.faceNormalW    = float4(sd.faceN, 0.f);
    gbuf.texC           = sd.uv;
    gbuf.mtlData        = uint4(sd.materialID, sd.mtl.packedData.x, sd.mtl.packedData.y, 0);

    // Get BSDF properties.
    let bsdfProperties = bsdf.getProperties(sd);

    // Setup legacy material channels.
    gbuf.diffuseOpacity = float4(bsdfProperties.diffuseReflectionAlbedo, sd.opacity);
    gbuf.specRough      = float4(bsdfProperties.specularReflectance, bsdfProperties.roughness);
    gbuf.emissive       = float4(bsdfProperties.emission, 0.f);

    return gbuf;
}
