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
import Scene.Shading;
import Scene.Camera.Camera;
import Utils.Color.ColorHelpers;
import PixelInspectorData;

cbuffer PerFrameCB
{
    uint2 gResolution;
    uint2 gSelectedPixel;
    uint2 gWorldPositionCoord;
    uint2 gWorldShadingNormalCoord;
    uint2 gWorldTangentCoord;
    uint2 gWorldFaceNormalCoord;
    uint2 gTextureCoordCoord;
    uint2 gTextureGradsCoord;
    uint2 gMaterialDataCoord;
    uint2 gLinearColorCoord;
    uint2 gOutputColorCoord;
    uint2 gVBufferCoord;
}

// G-buffer inputs
Texture2D<float4> gWorldPosition;
Texture2D<float4> gWorldShadingNormal;
Texture2D<float4> gWorldTangent;
Texture2D<float4> gWorldFaceNormal;
Texture2D<float4> gTextureCoord;
Texture2D<float4> gTextureGrads;
Texture2D<uint4>  gMaterialData;

Texture2D<float4> gLinearColor;
Texture2D<float4> gOutputColor;
Texture2D<PackedHitInfo> gVBuffer;

RWStructuredBuffer<PixelData> gPixelDataBuffer;

#define is_valid(name) (is_valid_##name != 0)

/** Helper to create a texture sampler instance.
    The method for computing texture level-of-detail depends on the configuration.
    \param[in] pixel Current pixel coordinates.
    \return Texture sampler instance.
*/
ITextureSampler createTextureSampler(const uint2 pixel)
{
    if (is_valid(gTextureGrads))
    {
        float4 texGrads = gTextureGrads[pixel];
        return ExplicitGradientTextureSampler(texGrads.xy, texGrads.zw);
    }
    else
    {
        return ExplicitLodTextureSampler(0.f);
    }
}

/** Helper to load G-buffer data and prepare shading data.
*/
bool loadShadingData(const float3 viewDir, const ITextureSampler lod, inout ShadingData sd)
{
    float4 worldPos = gWorldPosition[gWorldPositionCoord];
    if (worldPos.w != 0.f) // Using w to indicate valid geometry for now.
    {
        // Load G-buffer data.
        float3 normal = gWorldShadingNormal[gWorldShadingNormalCoord].xyz;
        float4 tangent = gWorldTangent[gWorldTangentCoord];
        float3 faceNormal = gWorldFaceNormal[gWorldFaceNormalCoord].xyz;
        float2 texC = gTextureCoord[gTextureCoordCoord].xy;
        uint materialID = gMaterialData[gMaterialDataCoord].x;

        VertexData v = {};
        v.posW = worldPos.xyz;
        v.normalW = normalize(normal);
        v.tangentW = float4(normalize(tangent.xyz), tangent.w);
        v.faceNormalW = normalize(faceNormal);
        v.texC = texC;
        v.coneTexLODValue = 0.f;

        // Prepare shading data.
        // Set 'useNormalMap' parameter to false as normal mapping has already been applied when generating the G-buffer.
        sd = gScene.materials.prepareShadingData(v, materialID, viewDir, lod, false);

        return true;
    }

    return false;
}

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    const uint2 pixel = gSelectedPixel;

    // Compute the view vector. This must exactly match what the G-buffer pass is doing (jitter etc.).
    // Note that we do not take depth-of-field into account as it would require exactly matching the
    // sample generator between the passes, which is error prone. The host side will issue a warning instead.
    const float3 viewDir = -gScene.camera.computeRayPinhole(pixel, gResolution).dir;

    PixelData data = PixelData();
    ShadingData sd = {};
    let lod = createTextureSampler(gTextureGradsCoord);

    if (loadShadingData(viewDir, lod, sd))
    {
        let bsdf = gScene.materials.getBSDF(sd, lod);
        let bsdfProperties = bsdf.getProperties(sd);

        // Store geometry data.
        data.posW = sd.posW;
        data.normal = sd.N;
        data.tangent = sd.T;
        data.bitangent = sd.B;
        data.faceNormal = sd.faceN;
        data.view = sd.V;
        data.texCoord = sd.uv;
        data.frontFacing = sd.frontFacing ? 1 : 0;

        // Store material data.
        data.materialID = sd.materialID;
        data.doubleSided = sd.mtl.isDoubleSided() ? 1 : 0;
        data.opacity = sd.opacity;
        data.IoR = sd.IoR;

        // Store BSDF properties.
        data.emission = bsdfProperties.emission;
        data.roughness = bsdfProperties.roughness;
        data.diffuseReflectionAlbedo = bsdfProperties.diffuseReflectionAlbedo;
        data.diffuseTransmissionAlbedo = bsdfProperties.diffuseTransmissionAlbedo;
        data.specularReflectionAlbedo = bsdfProperties.specularReflectionAlbedo;
        data.specularTransmissionAlbedo = bsdfProperties.specularTransmissionAlbedo;
        data.specularReflectance = bsdfProperties.specularReflectance;
        data.isTransmissive = bsdfProperties.isTransmissive ? 1 : 0;
    }

    // Store output data.
    // We compute output luminance using the Rec.709 color space.
    data.linearColor = gLinearColor[gLinearColorCoord];
    data.outputColor = gOutputColor[gOutputColorCoord];
    data.luminance = luminance(data.linearColor.rgb);

    // Decode visibility buffer.
    if (is_valid(gVBuffer))
    {
        const HitInfo hit = HitInfo(gVBuffer[gVBufferCoord]);
        if (hit.isValid())
        {
            data.hitType = (uint)hit.getType();
            if (hit.getType() == HitType::Triangle)
            {
                const TriangleHit triangleHit = hit.getTriangleHit();
                data.instanceID = triangleHit.instanceID.index;
                data.primitiveIndex = triangleHit.primitiveIndex;
                data.barycentrics = triangleHit.barycentrics;
            }
        }
    }

    // Store result
    gPixelDataBuffer[0] = data;
}
