/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "PixelInspectorData.h"

import Shading;
import Experimental.Scene.Material.MaterialHelpers;

cbuffer PerFrameCB
{
    uint2 gResolution;
    uint2 gSelectedPixel;
    uint2 gWorldPositionCoord;
    uint2 gWorldShadingNormalCoord;
    uint2 gWorldBitangentCoord;
    uint2 gWorldFaceNormalCoord;
    uint2 gTextureCoordinateCoord;
    uint2 gMaterialDiffuseOpacityCoord;
    uint2 gMaterialSpecularRoughnessCoord;
    uint2 gMaterialEmissiveCoord;
    uint2 gMaterialExtraParamsCoord;
    uint2 gLinearColorCoord;
    uint2 gOutputColorCoord;
    uint2 gVisBufferCoord;

    CameraData gCamera;
}

Texture2D<float4> gWorldPosition;
Texture2D<float4> gWorldShadingNormal;
Texture2D<float4> gWorldBitangent;
Texture2D<float4> gWorldFaceNormal;
Texture2D<float4> gTextureCoordinate;
Texture2D<float4> gMaterialDiffuseOpacity;
Texture2D<float4> gMaterialSpecularRoughness;
Texture2D<float4> gMaterialEmissive;
Texture2D<float4> gMaterialExtraParams;
Texture2D<float4> gLinearColor;
Texture2D<float4> gOutputColor;
Texture2D<uint4>  gVisBuffer;

RWStructuredBuffer<PixelData> gPixelDataBuffer;

/** Load material parameters.
    \return MaterialParams struct filled in with values that were available.
*/
MaterialParams loadMaterialParams()
{
    MaterialParams matParams;
    matParams = matParams.init();

    matParams.diffuseOpacity = gMaterialDiffuseOpacity[gMaterialDiffuseOpacityCoord];
    matParams.specularRoughness = gMaterialSpecularRoughness[gMaterialSpecularRoughnessCoord];
    matParams.emissive = gMaterialEmissive[gMaterialEmissiveCoord];
    matParams.extraParams = gMaterialExtraParams[gMaterialExtraParamsCoord];

    return matParams;
}

/** Calculates the primary ray direction in world space.
    The returned ray points from the camera origin towards the jittered pixel position.
    The function assumes a point camera, no depth-of-field supported in this version.
    \param[in] pixel Pixel coordinates with origin in the top-left corner.
    \param[in] frameDim Frame dimensions in pixels.
    \return World-space dir from camera origin to jittered pixel position (normalized).
*/
float3 getCameraRayDir(uint2 pixel, uint2 frameDim, const CameraData camera)
{
    float2 p = (pixel + float2(0.5f, 0.5f)) / frameDim; // Pixel center on image plane in [0,1] where (0,0) is top-left
    p += float2(-camera.jitterX, camera.jitterY);       // Camera jitter offsets the sample by +-0.5 pixels.
    float2 ndc = float2(2, -2) * p + float2(-1, 1);
    float3 rayDir = ndc.x * camera.cameraU + ndc.y * camera.cameraV + camera.cameraW;
    return normalize(rayDir);
}

[numthreads(1, 1, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    const uint2 pixelPos = gSelectedPixel;

    PixelData data;
    data.init();

    // Read material data
    const float4 worldPos = gWorldPosition[gWorldPositionCoord];
    const float3 normal = gWorldShadingNormal[gWorldShadingNormalCoord].xyz;
    const float3 bitangent = gWorldBitangent[gWorldBitangentCoord].xyz;
    const float3 faceNormal = gWorldFaceNormal[gWorldFaceNormalCoord].xyz;

    // Compute the view vector. This must exactly match what the G-buffer pass is doing (jitter etc.).
    // Note that we do not take depth-of-field into account as it would require exactly matching the
    // sample generator between the passes, which is error prone. The host side will issue a warning instead.
    const float3 viewDir = -getCameraRayDir(pixelPos, gResolution, gCamera);

    const GeometryParams geoParams = prepareGeometryParams(worldPos.xyz, viewDir, normal, bitangent, faceNormal, gTextureCoordinate[gTextureCoordinateCoord].xy);
    const MaterialParams matParams = loadMaterialParams();
    const ShadingData sd = prepareShadingData(geoParams, matParams);

    data.posW = sd.posW;
    data.texCoordU = sd.uv.x;

    data.normal = sd.N;
    data.texCoordV = sd.uv.y;

    data.tangent = sd.T;
    data.NdotV = sd.NdotV;

    data.bitangent = sd.B;
    data.faceNormal = sd.faceN;
    data.view = sd.V;

    data.diffuse = sd.diffuse;
    data.opacity = sd.opacity;

    data.specular = sd.specular;
    data.linearRoughness = sd.linearRoughness;

    data.emissive = sd.emissive;
    data.ggxAlpha = sd.ggxAlpha;

    data.IoR = sd.IoR;
    data.doubleSided = sd.doubleSided ? 1u : 0u;

    // Read output data
    data.linearColor = gLinearColor[gLinearColorCoord];
    data.outputColor = gOutputColor[gOutputColorCoord];

    // Compute luminance. Note we're using the Rec.709 color space now.
    data.luminance = luminance(data.linearColor.rgb);

    // Visibility buffer
    uint4 v = gVisBuffer[gVisBufferCoord];
    data.meshInstanceID = v.x;
    data.triangleIndex = v.y;
    data.barycentrics = asfloat(v.zw);

    // Store result
    gPixelDataBuffer[0] = data;
}
