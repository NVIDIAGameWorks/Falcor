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

/** G-buffer resources and helpers functions for loading the data.

    loadShadingData() loads everything and prepares the ShadingData struct.

    It is assumed the host sets up a define for all optional input buffers:
    is_valid_<name> is 1 if buffer with this name is bound, 0 otherwise.
*/

__exported import Shading;
import Utils.Math.MathHelpers;
import Experimental.Scene.Material.MaterialHelpers;

// G-buffer inputs
shared Texture2D<float4> gWorldPosition;
shared Texture2D<float4> gWorldShadingNormal;
shared Texture2D<float4> gWorldShadingBitangent;    // Optional
shared Texture2D<float4> gWorldView;                // Optional
shared Texture2D<float4> gWorldFaceNormal;
shared Texture2D<float4> gMaterialDiffuseOpacity;
shared Texture2D<float4> gMaterialSpecularRoughness;
shared Texture2D<float4> gMaterialEmissive;
shared Texture2D<float4> gMaterialExtraParams;

#define isValid(name) (is_valid_##name != 0)


/** Returns the primary ray's direction.
*/
float3 getPrimaryRayDir(uint2 pixel, uint2 frameDim, const CameraData camera)
{
    if (isValid(gWorldView))
    {
        // If we have the view vector bound as a buffer, just fetch it. No need to compute anything.
        return -gWorldView[pixel].xyz;
    }
    else
    {
        // Compute the view vector. This must exactly match what the G-buffer pass is doing (jitter etc.).
        // Note that we do not take depth-of-field into account as it would require exactly matching the
        // sample generator between the passes, which is error prone. The host side will issue a warning instead.

        // Compute sample position in screen space in [0,1] with origin at the top-left corner.
        // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
        float2 p = (pixel + float2(0.5f, 0.5f)) / frameDim + float2(-camera.jitterX, camera.jitterY);

        // Compute the normalized ray direction assuming a pinhole camera.
        float2 ndc = float2(2, -2) * p + float2(-1, 1);
        float3 rayDir = ndc.x * camera.cameraU + ndc.y * camera.cameraV + camera.cameraW;
        return normalize(rayDir);
    }
}

/** Helper to load the material attributes.
*/
MaterialParams loadMaterialParams(uint2 pixelPos)
{
    MaterialParams matParams;
    matParams = matParams.init();

    matParams.diffuseOpacity = gMaterialDiffuseOpacity[pixelPos];
    matParams.specularRoughness = gMaterialSpecularRoughness[pixelPos];
    matParams.emissive = gMaterialEmissive[pixelPos];
    matParams.extraParams = gMaterialExtraParams[pixelPos];

    return matParams;
}

/** Helper for setting up the ShadingData struct based on loaded data.
    \param[in] pixel Current pixel coordinates.
    \param[in] frameDim Frame dimensions in pixel.
    \param[out] sd ShadingData struct.
    \return True if the pixel has valid data (not a background pixel). Note sd.V is always valid.
*/
bool loadShadingData(uint2 pixel, uint2 frameDim, const CameraData camera, out ShadingData sd)
{
    sd = {};

    float3 rayDir = getPrimaryRayDir(pixel, frameDim, camera);
    float4 worldPos = gWorldPosition[pixel];
    bool valid = false;

    if (worldPos.w != 0.f) // Using w to indicate valid geometry for now.
    {
        // Load geometry and material parameters from G-buffer.
        float3 normal = gWorldShadingNormal[pixel].xyz;
        float3 bitangent = isValid(gWorldShadingBitangent) ? gWorldShadingBitangent[pixel].xyz : perp_stark(normal);
        float3 faceNormal = gWorldFaceNormal[pixel].xyz;
        GeometryParams geoParams = prepareGeometryParams(worldPos.xyz, -rayDir, normal, bitangent, faceNormal);
        MaterialParams matParams = loadMaterialParams(pixel);

        sd = prepareShadingData(geoParams, matParams);
        valid = true;
    }

    sd.V = -rayDir;
    return valid;
}
