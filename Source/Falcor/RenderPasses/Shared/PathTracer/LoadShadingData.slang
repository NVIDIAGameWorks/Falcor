/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

/** Helper functions for loading G/V-buffer data.

    loadShadingData() loads everything and prepares the ShadingData struct.

    It is assumed the host sets up a define for all optional input buffers:
    is_valid_<name> is 1 if buffer with this name is bound, 0 otherwise.
*/

import Scene.Scene;
import Scene.Shading;
import Scene.HitInfo;
import Utils.Math.MathHelpers;

#if USE_VBUFFER

// V-buffer inputs
Texture2D<uint2> gVBuffer;
#else
import Experimental.Scene.Material.MaterialHelpers;

// G-buffer inputs
Texture2D<float4> gWorldPosition;
Texture2D<float4> gWorldShadingNormal;
Texture2D<float4> gWorldShadingTangent;      // Optional
Texture2D<float4> gWorldView;                // Optional
Texture2D<float4> gWorldFaceNormal;
Texture2D<float4> gMaterialDiffuseOpacity;
Texture2D<float4> gMaterialSpecularRoughness;
Texture2D<float4> gMaterialEmissive;
Texture2D<float4> gMaterialExtraParams;
Texture2D<uint2>  gVBuffer;                  // Optional
#endif

#define isValid(name) (is_valid_##name != 0)


/** Returns the primary ray's direction.
*/
float3 getPrimaryRayDir(uint2 pixel, uint2 frameDim, const Camera camera)
{
#if !USE_VBUFFER
    if (isValid(gWorldView))
    {
        // If we have the view vector bound as a buffer, just fetch it. No need to compute anything.
        return -gWorldView[pixel].xyz;
    }
    else
#endif
    {
        // Compute the view vector. This must exactly match what the G-buffer pass is doing (jitter etc.).
        // Note that we do not take depth-of-field into account as it would require exactly matching the
        // sample generator between the passes, which is error prone. The host side will issue a warning instead.
        return camera.computeRayPinhole(pixel, frameDim).dir;
    }
}

/** Helper for setting up the ShadingData struct based on loaded data.
    \param[in] pixel Current pixel coordinates.
    \param[in] frameDim Frame dimensions in pixel.
    \param[in] camera Current camera.
    \param[out] sd ShadingData struct.
    \param[out] hit HitInfo struct returned with geometry fetched from vbuffer if available.
    \return True if the pixel has valid data (not a background pixel). Note sd.V is always valid.
*/
bool loadShadingData(uint2 pixel, uint2 frameDim, const Camera camera, out ShadingData sd, out HitInfo hit)
{
    sd = {};

    float3 rayDir = getPrimaryRayDir(pixel, frameDim, camera);
    bool valid = false;

#if USE_VBUFFER
    if (hit.decode(gVBuffer[pixel]))
    {
        // Evaluate Falcor's material parameters at the hit point.
        // TODO: Implement texLOD to enable texture filtering in prepareShadingData().
        VertexData v = gScene.getVertexData(hit);
        const uint materialID = gScene.getMaterialID(hit.meshInstanceID);
        sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -rayDir, 0.f);

        // Compute tangent space if it is invalid.
        if (!(dot(sd.T, sd.T) > 0.f))   // Note: Comparison written so that NaNs trigger
        {
            sd.T = perp_stark(sd.N);
            sd.B = cross(sd.N, sd.T);
        }

        valid = true;
    }
#else
    float4 worldPos = gWorldPosition[pixel];
    if (worldPos.w != 0.f) // Using w to indicate valid geometry for now.
    {
        // Load geometry and material parameters from G-buffer.
        float3 normal = gWorldShadingNormal[pixel].xyz;
        float4 tangent = isValid(gWorldShadingTangent) ? gWorldShadingTangent[pixel] : float4(perp_stark(normal), 1.f);
        float3 faceNormal = gWorldFaceNormal[pixel].xyz;
        GeometryParams geoParams = prepareGeometryParams(worldPos.xyz, -rayDir, normal, tangent, faceNormal);

        MaterialParams matParams;
        matParams.diffuseOpacity = gMaterialDiffuseOpacity[pixel];
        matParams.specularRoughness = gMaterialSpecularRoughness[pixel];
        matParams.emissive = gMaterialEmissive[pixel];
        matParams.extraParams = gMaterialExtraParams[pixel];

        sd = prepareShadingData(geoParams, matParams);
        if (isValid(gVBuffer))
        {
            if (hit.decode(gVBuffer[pixel])) sd.materialID = gScene.getMaterialID(hit.meshInstanceID);
        }
        else
        {
            hit = { HitInfo::kInvalidIndex };
        }
        valid = true;
    }
#endif

    sd.V = -rayDir;
    return valid;
}

/** Helper for setting up the ShadingData struct based on loaded data.
    \param[in] pixel Current pixel coordinates.
    \param[in] frameDim Frame dimensions in pixel.
    \param[in] camera Current camera.
    \param[out] sd ShadingData struct.
    \return True if the pixel has valid data (not a background pixel). Note sd.V is always valid.
*/
bool loadShadingData(uint2 pixel, uint2 frameDim, const Camera camera, out ShadingData sd)
{
    HitInfo hit;
    return loadShadingData(pixel, frameDim, camera, sd, hit);
}
