/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
__exported import Scene.Shading;
import Scene.Material.ShadingUtils;
import Utils.Math.MathHelpers;

#if USE_VBUFFER
// V-buffer inputs
Texture2D<PackedHitInfo> gVBuffer;
#else
// G-buffer inputs
Texture2D<float4> gWorldPosition;
Texture2D<float4> gWorldShadingNormal;
Texture2D<float4> gWorldShadingTangent;
Texture2D<float4> gWorldFaceNormal;
Texture2D<float2> gTextureCoord;
Texture2D<float4> gTextureGrads; // Optional
Texture2D<uint4>  gMaterialData;
#endif

#define is_valid(name) (is_valid_##name != 0)

/** Helper to create a texture sampler instance.
    The method for computing texture level-of-detail depends on the configuration.
    \param[in] pixel Current pixel coordinates.
    \return Texture sampler instance.
*/
ITextureSampler createTextureSampler(const uint2 pixel)
{
#if USE_VBUFFER
    return ExplicitLodTextureSampler(0.f);
#else
    if (is_valid(gTextureGrads))
    {
        float4 texGrads = gTextureGrads[pixel];
        return ExplicitGradientTextureSampler(texGrads.xy, texGrads.zw);
    }
    else
    {
        return ExplicitLodTextureSampler(0.f);
    }
#endif
}

/** Helper for setting up the ShadingData struct based on loaded data.
    \param[in] pixel Current pixel coordinates.
    \param[in] frameDim Frame dimensions in pixel.
    \param[in] camera Current camera.
    \param[in] lod Method for computing texture level-of-detail.
    \param[out] sd ShadingData struct.
    \return True if the pixel has valid data (not a background pixel). Note sd.V is always valid.
*/
bool loadShadingData(const uint2 pixel, const uint2 frameDim, const Camera camera, const ITextureSampler lod, out ShadingData sd)
{
    sd = {};

    float3 rayDir = camera.computeRayPinhole(pixel, frameDim).dir;
    bool valid = false;

#if USE_VBUFFER
    const HitInfo hit = HitInfo(gVBuffer[pixel]);
    if (hit.isValid() && hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        // Evaluate Falcor's material parameters at the hit point.
        // TODO: Implement texLOD to enable texture filtering in prepareShadingData().
        const VertexData v = gScene.getVertexData(triangleHit);
        const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
        sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);

        // Adjust shading normals if GBuffer pass has flag enabled.
#if GBUFFER_ADJUST_SHADING_NORMALS
        adjustShadingNormal(sd, v);
#endif
        valid = true;
    }
#else
    float4 worldPos = gWorldPosition[pixel];
    if (worldPos.w != 0.f) // Using w to indicate valid geometry for now.
    {
        // Load G-buffer data.
        float3 normal = gWorldShadingNormal[pixel].xyz;
        float4 tangent = gWorldShadingTangent[pixel];
        float3 faceNormal = gWorldFaceNormal[pixel].xyz;
        float2 texC = gTextureCoord[pixel];
        uint materialID = gMaterialData[pixel].x;

        VertexData v = {};
        v.posW = worldPos.xyz;
        v.normalW = normalize(normal);
        v.tangentW = float4(normalize(tangent.xyz), tangent.w);
        v.faceNormalW = normalize(faceNormal);
        v.texC = texC;
        v.coneTexLODValue = 0.f;

        // Prepare shading data.
        // Set 'useNormalMap' parameter to false as normal mapping has already been applied when generating the G-buffer.
        sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod, false);

        valid = true;
    }
#endif

    sd.V = -rayDir;
    return valid;
}
