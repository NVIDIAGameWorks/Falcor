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
#include "Utils/Math/MathConstants.slang"

import HostDeviceData;
import Scene;
import Experimental.Scene.Lights.LightCollection;

// TODO: Move the passes to separate .slang files to avoid confusion of which resources are bound, or
// use new Slang syntax for passing resources to the entry point when that's available.

// Resources used by the finalizeIntegration pass.
cbuffer FinalCB
{
    LightCollection gLights;                ///< The light collection.
}

ByteAddressBuffer gTexelSum;                ///< Sum over texels (RGB) + number of texels (A) in RGBA32Float format.

// Resources used by the build/update passes.
cbuffer PerMeshCB
{
    uint gLightIdx;                         ///< The mesh light index.
    uint gMeshInstanceID;                   ///< Mesh light's global mesh instance ID. We use this to find its data in the scene.
    uint gTriangleCount;                    ///< Number of triangles in current mesh light.
    uint gTriangleOffset;                   ///< Current offset into global list of emissive triangles.
}

RWByteAddressBuffer gVertexPosOutput;       ///< Vertex positions in world space for all mesh light triangles. Size: triangleCount * 3 * sizeof(float3).
RWByteAddressBuffer gTexCoordsOutput;       ///< Texture coordinates for all mesh light triangles. Size: triangleCount * 3 * sizeof(float2).
StructuredBuffer<MeshLightData> gMeshData;  ///< Per-mesh data for emissive meshes. Size. _NUM_MESH_LIGHTS * sizeof(MeshLightData).

// Resources used by all passes.
RWStructuredBuffer<EmissiveTriangle> gTriangleData;     ///< Per-triangle data for emissive triangles. Size: triangleCount * sizeof(EmissiveTriangle).


/** Kernel building the emissive triangles list for all mesh lights.
    One dispatch per mesh light with one thread per triangle.
*/
[numthreads(256, 1, 1)]
void buildTriangleList(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gTriangleCount) return;

    const uint triangleIndex = DTid.x; // Triangle index in current mesh.
    const uint triIdx = gTriangleOffset + triangleIndex;
    const uint vtxIdx = triIdx * 3; // The emissive triangle list is non-indexed.

    // Store vertex data.
    float3 p[3];
    float2 texC[3];
    gScene.getVertexPositionsW(gMeshInstanceID, triangleIndex, p);
    gScene.getVertexTexCoords(gMeshInstanceID, triangleIndex, texC);

    [unroll]
    for (int i = 0; i < 3; i++)
    {
        gVertexPosOutput.Store3((vtxIdx + i) * 12, asuint(p[i]));
        gTexCoordsOutput.Store2((vtxIdx + i) * 8, asuint(texC[i]));
    }

    // Store triangle data.
    float triangleArea;
    float3 faceNormal = gScene.computeFaceNormalAndAreaW(gMeshInstanceID, p, triangleArea);

    gTriangleData[triIdx].lightIdx = gLightIdx;
    gTriangleData[triIdx].normal = faceNormal;
    gTriangleData[triIdx].area = triangleArea;
}

/** Kernel updating the emissive triangles for all mesh lights.
    Single dispatch with one thread per triangle.
*/
[numthreads(256, 1, 1)]
void updateTriangleVertices(uint3 DTid : SV_DispatchThreadID)
{
    const uint triIdx = DTid.x; // Global emissive triangle index.
    const uint vtxIdx = triIdx * 3; // The emissive triangle list is non-indexed.

    if (triIdx >= gTriangleCount) return;

    // Get the data for the mesh that this triangle belongs to.
    uint lightIdx = gTriangleData[triIdx].lightIdx;
    const MeshLightData meshData = gMeshData[lightIdx];

    uint meshInstanceID = meshData.meshInstanceID;
    uint triangleIndex = triIdx - meshData.triangleOffset; // Local triangle index in the mesh

    // Update vertex data.
    float3 p[3];
    gScene.getVertexPositionsW(meshInstanceID, triangleIndex, p);

    [unroll]
    for (int i = 0; i < 3; i++)
    {
        gVertexPosOutput.Store3((vtxIdx + i) * 12, asuint(p[i]));
    }

    // Update triangle data.
    float triangleArea;
    float3 faceNormal = gScene.computeFaceNormalAndAreaW(meshInstanceID, p, triangleArea);

    gTriangleData[triIdx].normal = faceNormal;
    gTriangleData[triIdx].area = triangleArea;
}

/** Kernel computing the final pre-integrated triangle average radiance and flux.
    One dispatch with one thread per triangle (the dispatch is arranged as Y blocks of 256x1 threads).
*/
[numthreads(256, 1, 1)]
void finalizeIntegration(uint3 DTid : SV_DispatchThreadID)
{
    const uint triIdx = DTid.y * 256 + DTid.x;
    if (triIdx >= gLights.triangleCount) return;

    // Compute the triangle's average emitted radiance (RGB).
    const MeshLightData lightData = gLights.getMeshData(triIdx);
    float3 averageEmissiveColor = lightData.emissiveColor;

    if (lightData.isTextured())
    {
        // Compute the triangle's average textured emissive color based on the pre-integration results.
        // The alpha channel stores the texel count. If it is zero, the triangle didn't cover any samples
        // and we don't know anything about its radiance. In that case, assign a default value (1.0) to avoid bias.
        // TODO: This fallback can be removed when we use conservative rasterization in the pre-integration pass.
        float4 texelSum = asfloat(gTexelSum.Load4(triIdx * 16u));
        float weight = texelSum.a;
        averageEmissiveColor = weight > 0.f ? texelSum.rgb / weight : float3(1.0f);
    }
    float3 averageRadiance = averageEmissiveColor * lightData.emissiveFactor;
   
    // Pre-compute the luminous flux emitted, which is what we use during sampling to set probabilities.
    // We assume diffuse emitters and integrate per side (hemisphere) => the scale factor is pi.
    float area = gTriangleData[triIdx].area;                        // Triangle area in m^2 (the scene units are assumed to be in meters).
    float flux = luminance(averageRadiance) * area * (float) M_PI;  // Flux in lumens.

    gTriangleData[triIdx].averageRadiance = averageRadiance;
    gTriangleData[triIdx].flux = flux;
}
