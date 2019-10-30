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
#include "LightCollectionShared.h"
#include "MeshLightData.h"

import Shading;

/** This struct represents all emissive geometry in the scene.

    All triangle vertices are pre-transformed to world space and important
    quantities such as emitted flux are pre-computed.

    The user code instantiates LightCollection in e.g. a parameter block and
    calls member functions on the CPU-side class to update/bind it to their program.
*/
struct LightCollection
{
    uint                triangleCount;                      ///< Total number of triangles in all mesh lights.
    uint                meshCount;                          ///< Total number of mesh lights.

    ByteAddressBuffer   meshLightsVertexPos;                ///< Vertex positions in world space for all mesh light triangles. Size: triangleCount * 3 * sizeof(float3).
    ByteAddressBuffer   meshLightsTexCoords;                ///< Texture coordinates for all mesh light triangles. Size: triangleCount * 3 * sizeof(float2).
    Buffer<uint>        perMeshInstanceOffset;              ///< Per-mesh instance offset into emissive triangles array. Size: meshInstanceCount * sizeof(uint).
    StructuredBuffer<EmissiveTriangle>  triangleData;       ///< Per-triangle data for emissive triangles. Size: triangleCount * sizeof(EmissiveTriangle).
    StructuredBuffer<MeshLightData>     meshData;           ///< Per-mesh data for emissive meshes. Size. _NUM_MESH_LIGHTS * sizeof(MeshLightData).

#if _NUM_MESH_LIGHTS > 0
    // TODO: Replace emissive textures array by materialID indices into the Scene's material array.
    Texture2D           emissiveTextures[_NUM_MESH_LIGHTS]; ///< Per-mesh light emissive textures, or null if not textured.
#endif
    SamplerState        samplerState;                       ///< Sampler state to be used for textured lights.

    // Accessors

    /** Returns the total number of emissive triangles.
    */
    uint getTriangleCount() { return triangleCount; }

    /** Returns the data for a given triangle.
    */
    EmissiveTriangle getTriangleData(uint triIdx)
    {
        return triangleData[triIdx];
    }

    /** Returns the mesh light index for a given triangle.
    */
    uint getLightIdx(uint triIdx)
    {
        return triangleData[triIdx].lightIdx;
    }

    /** Returns the mesh light data for a given emissive triangle.
        \param[in] triIdx Emissive triangle index.
        \return Mesh light data.
    */
    MeshLightData getMeshData(uint triIdx)
    {
#if _NUM_MESH_LIGHTS > 0
        return meshData[getLightIdx(triIdx)];
#else
        MeshLightData data;
        data.init();
        return data;
#endif
    }

    /** Returns world space vertex position based on global vertex index.
        Note that the vertex buffer is non-indexed.
    */
    float3 getVtxPos(uint vtxIdx)
    {
        uint address = (vtxIdx * 3) * 4;    // The data is float3 (3*4B per vertex).
        return asfloat(meshLightsVertexPos.Load3(address));
    }

    /** Returns texture coordinate based on global vertex index.
        Note that the vertex buffer is non-indexed.
    */
    float2 getVtxTexCoord(uint vtxIdx)
    {
        int address = (vtxIdx * 2) * 4;     // The data is float2 (2*4B per vertex).
        return asfloat(meshLightsTexCoords.Load2(address));
    }

    /** Returns a triangle's vertex positions in world space.
        \param[in] triIdx Emissive triangle index.
        \param[out] p Position of vertex 0,1,2 in world space.
    */
    void getVtxPositions(uint triIdx, out float3 p[3])
    {
        p[0] = getVtxPos(triIdx * 3 + 0);
        p[1] = getVtxPos(triIdx * 3 + 1);
        p[2] = getVtxPos(triIdx * 3 + 2);
    }

    /** Returns interpolated position on a given triangle.
        \param[in] triIdx Emissive triangle index.
        \param[in] barycentrics Barycentric coordinates.
        \return Interpolated position in world space.
    */
    float3 getPosition(uint triIdx, float3 barycentrics)
    {
        return getVtxPos(triIdx * 3 + 0) * barycentrics[0] +
               getVtxPos(triIdx * 3 + 1) * barycentrics[1] +
               getVtxPos(triIdx * 3 + 2) * barycentrics[2];
    }

    /** Returns interpolated texture coordinate on a given triangle.
        \param[in] triIdx Emissive triangle index.
        \param[in] barycentrics Barycentric coordinates.
        \return Interpolated texture coordinate.
    */
    float2 getTexCoord(uint triIdx, float3 barycentrics)
    {
        return getVtxTexCoord(triIdx * 3 + 0) * barycentrics[0] +
               getVtxTexCoord(triIdx * 3 + 1) * barycentrics[1] +
               getVtxTexCoord(triIdx * 3 + 2) * barycentrics[2];
    }

#if _NUM_MESH_LIGHTS > 0
    /** Evaluates the outgoing radiance on the front-facing side of a given triangle.
        This function samples the triangle's emissive texture if its textured.
        Note the caller check the facing status themselves to discard back-facing hits.
        \param[in] triIdx Emissive triangle index.
        \param[in] barycentrics Barycentric coordinates.
        \return Evaluated outgoing radiance on the front-facing side.
    */
    float3 getEmissive(uint triIdx, float3 barycentrics)
    {
        // Get the mesh light index. We need this below to fetch material properties
        // in order to evaluate the triangles emissive color (possibly textured).
        uint lightIndex = getLightIdx(triIdx);

        // Interpolate the tex coords.
        float2 uv = getTexCoord(triIdx, barycentrics);

        // Sample emitted radiance.
        ExplicitLodTextureSampler lod = { 0.f };
        float3 emissiveColor = meshData[lightIndex].emissiveColor;
        return sampleTexture(emissiveTextures[lightIndex], samplerState, uv, float4(emissiveColor, 1.0f), EXTRACT_EMISSIVE_TYPE(meshData[lightIndex].flags), lod).rgb * meshData[lightIndex].emissiveFactor;
    }
#endif
};
