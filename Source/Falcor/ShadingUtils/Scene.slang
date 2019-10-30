/***************************************************************************
# Copyright (c) 201*, NVIDIA CORPORATION. All rights reserved.
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
#include "HostDeviceData.h"

// Local stuff
struct LocalMesh
{
    uint meshID;
    uint meshInstanceID;
    uint materialID;
};

#ifndef MATERIAL_COUNT
#define MATERIAL_COUNT 1
#endif

/** Data required for rendering
*/
struct Scene
{
    StructuredBuffer<MeshInstanceData> meshInstances;
    StructuredBuffer<MeshDesc> meshes;
    StructuredBuffer<LightData> lights;

    Buffer<float4> worldMatrices;
    Buffer<float4> inverseTransposeWorldMatrices; // #SCENEV2 Should this be 3x3?
    Buffer<float4> previousFrameWorldMatrices;
    MaterialData materials[MATERIAL_COUNT];
    LightProbeData lightProbe;
    LightProbeSharedResources probeShared; // Should this be its own parameter block? Doesn't really fit here, and Lights.slang currently needs to import Scene
    CameraData camera;
    Texture2D envMap;

    StructuredBuffer<StaticVertexData> vertices;
    ByteAddressBuffer indices;

    float4x4 getWorldMatrix(uint meshInstanceID)
    {
        uint matrixID = meshInstances[meshInstanceID].globalMatrixID;

        float4x4 m =
        {
            worldMatrices[matrixID * 4 + 0],
            worldMatrices[matrixID * 4 + 1],
            worldMatrices[matrixID * 4 + 2],
            worldMatrices[matrixID * 4 + 3]
        };
        return m;
    };

    float4x4 getInverseTransposeWorldMatrix(uint meshInstanceID)
    {
        uint matrixID = meshInstances[meshInstanceID].globalMatrixID;

        float4x4 m =
        {
            inverseTransposeWorldMatrices[matrixID * 4 + 0],
            inverseTransposeWorldMatrices[matrixID * 4 + 1],
            inverseTransposeWorldMatrices[matrixID * 4 + 2],
            inverseTransposeWorldMatrices[matrixID * 4 + 3]
        };

        return m;
    };

    float4x4 getPrevWorldMatrix(uint meshInstanceID)
    {
        uint matrixID = meshInstances[meshInstanceID].globalMatrixID;

        float4x4 m =
        {
            previousFrameWorldMatrices[matrixID * 4 + 0],
            previousFrameWorldMatrices[matrixID * 4 + 1],
            previousFrameWorldMatrices[matrixID * 4 + 2],
            previousFrameWorldMatrices[matrixID * 4 + 3]
        };
        return m;
    };

    uint getMaterialID(uint meshInstanceID)
    {
        return meshInstances[meshInstanceID].materialID;
    };

    uint getMaterialCount()
    {
        return MATERIAL_COUNT;
    }

    MaterialData getMaterial(uint meshInstanceID)
    {
        return materials[getMaterialID(meshInstanceID)];
    }

    MeshDesc getMeshDesc(uint meshInstanceID)
    {
        return meshes[meshInstances[meshInstanceID].meshID];
    }

    uint getLightCount()
    {
        uint count, stride;
        lights.GetDimensions(count, stride);
        return count;
    }

    LightData getLight(uint lightIndex)
    {
        return lights[lightIndex];
    }

    bool isWorldMatrixFlippedWinding(uint meshInstanceID)
    {
        return (meshInstances[meshInstanceID].flags & uint(MeshInstanceFlags::Flipped)) != 0;
    }


    // Geometry access

    /** Returns the global vertex indices for a given triangle.
        \param[in] meshInstanceID The mesh instance ID.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \return Vertex indices into the global vertex buffer.
    */
    uint3 getIndices(uint meshInstanceID, uint triangleIndex)
    {
        uint baseIndex = getMeshDesc(meshInstanceID).ibOffset + (triangleIndex * 3);
        uint3 vtxIndices = indices.Load3(baseIndex * 4);
        vtxIndices += getMeshDesc(meshInstanceID).vbOffset;
        return vtxIndices;
    }

    /** Returns a triangle's face normal in object space.
        \param[in] vtxIndices Indices into the scene's global vertex buffer.
        \param[out] Face normal in object space (normalized). Front facing for counter-clockwise winding.
    */
    float3 getFaceNormalInObjectSpace(uint3 vtxIndices)
    {
        float3 p0 = vertices[vtxIndices[0]].position;
        float3 p1 = vertices[vtxIndices[1]].position;
        float3 p2 = vertices[vtxIndices[2]].position;
        return normalize(cross(p1 - p0, p2 - p0));
    }

    /** Returns a triangle's face normal in world space.
        \param[in] meshInstanceID The mesh instance ID.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[out] Face normal in world space (normalized).
    */
    float3 getFaceNormalW(uint meshInstanceID, uint triangleIndex)
    {
        uint3 vtxIndices = getIndices(meshInstanceID, triangleIndex);
        float3 p0 = vertices[vtxIndices[0]].position;
        float3 p1 = vertices[vtxIndices[1]].position;
        float3 p2 = vertices[vtxIndices[2]].position;
        float3 N = cross(p1 - p0, p2 - p0);
        float3x3 worldInvTransposeMat = (float3x3) getInverseTransposeWorldMatrix(meshInstanceID);
        return normalize(mul(N, worldInvTransposeMat));
    }

    /** Returns a triangle's area in world space.
        \param[in] meshInstanceID The mesh instance ID.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[out] Triangle area.
    */
    float getFaceAreaW(uint meshInstanceID, uint triangleIndex)
    {
        float3 p[3];
        getVertexPositionsW(meshInstanceID, triangleIndex, p);
        return 0.5f * length(cross(p[1] - p[0], p[2] - p[0]));
    }

    /** Computes the face normal and area of a triangle given its vertices.
        \param[in] meshInstanceID The mesh instance ID.
        \param[in] p Position of vertex 0,1,2 in world space.
        \param[out] triangleArea Triangle area in world space units.
        \return Face normal in world space (normalized).
    */
    float3 computeFaceNormalAndAreaW(uint meshInstanceID, const float3 p[3], out float triangleArea)
    {
        // Compute face normal in world space.
        // The length of the vector is twice the triangle area since we're in world space.
        // Note that this is not true if the normal is transformed using the inverse-transpose.
        float3 e[2];
        e[0] = p[1] - p[0];
        e[1] = p[2] - p[0];
        float3 N = cross(e[0], e[1]);
        triangleArea = 0.5f * length(N);

        // Flip the normal if the instance transform changed the handedness of the coordinate system.
        if (isWorldMatrixFlippedWinding(meshInstanceID)) N = -N;

        return normalize(N);
    }

    /** Returns a triangle's face normal and area in world space.
        This function should only be used if the triangle area is needed, as it is less efficient than computing just its normal.
        \param[in] meshInstanceID The mesh instance ID.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[out] triangleArea Triangle area in world space units.
        \return Face normal in world space (normalized).
    */
    float3 getFaceNormalAndAreaW(uint meshInstanceID, uint triangleIndex, out float triangleArea)
    {
        uint3 vtxIndices = getIndices(meshInstanceID, triangleIndex);

        // Load vertices and transform to world space.
        float3 p[3];
        [unroll]
        for (int i = 0; i < 3; i++)
        {
            p[i] = vertices[vtxIndices[i]].position;
            p[i] = mul(float4(p[i], 1.f), getWorldMatrix(meshInstanceID)).xyz;
        }

        return computeFaceNormalAndAreaW(meshInstanceID, p, triangleArea);
    }

    /** Returns the interpolated vertex attributes for a given triangle.
        \param[in] meshInstanceID The mesh instance ID.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[in] barycentrics Barycentric coordinates in the triangle.
        \return Interpolated vertex attributes.
    */
    VertexData getVertexData(uint meshInstanceID, uint triangleIndex, float3 barycentrics)
    {
        const uint3 vtxIndices = getIndices(meshInstanceID, triangleIndex);
        VertexData v = {};

        [unroll]
        for (int i = 0; i < 3; i++)
        {
            v.posW       += vertices[vtxIndices[i]].position  * barycentrics[i];
            v.normalW    += vertices[vtxIndices[i]].normal    * barycentrics[i];
            v.bitangentW += vertices[vtxIndices[i]].bitangent * barycentrics[i];
            v.texC       += vertices[vtxIndices[i]].texCrd    * barycentrics[i];
        }
        v.faceNormalW = getFaceNormalInObjectSpace(vtxIndices);

        float4x4 worldMat = getWorldMatrix(meshInstanceID);
        float3x3 worldInvTransposeMat = (float3x3) getInverseTransposeWorldMatrix(meshInstanceID);

        v.posW = mul(float4(v.posW, 1.f), worldMat).xyz;
        v.normalW = mul(v.normalW, worldInvTransposeMat).xyz;
        v.faceNormalW = mul(v.faceNormalW, worldInvTransposeMat).xyz;
        v.bitangentW = mul(v.bitangentW, (float3x3)worldMat).xyz;

        v.normalW = normalize(v.normalW);
        v.faceNormalW = normalize(v.faceNormalW);
        // Handle invalid bitangents gracefully (avoid NaN from normalization).
        v.bitangentW = dot(v.bitangentW, v.bitangentW) > 0.f ? normalize(v.bitangentW) : float3(0, 0, 0);

        return v;
    }

    /** Returns interpolated position on a triangle in world space for the previous frame.
        \param[in] meshInstanceID The mesh instance ID.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[in] barycentrics Barycentric coordinates in the triangle.
        \return Interpolated position in world space for the previous frame.
    */
    float3 getPrevPosW(uint meshInstanceID, uint triangleIndex, float3 barycentrics)
    {
        const uint3 vtxIndices = getIndices(meshInstanceID, triangleIndex);
        float3 prevPos = float3(0, 0, 0);

        [unroll]
        for (int i = 0; i < 3; i++)
        {
            prevPos += vertices[vtxIndices[i]].prevPosition * barycentrics[i];
        }

        float4x4 prevWorldMat = getPrevWorldMatrix(meshInstanceID);
        return mul(float4(prevPos, 1.f), prevWorldMat).xyz;
    }

    /** Returns a triangle's vertex positions in world space.
        \param[in] meshInstanceID The mesh instance ID.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[out] p Position of vertex 0,1,2 in world space.
    */
    void getVertexPositionsW(uint meshInstanceID, uint triangleIndex, out float3 p[3])
    {
        uint3 vtxIndices = getIndices(meshInstanceID, triangleIndex);
        float4x4 worldMat = getWorldMatrix(meshInstanceID);

        [unroll]
        for (int i = 0; i < 3; i++)
        {
            p[i] = vertices[vtxIndices[i]].position;
            p[i] = mul(float4(p[i], 1.f), worldMat).xyz;
        }
    }

    /** Returns a triangle's texture coordinates.
        \param[in] meshInstanceID The mesh instance ID.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[out] texC Texture coordinate of vertex 0,1,2.
    */
    void getVertexTexCoords(uint meshInstanceID, uint triangleIndex, out float2 texC[3])
    {
        uint3 vtxIndices = getIndices(meshInstanceID, triangleIndex);

        [unroll]
        for (int i = 0; i < 3; i++)
        {
            texC[i] = vertices[vtxIndices[i]].texCrd;
        }
    }
};

shared ParameterBlock<Scene> gScene;
