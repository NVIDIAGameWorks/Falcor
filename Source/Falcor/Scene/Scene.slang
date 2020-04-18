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
 **************************************************************************/
#include "Scene/Material/MaterialDefines.slangh"

__exported import Scene.SceneTypes;
__exported import Scene.Camera.Camera;
__exported import Scene.Lights.LightData;
__exported import Scene.Lights.LightProbeData;
__exported import Scene.Material.MaterialData;

import HitInfo;
import TextureSampler;
import Utils.Attributes;
import Experimental.Scene.Lights.LightCollection;
import Experimental.Scene.Material.TexLODHelpers;

#ifndef MATERIAL_COUNT
// This error occurs when a shader imports Scene.slang without setting the defines
// returned by Scene::getSceneDefines().
#error "MATERIAL_COUNT not defined!"
#endif

/** Data required for rendering
*/
struct Scene
{
    // Geometry
    [root] StructuredBuffer<MeshInstanceData> meshInstances;
    StructuredBuffer<MeshDesc> meshes;

    [root] StructuredBuffer<float4> worldMatrices;
    [root] StructuredBuffer<float4> inverseTransposeWorldMatrices; // TODO: Make this 3x3 matrices (stored as 4x3). See #795.
    StructuredBuffer<float4> previousFrameWorldMatrices;

    [root] StructuredBuffer<PackedStaticVertexData> vertices;       ///< Vertex data for this frame.
    StructuredBuffer<PrevVertexData> prevVertices;                  ///< Vertex data for the previous frame, to handle skinned meshes.
    [root] ByteAddressBuffer indices;                               ///< Vertex indices, three 32-bit indices per triangle packed tightly.

    // Materials
    StructuredBuffer<MaterialData> materials;
    MaterialResources materialResources[MATERIAL_COUNT];

    // Lights and camera
    StructuredBuffer<LightData> lights;
    LightCollection lightCollection;
    LightProbeData lightProbe;
    Texture2D envMap;
    Camera camera;


    float4x4 loadWorldMatrix(uint matrixID)
    {
        float4x4 m =
        {
            worldMatrices[matrixID * 4 + 0],
            worldMatrices[matrixID * 4 + 1],
            worldMatrices[matrixID * 4 + 2],
            worldMatrices[matrixID * 4 + 3]
        };
        return m;
    }

    float4x4 getWorldMatrix(uint meshInstanceID)
    {
        uint matrixID = meshInstances[meshInstanceID].globalMatrixID;
        return loadWorldMatrix(matrixID);
    };

    float3x3 loadInverseTransposeWorldMatrix(uint matrixID)
    {
        float3x3 m =
        {
            inverseTransposeWorldMatrices[matrixID * 4 + 0].xyz,
            inverseTransposeWorldMatrices[matrixID * 4 + 1].xyz,
            inverseTransposeWorldMatrices[matrixID * 4 + 2].xyz
        };
        return m;
    }

    float3x3 getInverseTransposeWorldMatrix(uint meshInstanceID)
    {
        uint matrixID = meshInstances[meshInstanceID].globalMatrixID;
        return loadInverseTransposeWorldMatrix(matrixID);
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

    MaterialData getMaterial(uint materialID)
    {
        return materials[materialID];
    }

    MeshInstanceData getMeshInstance(uint meshInstanceID)
    {
        return meshInstances[meshInstanceID];
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
        uint baseIndex = meshInstances[meshInstanceID].ibOffset + (triangleIndex * 3);
        uint3 vtxIndices = indices.Load3(baseIndex * 4);
        vtxIndices += meshInstances[meshInstanceID].vbOffset;
        return vtxIndices;
    }

    /** Returns vertex data for a vertex.
        \param[in] index Global vertex index.
        \return Vertex data.
    */
    StaticVertexData getVertex(uint index)
    {
        return vertices[index].unpack();
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
        float3x3 worldInvTransposeMat = getInverseTransposeWorldMatrix(meshInstanceID);
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

        StaticVertexData vtx[3] = { gScene.getVertex(vtxIndices[0]), gScene.getVertex(vtxIndices[1]), gScene.getVertex(vtxIndices[2]) };

        v.posW += vtx[0].position * barycentrics[0];
        v.posW += vtx[1].position * barycentrics[1];
        v.posW += vtx[2].position * barycentrics[2];

        v.normalW += vtx[0].normal * barycentrics[0];
        v.normalW += vtx[1].normal * barycentrics[1];
        v.normalW += vtx[2].normal * barycentrics[2];

        v.bitangentW += vtx[0].bitangent * barycentrics[0];
        v.bitangentW += vtx[1].bitangent * barycentrics[1];
        v.bitangentW += vtx[2].bitangent * barycentrics[2];

        v.texC += vtx[0].texCrd * barycentrics[0];
        v.texC += vtx[1].texCrd * barycentrics[1];
        v.texC += vtx[2].texCrd * barycentrics[2];

        v.faceNormalW = getFaceNormalInObjectSpace(vtxIndices);

        float4x4 worldMat = getWorldMatrix(meshInstanceID);
        float3x3 worldInvTransposeMat = getInverseTransposeWorldMatrix(meshInstanceID);

        v.posW = mul(float4(v.posW, 1.f), worldMat).xyz;
        v.normalW = mul(v.normalW, worldInvTransposeMat);
        v.faceNormalW = mul(v.faceNormalW, worldInvTransposeMat);
        v.bitangentW = mul(v.bitangentW, (float3x3)worldMat);

        v.normalW = normalize(v.normalW);
        v.faceNormalW = normalize(v.faceNormalW);
        // Handle invalid bitangents gracefully (avoid NaN from normalization).
        v.bitangentW = dot(v.bitangentW, v.bitangentW) > 0.f ? normalize(v.bitangentW) : float3(0, 0, 0);
        return v;
    }

    /** Returns the interpolated vertex attributes for a given hitpoint.
        \param[in] hit Hit info.
        \return Interpolated vertex attributes.
    */
    VertexData getVertexData(HitInfo hit)
    {
        return getVertexData(hit.meshInstanceID, hit.primitiveIndex, hit.getBarycentricWeights());
    }

    /** Returns interpolated vertex attributes in a ray tracing hit program when ray cones are used for texture LOD.
        \param[in] meshInstanceID The mesh instance ID.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[in] barycentrics Barycentric coordinates in the triangle.
        \return Interpolated vertex attributes.
    */
    VertexData getVertexDataRayCones(uint meshInstanceID, uint triangleIndex, float3 barycentrics)
    {
        const uint3 vtxIndices = getIndices(meshInstanceID, triangleIndex);
        VertexData v = getVertexData(meshInstanceID, triangleIndex, barycentrics);

        float2 txcoords[3];
        float3 vtxs[3];
        txcoords[0] = vertices[vtxIndices[0]].texCrd;
        txcoords[1] = vertices[vtxIndices[1]].texCrd;
        txcoords[2] = vertices[vtxIndices[2]].texCrd;
        vtxs[0] = vertices[vtxIndices[0]].position;
        vtxs[1] = vertices[vtxIndices[1]].position;
        vtxs[2] = vertices[vtxIndices[2]].position;

        float4x4 worldMat = getWorldMatrix(meshInstanceID);
        v.coneTexLODValue = computeRayConeTriangleLODValue(vtxs, txcoords, float3x3(worldMat), v.faceNormalW);

        return v;
    }

    /** Returns interpolated vertex attributes in a ray tracing hit program when ray differentials are used for texture LOD.
        \param[in] meshInstanceID The mesh instance ID.
        \param[in] triangleIndex Index of the triangle in the given mesh.
        \param[in] barycentrics Barycentric coordinates in the triangle.
        \param[in] rayDir Ray direction.
        \param[in] hitT Distance to hit point.
        \param[in,out] rayDiff The ray differential used as input and output.
        \param[out] dUVdx The differential of the texture coordinates in pixel coordinate x.
        \param[out] dUVdy The differential of the texture coordinates in pixel coordinate y.
        \return Interpolated vertex attributes.
    */
    VertexData getVertexDataRayDiff(uint meshInstanceID, uint triangleIndex, float3 barycentrics, in float3 rayDir, in float hitT,
        inout RayDiff rayDiff, out float2 dUVdx, out float2 dUVdy)
    {
        const uint3 vtxIndices = getIndices(meshInstanceID, triangleIndex);
        VertexData v = getVertexData(meshInstanceID, triangleIndex, barycentrics);

        float3 unnormalizedN;
        float3 vtxs[3];
        float3 normals[3];
        float2 txcoords[3];
        float2 dBarydx, dBarydy;
        float4x4 worldMat = getWorldMatrix(meshInstanceID);
        float3x3 worldInvTransposeMat = getInverseTransposeWorldMatrix(meshInstanceID);

        StaticVertexData vtx[3] = { gScene.getVertex(vtxIndices[0]), gScene.getVertex(vtxIndices[1]), gScene.getVertex(vtxIndices[2]) };

        vtxs[0] = mul(float4(vtx[0].position, 1.0), worldMat).xyz;
        vtxs[1] = mul(float4(vtx[1].position, 1.0), worldMat).xyz;
        vtxs[2] = mul(float4(vtx[2].position, 1.0), worldMat).xyz;
        normals[0] = normalize(mul(vtx[0].normal, worldInvTransposeMat));
        normals[1] = normalize(mul(vtx[1].normal, worldInvTransposeMat));
        normals[2] = normalize(mul(vtx[2].normal, worldInvTransposeMat));
        txcoords[0] = vtx[0].texCrd;
        txcoords[1] = vtx[1].texCrd;
        txcoords[2] = vtx[2].texCrd;

        unnormalizedN = normals[0] * barycentrics[0];
        unnormalizedN += normals[1] * barycentrics[1];
        unnormalizedN += normals[2] * barycentrics[2];

        float3 e1 = vtxs[1] - vtxs[0];
        float3 e2 = vtxs[2] - vtxs[0];
        computeDifferentialsBarysAndUVs(rayDiff, rayDir, vtxs, txcoords, e1, e2, v.faceNormalW, hitT, dBarydx, dBarydy, dUVdx, dUVdy);
        reflectRayDifferential(rayDiff, rayDir, unnormalizedN, v.normalW, dBarydx, dBarydy, e1, e2, normals);

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
            prevPos += prevVertices[vtxIndices[i]].position * barycentrics[i];
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

    // Emissive access

    /** Check if a material has an emissive texture.
        \param[in] materialID Material ID
        \return Returns true if material has emissive texture.
    */
    bool isEmissiveTextured(uint materialID)
    {
        return EXTRACT_EMISSIVE_TYPE(materials[materialID].flags) == ChannelTypeTexture;
    }

    /** Evaluates the outgoing radiance on a given material.
        This function samples the emissive texture if its textured.
        Note the caller check the facing status themselves to discard back-facing hits.
        \param[in] materialID Material ID
        \param[in] uv Texture coordinates
        \param[in] mr MaterialResources struct for the given material ID. This is a workaround to avoid perf problems due to dynamic indexing.
        \return Evaluated outgoing radiance on the front-facing side.
    */
    float3 getEmissive(uint materialID, float2 uv, MaterialResources mr)
    {
        MaterialData m = materials[materialID];
        ExplicitLodTextureSampler lod = { 0.0 };
        return sampleTexture(mr.emissive, mr.samplerState, uv, float4(m.emissive, 1.0), EXTRACT_EMISSIVE_TYPE(m.flags), lod).rgb * m.emissiveFactor;
    }
};

ParameterBlock<Scene> gScene;
