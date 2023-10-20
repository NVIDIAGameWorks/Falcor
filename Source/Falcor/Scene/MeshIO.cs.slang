/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
import Scene.SceneTypes;
import Scene.Scene;

struct MeshLoader
{
    uint vertexCount;
    uint vbOffset;
    uint triangleCount;
    uint ibOffset;
    bool use16BitIndices;

    ParameterBlock<Scene> scene;

    // Output
    RWStructuredBuffer<float3> positions;
    RWStructuredBuffer<float3> texcrds;
    RWStructuredBuffer<uint3> triangleIndices;

    void getMeshIndices(uint triangleId)
    {
        if (triangleId >= triangleCount) return;
        uint3 vtxIndices = scene.getLocalIndices(ibOffset, triangleId, use16BitIndices);
        triangleIndices[triangleId] = vtxIndices;
    }

    void getMeshVertexData(uint vertexId)
    {
        if (vertexId >= vertexCount) return;
        StaticVertexData vtxData = scene.getVertex(vertexId + vbOffset);
        positions[vertexId] = vtxData.position;
        texcrds[vertexId] = float3(vtxData.texCrd, 0.f);
    }
};

struct MeshUpdater
{
    uint vertexCount;
    uint vbOffset;

    StructuredBuffer<float3> positions;
    StructuredBuffer<float3> normals;
    StructuredBuffer<float3> tangents;
    StructuredBuffer<float3> texcrds;

    // Output
    RWStructuredBuffer<PackedStaticVertexData> vertexData;

    void setMeshVertexData(uint vertexId)
    {
        if (vertexId >= vertexCount) return;
        StaticVertexData vtxData;
        vtxData.position = positions[vertexId];
        vtxData.normal = normals[vertexId];
        vtxData.tangent = float4(tangents[vertexId], 1.f); // Tangent follows the orientation such that `b = cross(n, t)`.
        vtxData.texCrd = texcrds[vertexId].xy;
        vertexData[vertexId + vbOffset].pack(vtxData);
    }
};

ParameterBlock<MeshLoader> meshLoader;
ParameterBlock<MeshUpdater> meshUpdater;

[numthreads(256, 1, 1)]
void getMeshVerticesAndIndices(uint3 tid: SV_DispatchThreadID)
{
    meshLoader.getMeshIndices(tid.x);
    meshLoader.getMeshVertexData(tid.x);
}

[numthreads(256, 1, 1)]
void setMeshVertices(uint3 tid: SV_DispatchThreadID)
{
    meshUpdater.setMeshVertexData(tid.x);
}
