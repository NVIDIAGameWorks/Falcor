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
import Scene.SceneTypes;

/** Compute pass for skinned vertex animation.

    The dispatch size is one thread per dynamic vertex.
*/


struct SkinningData
{
    bool initPrev; ///< Copy current frame data to previous frame data.

    // Vertex data
    StructuredBuffer<PackedStaticVertexData> staticData;            ///< Original global vertex buffer. This holds the unmodified input vertices.
    StructuredBuffer<SkinningVertexData> skinningData;              ///< Bone IDs and weights for all skinned vertices.
    RWStructuredBuffer<PackedStaticVertexData> skinnedVertices;     ///< Skinned global vertex buffer. We'll update the positions only for the dynamic meshes.
    RWStructuredBuffer<PrevVertexData> prevSkinnedVertices;         ///< Previous frame vertex positions for all dynamic meshes.

    // Transforms
    StructuredBuffer<float4x4> boneMatrices;
    StructuredBuffer<float4x4> inverseTransposeBoneMatrices;
    StructuredBuffer<float4x4> worldMatrices;
    StructuredBuffer<float4x4> meshBindMatrices;
    StructuredBuffer<float4x4> meshInvBindMatrices;
    StructuredBuffer<float4x4> inverseTransposeWorldMatrices;

    // Accessors

    float4x4 getTransposeWorldMatrix(uint matrixID)
    {
        return transpose(worldMatrices[matrixID]);
    }

    float4x4 getInverseWorldMatrix(uint matrixID)
    {
        return transpose(inverseTransposeWorldMatrices[matrixID]);
    }

    float4x4 getMeshBindMatrix(uint matrixID)
    {
        return meshBindMatrices[matrixID];
    }

    float4x4 getInverseMeshBindMatrix(uint matrixID)
    {
        return meshInvBindMatrices[matrixID];
    }

    float4x4 getBoneMatrix(uint matrixID)
    {
        return boneMatrices[matrixID];
    }

    float4x4 getInverseTransposeBoneMatrix(uint matrixID)
    {
        return inverseTransposeBoneMatrices[matrixID];
    }

    float4x4 getBlendedMatrix(uint vertexId)
    {
        SkinningVertexData s = skinningData[vertexId];

        float4x4 boneMat = getBoneMatrix(s.boneID.x) * s.boneWeight.x;
        boneMat += getBoneMatrix(s.boneID.y) * s.boneWeight.y;
        boneMat += getBoneMatrix(s.boneID.z) * s.boneWeight.z;
        boneMat += getBoneMatrix(s.boneID.w) * s.boneWeight.w;

        // Apply mesh bind transform before skinning (mesh to skeleton local at bind time)
        boneMat = mul(boneMat, getMeshBindMatrix(s.bindMatrixID));

        // Skinning takes us to world space, so apply the inverse to return to skeleton local
        boneMat = mul(getInverseWorldMatrix(s.skeletonMatrixID), boneMat);

        // Apply inverse bind matrix to return to mesh-local
        boneMat = mul(getInverseMeshBindMatrix(s.bindMatrixID), boneMat);
        return boneMat;
    }

    float4x4 getInverseTransposeBlendedMatrix(uint vertexId)
    {
        SkinningVertexData s = skinningData[vertexId];

        float4x4 boneMat = getInverseTransposeBoneMatrix(s.boneID.x) * s.boneWeight.x;
        boneMat += getInverseTransposeBoneMatrix(s.boneID.y) * s.boneWeight.y;
        boneMat += getInverseTransposeBoneMatrix(s.boneID.z) * s.boneWeight.z;
        boneMat += getInverseTransposeBoneMatrix(s.boneID.w) * s.boneWeight.w;

        boneMat = mul(getTransposeWorldMatrix(s.skeletonMatrixID), boneMat);
        return boneMat;
    }

    uint getStaticVertexID(uint vertexId)
    {
        return skinningData[vertexId].staticIndex;
    }

    StaticVertexData getStaticVertexData(uint vertexId)
    {
        return staticData[getStaticVertexID(vertexId)].unpack();
    }

    void storeSkinnedVertexData(uint vertexId, StaticVertexData data, PrevVertexData prevData)
    {
        gData.skinnedVertices[getStaticVertexID(vertexId)].pack(data);
        gData.prevSkinnedVertices[vertexId] = prevData;
    }

    float3 getCurrentPosition(uint vertexId)
    {
        return gData.skinnedVertices[getStaticVertexID(vertexId)].position;
    }
};

ParameterBlock<SkinningData> gData;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    // Check that this is an active vertex
    uint vertexId = dispatchThreadID.x;
    uint vertexCount, stride;
    gData.skinningData.GetDimensions(vertexCount, stride);
    if (vertexId >= vertexCount) return;

    // Blend the vertices
    StaticVertexData s = gData.getStaticVertexData(vertexId);
    float4x4 boneMat = gData.getBlendedMatrix(vertexId);
    float4x4 invTransposeMat = gData.getInverseTransposeBlendedMatrix(vertexId);

    s.position = mul(boneMat, float4(s.position, 1.f)).xyz;
    s.tangent.xyz = mul((float3x3) boneMat, s.tangent.xyz);
    s.normal = mul((float3x3) transpose(invTransposeMat), s.normal);

    // Copy the previous skinned data
    PrevVertexData prev;
    prev.position = gData.initPrev ? s.position : gData.getCurrentPosition(vertexId);

    // Store the result
    gData.storeSkinnedVertexData(vertexId, s, prev);
}
