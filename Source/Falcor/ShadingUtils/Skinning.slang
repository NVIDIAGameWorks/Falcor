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
#pragma once
#include "HostDeviceData.h"

struct SkinningData
{
    StructuredBuffer<StaticVertexData> staticData;
    StructuredBuffer<DynamicVertexData> dynamicData;
    RWStructuredBuffer<StaticVertexData> skinnedVertices;
    Buffer<float4> boneMatrices;
    Buffer<float4> inverseTransposeBoneMatrices;
    Buffer<float4> worldMatrices;
    Buffer<float4> inverseTransposeWorldMatrices;

    float4x4 getTransposeWorldMatrix(uint matrixID)
    {
        float4x4 m = float4x4(worldMatrices[matrixID * 4 + 0],
        worldMatrices[matrixID * 4 + 1],
        worldMatrices[matrixID * 4 + 2],
        worldMatrices[matrixID * 4 + 3]);

        return transpose(m);
    }

    float4x4 getInverseWorldMatrix(uint matrixID)
    {
        float4x4 m = float4x4(inverseTransposeWorldMatrices[matrixID * 4 + 0],
        inverseTransposeWorldMatrices[matrixID * 4 + 1],
        inverseTransposeWorldMatrices[matrixID * 4 + 2],
        inverseTransposeWorldMatrices[matrixID * 4 + 3]);

        return transpose(m);
    }

    float4x4 getBoneMatrix(uint matrixID)
    {
        return float4x4(boneMatrices[matrixID * 4 + 0],
        boneMatrices[matrixID * 4 + 1],
        boneMatrices[matrixID * 4 + 2],
        boneMatrices[matrixID * 4 + 3]);
    }

    float4x4 getInverseTransposeBoneMatrix(uint matrixID)
    {
        return float4x4(inverseTransposeBoneMatrices[matrixID * 4 + 0],
        inverseTransposeBoneMatrices[matrixID * 4 + 1],
        inverseTransposeBoneMatrices[matrixID * 4 + 2],
        inverseTransposeBoneMatrices[matrixID * 4 + 3]);
    }

    float4x4 getBlendedMatrix(uint vertexId)
    {
        DynamicVertexData d = dynamicData[vertexId];

        float4x4 boneMat = getBoneMatrix(d.boneID.x) * d.boneWeight.x;
        boneMat += getBoneMatrix(d.boneID.y) * d.boneWeight.y;
        boneMat += getBoneMatrix(d.boneID.z) * d.boneWeight.z;
        boneMat += getBoneMatrix(d.boneID.w) * d.boneWeight.w;

        boneMat = mul(boneMat, getInverseWorldMatrix(d.globalMatrixID));
        return boneMat;
    }

    float4x4 getInverseTransposeBlendedMatrix(uint vertexId)
    {
        DynamicVertexData d = dynamicData[vertexId];

        float4x4 boneMat = getInverseTransposeBoneMatrix(d.boneID.x) * d.boneWeight.x;
        boneMat += getInverseTransposeBoneMatrix(d.boneID.y) * d.boneWeight.y;
        boneMat += getInverseTransposeBoneMatrix(d.boneID.z) * d.boneWeight.z;
        boneMat += getInverseTransposeBoneMatrix(d.boneID.w) * d.boneWeight.w;

        boneMat = mul(boneMat, getTransposeWorldMatrix(d.globalMatrixID));
        return boneMat;
    }

    uint getStaticVertexID(uint vertexId)
    {
        return dynamicData[vertexId].staticIndex;
    }

    StaticVertexData getStaticVertexData(uint vertexId)
    {
        return staticData[getStaticVertexID(vertexId)];
    }

    void storeStaticData(uint vertexId, StaticVertexData data)
    {
        gData.skinnedVertices[getStaticVertexID(vertexId)] = data;
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
    gData.dynamicData.GetDimensions(vertexCount, stride);
    if (vertexId >= vertexCount) return;

    // Blend the vertices
    StaticVertexData s = gData.getStaticVertexData(vertexId);
    float4x4 boneMat = gData.getBlendedMatrix(vertexId);
    float4x4 invTransposeMat = gData.getInverseTransposeBlendedMatrix(vertexId);

    s.prevPosition = gData.getCurrentPosition(vertexId);
    s.position = mul(float4(s.position, 1.f), boneMat).xyz;
    s.bitangent = mul(s.bitangent, (float3x3) boneMat);
    s.normal = mul(s.normal, (float3x3) transpose(invTransposeMat));

    // Store the result
    gData.storeStaticData(vertexId, s);
}

    
