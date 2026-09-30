/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "Common.hlsl"

cbuffer PerFrameCB
{
    int2 taskDim;
    int2 screenDim;
    int mipLevel;
};

//RWStructuredBuffer<Photon> gPhotonBuffer;
//RWStructuredBuffer<DrawArguments> gDrawArgument;
RWStructuredBuffer<RayArgument> gRayArgument;
//RWStructuredBuffer<RayTask> gRayTask;
//StructuredBuffer<PixelInfo> gPixelInfo;
//Texture2D gDepthTex;
Texture2D<float4> gRayDensityTex;
RWStructuredBuffer<uint4> gRayCountQuadTree;

int getRayTaskID(uint2 pos)
{
    return pos.y * taskDim.x + pos.x;
}

uint getSampleCount(float v00, float v10, float v01, float v11)
{
    float sampleCountF = 0.25 * (v00 + v10 + v01 + v11);
    int sampleDim = (int) ceil(sqrt(sampleCountF));
    uint sampleCount = sampleDim * sampleDim;
    return sampleCount;
}

[numthreads(8, 8, 1)]
void generateMip0(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 threadIdx : SV_DispatchThreadID)
{
    uint rayIdx = getRayTaskID(threadIdx.xy);
    //gPixelInfo[rayIdx].screenArea = 0;
    //gPixelInfo[rayIdx].screenAreaSq = 0;
    //gPixelInfo[rayIdx].count = 0;

    //int idx0 = gPixelInfo[rayIdx].photonIdx;

    int2 pixel00 = threadIdx.xy * 2;
    float v00 = gRayDensityTex.Load(int3(pixel00 + int2(0, 0), 0)).r;
    float v10 = gRayDensityTex.Load(int3(pixel00 + int2(1, 0), 0)).r;
    float v20 = gRayDensityTex.Load(int3(pixel00 + int2(2, 0), 0)).r;

    float v01 = gRayDensityTex.Load(int3(pixel00 + int2(0, 1), 0)).r;
    float v11 = gRayDensityTex.Load(int3(pixel00 + int2(1, 1), 0)).r;
    float v21 = gRayDensityTex.Load(int3(pixel00 + int2(2, 1), 0)).r;

    float v02 = gRayDensityTex.Load(int3(pixel00 + int2(0, 2), 0)).r;
    float v12 = gRayDensityTex.Load(int3(pixel00 + int2(1, 2), 0)).r;
    float v22 = gRayDensityTex.Load(int3(pixel00 + int2(2, 2), 0)).r;

    uint count00 = getSampleCount(v00, v10, v01, v11);
    uint count10 = count00 + getSampleCount(v10, v20, v11, v21);
    uint count01 = count10 + getSampleCount(v01, v11, v02, v12);
    uint count11 = count01 + getSampleCount(v11, v21, v12, v22);
    //int sampleCount = count00 + count10 + count01 + count11;
    //int taskIdx = 0;
    //InterlockedAdd(gRayArgument[0].rayTaskCount, sampleCount, taskIdx);

    int offset = getTextureOffset(threadIdx.xy, mipLevel);
    gRayCountQuadTree[offset] = uint4(count00, count10, count01, count11);
}

[numthreads(8, 8, 1)]
void generateMipLevel(uint3 threadIdx : SV_DispatchThreadID)
{
    int mipDim = getMipSize(mipLevel);
    if (any(threadIdx.xy >= mipDim))
    {
        return;
    }

    int2 pixel00 = threadIdx.xy * 2;
    int nextMipLevel = mipLevel + 1;
    uint4 count00 = gRayCountQuadTree[getTextureOffset(pixel00 + int2(0, 0), nextMipLevel)];
    uint4 count10 = gRayCountQuadTree[getTextureOffset(pixel00 + int2(1, 0), nextMipLevel)];
    uint4 count01 = gRayCountQuadTree[getTextureOffset(pixel00 + int2(0, 1), nextMipLevel)];
    uint4 count11 = gRayCountQuadTree[getTextureOffset(pixel00 + int2(1, 1), nextMipLevel)];

    uint4 value;
    value.x = count00.w;
    value.y = value.x + count10.w;
    value.z = value.y + count01.w;
    value.w = value.z + count11.w;
    gRayCountQuadTree[getTextureOffset(threadIdx.xy, mipLevel)] = value; // value;
}
