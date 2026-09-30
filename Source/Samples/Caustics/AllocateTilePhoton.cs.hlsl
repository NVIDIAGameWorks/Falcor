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

//import ShaderCommon;

#include "Common.hlsl"

cbuffer PerFrameCB
{
    float4x4 gViewProjMat;
    int2 screenDim;
    int2 tileDim;
    int2 blockCount;
    float gSplatSize;
    float minColor;
};

#define PHOTON_COUNT_BLOCK_SIZE 32

RWStructuredBuffer<DrawArguments> gDrawArgument;
RWStructuredBuffer<Photon> gPhotonBuffer;

RWStructuredBuffer<IDBlock> gTileInfo;
RWByteAddressBuffer gIDBuffer;
RWByteAddressBuffer gIDCounter;

int getTileOffset(int x, int y)
{
    return tileDim.x * y + x;
}

void GetPhotonScreenRange(Photon photon, out int2 minTileID, out int2 maxTileID)
{
    // get screen position
    float3 corner0 = photon.dPdx + photon.dPdy;
    float3 corner1 = photon.dPdx - photon.dPdy;
    float4 px0 = mul(float4(photon.posW + corner0, 1), gViewProjMat);
    float4 px1 = mul(float4(photon.posW - corner0, 1), gViewProjMat);
    float4 py0 = mul(float4(photon.posW + corner1, 1), gViewProjMat);
    float4 py1 = mul(float4(photon.posW - corner1, 1), gViewProjMat);
    if ((px0.z < 0 || px0.z > px0.w) ||
        (px1.z < 0 || px1.z > px1.w) ||
        (py0.z < 0 || py0.z > py0.w) ||
        (py1.z < 0 || py1.z > py1.w))
    {
        minTileID = 1;
        maxTileID = -1;
        return;
    }
    px0.xyz /= px0.w;
    px1.xyz /= px1.w;
    py0.xyz /= py0.w;
    py1.xyz /= py1.w;
    px0.y *= -1;
    px1.y *= -1;
    py0.y *= -1;
    py1.y *= -1;

    // get range
    float3 minCoord = 10000.0f;
    float3 maxCoord = -10000.0f;
    minCoord = min(minCoord, px0.xyz);
    minCoord = min(minCoord, px1.xyz);
    minCoord = min(minCoord, py0.xyz);
    minCoord = min(minCoord, py1.xyz);
    maxCoord = max(maxCoord, px0.xyz);
    maxCoord = max(maxCoord, px1.xyz);
    maxCoord = max(maxCoord, py0.xyz);
    maxCoord = max(maxCoord, py1.xyz);
    minCoord.xy = (minCoord.xy + 1) * 0.5;
    maxCoord.xy = (maxCoord.xy + 1) * 0.5;
    if (any(minCoord > 1) || any(maxCoord < 0))
    {
        minTileID = 1;
        maxTileID = -1;
        return;
    }
    minTileID = minCoord.xy * screenDim / uint2(GATHER_TILE_SIZE_X, GATHER_TILE_SIZE_Y);
    maxTileID = maxCoord.xy * screenDim / uint2(GATHER_TILE_SIZE_X, GATHER_TILE_SIZE_Y);

    minTileID = clamp(minTileID, 0, tileDim - 1);
    maxTileID = clamp(maxTileID, 0, tileDim - 1);
}

bool checkPhoton(Photon p)
{
    return dot(p.color, float3(1, 1, 1)) > minColor;
}

int threadIDToPhotonID(uint3 threadIdx)
{
    return threadIdx.y * blockCount.x * PHOTON_COUNT_BLOCK_SIZE + threadIdx.x;
}


bool orthogonalizeFrame(float3 a, float3 b, out float3 orthoA, out float3 orthoB)
{
    float a11 = dot(a, a);
    float a12 = dot(a, b);
    float a22 = dot(b, b);

    if (abs(a12) > 1e-5)
    {
        float sum = a11 + a22;
        float diff = a11 - a22;
        float delta = sqrt(diff * diff + 4 * a12 * a12);

        float lambda1 = (sum + delta) * 0.5f;
        float lambda2 = (sum - delta) * 0.5f;

        float2 p1 = normalize(float2(a12, lambda1 - a11));
        float2 p2 = normalize(float2(a12, lambda2 - a11));

        orthoA = a * p1.x + b * p1.y;
        orthoB = a * p2.x + b * p2.y;
    }
    else
    {
        orthoA = a;
        orthoB = b;
    }

    float3 normal = cross(a, b);
    float lengthA = length(orthoA);
    float lengthB = length(orthoB);
    float lengthRatio = lengthA / lengthB;
    const float ratioThreshold = 1e3;
    if (min(lengthA, lengthB) < 0.005 || lengthRatio < 1 / ratioThreshold || lengthRatio > ratioThreshold)
    {
        return false;
    }
    if (lengthB > lengthA)
    {
        float3 v = orthoB;
        orthoB = orthoA;
        orthoA = v;
    }
    float3 newNormal = cross(orthoA, orthoB);
    if (dot(normal, newNormal) < 0)
    {
        orthoB *= -1;
    }

    orthoA *= gSplatSize;
    orthoB *= gSplatSize;
    return true;
}

[numthreads(PHOTON_COUNT_BLOCK_SIZE, PHOTON_COUNT_BLOCK_SIZE, 1)]
void OrthogonalizePhoton(uint3 threadIdx : SV_DispatchThreadID)
{
    int photonID = threadIDToPhotonID(threadIdx);
    if (photonID >= gDrawArgument[0].InstanceCount)
    {
        return;
    }

    Photon p = gPhotonBuffer[photonID];
    float3 dPdx, dPdy;
    if (!orthogonalizeFrame(p.dPdx, p.dPdy, dPdx, dPdy))
    {
        p.color = 0;
    }
    p.dPdx = dPdx;
    p.dPdy = dPdy;
    gPhotonBuffer[photonID] = p;
}

[numthreads(PHOTON_COUNT_BLOCK_SIZE, PHOTON_COUNT_BLOCK_SIZE, 1)]
void CountTilePhoton(uint3 threadIdx : SV_DispatchThreadID)
{
    int photonID = threadIDToPhotonID(threadIdx);
    if (photonID >= gDrawArgument[0].InstanceCount)
    {
        return;
    }
    if (photonID == 0)
    {
        gIDCounter.Store(0, 0);
    }

    Photon photon = gPhotonBuffer[photonID];
    if (!checkPhoton(photon))
    {
        return;
    }

    int2 minTileID, maxTileID;
    GetPhotonScreenRange(photon, minTileID, maxTileID);
    for (int x = minTileID.x; x <= maxTileID.x; x++)
    {
        for (int y = minTileID.y; y <= maxTileID.y; y++)
        {
            int offset = getTileOffset(x, y);
            int count;
            InterlockedAdd(gTileInfo[offset].count, 1, count);
        }
    }
}

[numthreads(GATHER_TILE_SIZE_X, GATHER_TILE_SIZE_Y, 1)]
void AllocateMemory(uint3 threadIdx : SV_DispatchThreadID)
{
    int2 tileID = threadIdx.xy;
    if (any(tileID >= tileDim))
    {
        return;
    }

    int offset = getTileOffset(tileID.x, tileID.y);
    uint count = gTileInfo[offset].count;
    uint address = 0;
    gIDCounter.InterlockedAdd(0, count, address);
    gTileInfo[offset].address = address;
    gTileInfo[offset].count = 0;
}

[numthreads(PHOTON_COUNT_BLOCK_SIZE, PHOTON_COUNT_BLOCK_SIZE, 1)]
void StoreTilePhoton(uint3 threadIdx : SV_DispatchThreadID)
{
    int photonID = threadIDToPhotonID(threadIdx);
    if (photonID >= gDrawArgument[0].InstanceCount)
    {
        return;
    }

    Photon photon = gPhotonBuffer[photonID];
    if (!checkPhoton(photon))
    {
        return;
    }

    int2 minTileID, maxTileID;
    GetPhotonScreenRange(photon, minTileID, maxTileID);
    for (int x = minTileID.x; x <= maxTileID.x; x++)
    {
        for (int y = minTileID.y; y <= maxTileID.y; y++)
        {
            int offset = getTileOffset(x, y);
            int address = gTileInfo[offset].address;
            int idx = 0;
            InterlockedAdd(gTileInfo[offset].count, 1, idx);
            gIDBuffer.Store((address + idx) * 4, photonID);
        }
    }
}
