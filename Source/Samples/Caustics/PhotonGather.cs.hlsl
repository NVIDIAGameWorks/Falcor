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
    float4x4 gInvViewProjMat;
    int2 screenDim;
    int2 tileDim;
    float gSplatSize;
    float gDepthRadius;
    int gShowTileCount;
    int gTileCountScale;
    float gKernelPower;
    int causticsMapResRatio;
};

//struct DrawArguments
//{
//    uint IndexCountPerInstance;
//    uint InstanceCount;
//    uint StartIndexLocation;
//    int BaseVertexLocation;
//    uint StartInstanceLocation;
//};

StructuredBuffer<Photon> gPhotonBuffer;
StructuredBuffer<IDBlock> gTileInfo;
ByteAddressBuffer gIDBuffer;

Texture2D gDepthTex;
Texture2D gNormalTex;
RWTexture2D gPhotonTex;

#define PHOTON_CACHE_SIZE 64

groupshared Photon photonList[PHOTON_CACHE_SIZE];
groupshared float3 normalList[PHOTON_CACHE_SIZE];
groupshared int photonCount;
groupshared int beginAddress;

#define NUM_GROUP_PER_TILE 2
#define BLOCK_SIZE_X GATHER_TILE_SIZE_X
#define BLOCK_SIZE_Y (GATHER_TILE_SIZE_Y/NUM_GROUP_PER_TILE)

int getTileOffset(int x, int y)
{
    return tileDim.x * y + x;
}

float getLightFactor(float3 pos, float3 photonPos, float3 dPdx, float3 dPdy, float3 normal)
{
    float3 dPos = pos - photonPos;
    float3 localCoord = float3(dot(dPdx, dPos), dot(dPdy, dPos), dot(normal, dPos));
    float r = length(localCoord);
    return pow(saturate(smoothKernel(r)), gKernelPower);
}

[numthreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)]
void main(uint3 groupID : SV_GroupID, uint3 groupThreadID : SV_GroupThreadID, uint3 threadIdx : SV_DispatchThreadID)
{
    uint2 tileID = groupID.xy;
    uint2 pixelTileIndex = groupThreadID.xy; // threadIdx.xy - tileID * GATHER_TILE_SIZE;
    uint2 pixelLocation0 = tileID * uint2(GATHER_TILE_SIZE_X, GATHER_TILE_SIZE_Y) + pixelTileIndex;

    float3 worldPnt[NUM_GROUP_PER_TILE];
    [unroll]
    for (int i = 0; i < NUM_GROUP_PER_TILE; i++)
    {
        uint2 pixelLocation = pixelLocation0 + uint2(0, i * BLOCK_SIZE_Y);
        float depth = gDepthTex.Load(int3(pixelLocation * causticsMapResRatio, 0)).r;
        //float3 normal = gNormalTex.Load(int3(pixelLocation* causticsMapResRatio,0)).rgb;
        float2 uv = pixelLocation / float2(screenDim);
        float4 ndc = float4(uv * float2(2, -2) + float2(-1, 1), depth, 1);
        float4 worldPnt0 = mul(ndc, gInvViewProjMat);
        worldPnt[i] = worldPnt0.xyz / worldPnt0.w;
    }

    if (all(groupThreadID == uint3(0, 0, 0)))
    {
        int tileOffset = getTileOffset(tileID.x, tileID.y);
        beginAddress = gTileInfo[tileOffset].address;
        photonCount = gTileInfo[tileOffset].count;
    }
    GroupMemoryBarrierWithGroupSync();

    if (gShowTileCount)
    {
        float intensity = float(photonCount) / gTileCountScale;
        float4 color;
        if (intensity <= 1)
        {
            color = float4(intensity.xxx, 1);
        }
        else
        {
            color = float4(1, 0, 0, 1);
        }
        [unroll]
        for (int i = 0; i < NUM_GROUP_PER_TILE; i++)
        {
            uint2 pixelLocation = pixelLocation0 + uint2(0, i * BLOCK_SIZE_Y);
            if (all(pixelLocation < screenDim))
                gPhotonTex[pixelLocation] = color;
        }
        return;
    }

    float3 totalLight[NUM_GROUP_PER_TILE];
    [unroll]
    for (int i = 0; i < NUM_GROUP_PER_TILE; i++)
        totalLight[i] = 0;

    int threadGroupOffset = pixelTileIndex.y * BLOCK_SIZE_X + pixelTileIndex.x;
    for (int photonIdx = 0; photonIdx < photonCount; photonIdx += PHOTON_CACHE_SIZE)
    {
        int idOffset = photonIdx + threadGroupOffset;
        if (threadGroupOffset < PHOTON_CACHE_SIZE && idOffset < photonCount)
        {
            int id = gIDBuffer.Load((beginAddress + idOffset) * 4);
            Photon p = gPhotonBuffer[id];
            p.dPdx /= dot(p.dPdx, p.dPdx);
            p.dPdy /= dot(p.dPdy, p.dPdy);
            photonList[threadGroupOffset] = p;
            normalList[threadGroupOffset] = normalize(cross(p.dPdx, p.dPdy)) / (gDepthRadius * gSplatSize);
        }
        GroupMemoryBarrierWithGroupSync();

        for (int i = 0; i < min(PHOTON_CACHE_SIZE, photonCount - photonIdx); i++)
        {
            Photon p = photonList[i];
            float3 normal = normalList[i];
            for (int ithPixel = 0; ithPixel < NUM_GROUP_PER_TILE; ithPixel++)
            {
                float lightFactor = getLightFactor(worldPnt[ithPixel], p.posW, p.dPdx, p.dPdy, normal);
                totalLight[ithPixel] += lightFactor * p.color;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [unroll]
    for (int i = 0; i < NUM_GROUP_PER_TILE; i++)
    {
        uint2 pixelLocation = pixelLocation0 + uint2(0, i * BLOCK_SIZE_Y);
        if (all(pixelLocation < screenDim))
            gPhotonTex[pixelLocation] = float4(totalLight[i], photonCount);
    }
}
