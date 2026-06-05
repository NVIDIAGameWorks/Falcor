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
    float4x4 viewProjMat;

    int2 taskDim;
    int2 screenDim;

    float normalThreshold;
    float distanceThreshold;
    float planarThreshold;
    float pixelLuminanceThreshold;

    float minPhotonPixelSize;
    float trimDirectionThreshold;
    uint enableMedianFilter;
    uint removeIsolatedPhoton;

    int minNeighbourCount;
};

//struct DrawArguments
//{
//    uint IndexCountPerInstance;
//    uint InstanceCount;
//    uint StartIndexLocation;
//    int BaseVertexLocation;
//    uint StartInstanceLocation;
//};

StructuredBuffer<Photon> gSrcPhotonBuffer;
RWStructuredBuffer<Photon> gDstPhotonBuffer;
//RWStructuredBuffer<DrawArguments> gDrawArgument;
RWStructuredBuffer<RayArgument> gRayArgument;
StructuredBuffer<PixelInfo> gRayTask;
Texture2D gDepthTex;

bool checkPixelNeighbour(uint2 pixelCoord0, Photon photon0, uint2 offset)
{
    uint2 pixelCoord1 = min(taskDim - 1, max(uint2(0, 0), pixelCoord0 + offset));
    uint pixelIdx1 = pixelCoord1.y * taskDim.x + pixelCoord1.x;

    PixelInfo task1 = gRayTask[pixelIdx1];
    bool isContinue = task1.photonIdx != -1;
    float3 dColor = photon0.color;
    Photon photon1;
    if (isContinue)
    {
        photon1 = gSrcPhotonBuffer[task1.photonIdx];
        isContinue &= isPhotonAdjecent(photon0, photon1, normalThreshold, distanceThreshold, planarThreshold);
    }
    return isContinue;
}

bool getPhoton(uint2 pixelCoord, inout Photon photon)
{
    uint2 pixelCoord1 = min(taskDim - 1, max(uint2(0, 0), pixelCoord));
    uint pixelIdx1 = pixelCoord1.y * taskDim.x + pixelCoord1.x;
    PixelInfo task1 = gRayTask[pixelIdx1];
    if (task1.photonIdx == -1)
    {
        return false;
    }
    photon = gSrcPhotonBuffer[task1.photonIdx];
    return true;
}

void getTrimLength(float3 P1, float3 D1, float3 P2, float3 D2, inout float l)
{
    D1 = normalize(D1);
    D2 = normalize(D2);
    float D1D2 = dot(D1, D2);
    if (abs(D1D2) > trimDirectionThreshold)
    {
        return;
    }
    //float3 P12 = P1 - P2;
    //float P12D1 = dot(P12, D1);
    //float P12D2 = dot(P12, D2);
    //float det = -1 + D1D2 * D1D2;
    //float t1 = P12D1 - P12D2 * D1D2;
    //l = min(l, abs(t1 / det));

    float3 P21 = P2 - P1;
    float proj = length(P21); // dot(P21, D1);
    l = min(l, abs(proj));
}

PixelInfo getRayTask(uint2 pixelCoord)
{
    uint2 pixelCoord1 = min(taskDim - 1, max(uint2(0, 0), pixelCoord));
    uint pixelIdx1 = pixelCoord1.y * taskDim.x + pixelCoord1.x;
    return gRayTask[pixelIdx1];
}

float3 medianFilter(uint2 pixelCoord0)
{
    uint2 dir[9] =
    {
        uint2(-1, -1),
        uint2(-1, 0),
        uint2(-1, 1),
        uint2(0, -1),
        uint2(0, 0),
        uint2(0, 1),
        uint2(1, -1),
        uint2(1, 0),
        uint2(1, 1),
    };

    half2 luminance[9];
    for (int i = 0; i < 8; i++)
    {
        //bool isContinue = checkPixelNeighbour(threadIdx.xy, photon0, dir[i]);
        //uint2 pixelCoord1 = min(taskDim - 1, max(uint2(0, 0), pixelCoord0 + dir[i]));
        //uint pixelIdx1 = pixelCoord1.y * taskDim.x + pixelCoord1.x;

        PixelInfo task1 = getRayTask(pixelCoord0 + dir[i]);
        if (task1.photonIdx != -1)
        {
            Photon photon = gSrcPhotonBuffer[task1.photonIdx];
            luminance[i].x = dot(photon.color, float3(0.299, 0.587, 0.114));
        }
        else
        {
            luminance[i].x = 0;
        }
        luminance[i].y = i;
    }

    for (int i = 0; i < 9; i++)
    {
        for (int j = 0; j < i; j++)
        {
            if (luminance[j].x > luminance[j + 1].x)
            {
                half2 a = luminance[j + 1];
                luminance[j + 1] = luminance[j];
                luminance[j] = a;
            }
        }
    }

    int idx = luminance[4].y;
    PixelInfo task1 = getRayTask(pixelCoord0 + dir[idx]);
    if (task1.photonIdx != -1)
    {
        Photon photon = gSrcPhotonBuffer[task1.photonIdx];
        return photon.color;
    }
    return 0;
}

bool isPhotonContinue(Photon photon0, Photon photon1)
{
    bool isContinue = true;
    float3 normal0 = normalize(cross(photon0.dPdx, photon0.dPdy));
    float3 normal1 = normalize(cross(photon1.dPdx, photon1.dPdy));
    isContinue &= dot(normal0, normal1) > normalThreshold;

    float distanceFactor = 0.5 * (length(photon0.dPdx) + length(photon0.dPdx));
    isContinue &= length(photon0.posW - photon1.posW) / distanceFactor < distanceThreshold;

    isContinue &= dot(normal0, photon0.posW - photon1.posW) < planarThreshold;
    return isContinue;
}

[numthreads(16, 16, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex, uint3 threadIdx : SV_DispatchThreadID)
{

    //uint length, stride;
    //gSrcPhotonBuffer.GetDimensions(length, stride);
    uint2 pixelCoord0 = threadIdx.xy;
    uint rayIdx = threadIdx.y * taskDim.x + threadIdx.x;
    PixelInfo task0 = gRayTask[rayIdx];
    int idx0 = task0.photonIdx;
    if (idx0 == -1)
    {
        return;
    }

    Photon photon0 = gSrcPhotonBuffer[idx0];

    uint2 dir[8] =
    {
        uint2(-1, -1),
        uint2(-1, 0),
        uint2(-1, 1),
        uint2(0, -1),
        //uint2(0,0),
        uint2(0, 1),
        uint2(1, -1),
        uint2(1, 0),
        uint2(1, 1),
    };
    uint continueFlag = 0;
    float3 color = photon0.color; // / 9;
    int continueCount = 0;
    for (int i = 0; i < 8; i++)
    {
        //bool isContinue = checkPixelNeighbour(threadIdx.xy, photon0, dir[i]);
        uint2 pixelCoord1 = min(taskDim - 1, max(uint2(0, 0), pixelCoord0 + dir[i]));
        uint pixelIdx1 = pixelCoord1.y * taskDim.x + pixelCoord1.x;

        PixelInfo task1 = gRayTask[pixelIdx1];
        bool isContinue = task1.photonIdx != -1;
        Photon photon1;
        if (isContinue)
        {
            photon1 = gSrcPhotonBuffer[task1.photonIdx];
            isContinue &= isPhotonContinue(photon0, photon1);
        }
        if (isContinue)
        {
            //color += photon1.color / 9;
            continueCount++;
            continueFlag |= (1 << i);
        }
    }
    if (removeIsolatedPhoton && continueCount < minNeighbourCount)
    {
        //float w = saturate((continueCount - 0.0) / (8.0 - 0.0));
        //color *= lerp(0.0,1,w);
        photon0.color = 0;
    }
    //gSrcPhotonBuffer[idx0].color = color;

    //Photon photon1;
    //float trimmedLength = length(photon0.dPdx);
    //if (getPhoton(pixelCoord0 + uint2(1, 0), photon1) && (continueFlag & (1 << 6)))
    //{
    //    getTrimLength(photon0.posW, photon0.dPdx, photon1.posW, photon1.dPdx, trimmedLength);
    //}
    //if (getPhoton(pixelCoord0 + uint2(-1, 0), photon1) && (continueFlag & (1 << 1)))
    //{
    //    getTrimLength(photon0.posW, photon0.dPdx, photon1.posW, photon1.dPdx, trimmedLength);
    //}
    //photon0.dPdx = photon0.dPdx * trimmedLength / length(photon0.dPdx);

    //trimmedLength = length(photon0.dPdy);
    //if (getPhoton(pixelCoord0 + uint2(0, 1), photon1) && (continueFlag & (1 << 4)))
    //{
    //    getTrimLength(photon0.posW, photon0.dPdy, photon1.posW, photon1.dPdy, trimmedLength);
    //}
    //if (getPhoton(pixelCoord0 + uint2(0, -1), photon1) && (continueFlag & (1 << 3)))
    //{
    //    getTrimLength(photon0.posW, photon0.dPdy, photon1.posW, photon1.dPdy, trimmedLength);
    //}
    //photon0.dPdy = photon0.dPdy * trimmedLength / length(photon0.dPdy);

    //uint fourNeighbourFlag = (1 << 1) | (1 << 3) | (1 << 4) | (1 << 6);
    //if ((continueFlag & fourNeighbourFlag) == 0)
    //{
    //    photon0.color = 0;
    //}

    //float width = length(photon0.dPdx);
    //float height = length(photon0.dPdy);
    //float area0 = width * height;
    //float area = length(cross(photon0.dPdx, photon0.dPdy));
    //float ratio = area / area0;
    //if (ratio < trimDirectionThreshold)
    //{
    //    photon0.color = 0;
    //}

    if (enableMedianFilter)
    {
        photon0.color = medianFilter(pixelCoord0);
    }
    gDstPhotonBuffer[idx0] = photon0;
}
