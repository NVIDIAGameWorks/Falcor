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
#include "Utils/Math/MathConstants.slangh"

import Experimental.Scene.Lights.LightBVH;
import Experimental.Scene.Lights.LightCollection;

cbuffer CB
{
    LightCollection     gLights;              ///< The light sources.
    RWLightBVH          gLightBVH;            ///< The BVH around the light sources.
    uint                gFirstNodeIndex;      ///< The index of the first node to be processed.
    uint                gNodeCount;           ///< Amount of nodes that need to be processed.
};

Buffer<uint> gNodeOffsets; ///< Buffer containing the offsets to all the nodes. The offsets are sorted by depths and laid out contiguously in memory; the offsets for all the leaves are placed in the lowest level.

/** Compute shader for refitting the leaf nodes.
    The code assumes a leaf stores an indexed list of emissive triangles.
*/
[numthreads(256, 1, 1)]
void updateLeafNodes(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gNodeCount) return;

    const uint nodeOffset = gNodeOffsets[gFirstNodeIndex + DTid.x];
    const uint triangleCount = gLightBVH.getNodeTriangleCount(nodeOffset);

    float3 aabbMin = float3(FLT_MAX);
    float3 aabbMax = float3(-FLT_MAX);
    float3 normalsSum = float3(0.0f);
    for (uint triangleIndicesIndex = 0u; triangleIndicesIndex < triangleCount; ++triangleIndicesIndex)
    {
        const uint triangleIndex = gLightBVH.getNodeTriangleIndex(nodeOffset, triangleIndicesIndex);
        for (uint vertexIndex = 0u; vertexIndex < 3u; ++vertexIndex)
        {
            const float3 p = gLights.getVtxPos(triangleIndex * 3 + vertexIndex);
            aabbMin = min(aabbMin, p);
            aabbMax = max(aabbMax, p);
        }
        normalsSum += gLights.getTriangleData(triangleIndex).normal;
    }

    gLightBVH.setNodeAABB(nodeOffset, aabbMin, aabbMax);

    const float coneDirectionLength = length(normalsSum);
    const float3 coneDirection = normalsSum / coneDirectionLength;
    float cosConeAngle = kInvalidCosConeAngle;

    if (coneDirectionLength >= FLT_MIN)
    {
        cosConeAngle = 1.0f;
        for (uint triangleIndicesIndex = 0u; triangleIndicesIndex < triangleCount; ++triangleIndicesIndex)
        {
            const uint triangleIndex = gLightBVH.getNodeTriangleIndex(nodeOffset, triangleIndicesIndex);
            const float3 normal = gLights.getTriangleData(triangleIndex).normal;
            const float cosDiffAngle = dot(coneDirection, normal);
            cosConeAngle = min(cosConeAngle, cosDiffAngle);
        }
    }

    gLightBVH.setNodeCosConeAngle(nodeOffset, cosConeAngle);
    gLightBVH.setNodeConeDirection(nodeOffset, coneDirection);
}

// Returns sin(a) based on cos(a) for a in [0,pi].
float sinFromCos(float cosAngle)
{
    return sqrt(max(0.0f, 1.0f - cosAngle * cosAngle));
}

/** Compute shader for refitting the internal nodes.
    This should be executed after updateLeafNodes().
*/
[numthreads(256, 1, 1)]
void updateInternalNodes(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gNodeCount) return;

    const uint nodeOffset = gNodeOffsets[gFirstNodeIndex + DTid.x];
    const uint leftChildOffset = gLightBVH.getNodeLeftByteOffset(nodeOffset);
    const uint rightChildOffset = gLightBVH.getNodeRightByteOffset(nodeOffset);

    float3 leftAabbMin, leftAabbMax;
    gLightBVH.getNodeAABB(leftChildOffset, leftAabbMin, leftAabbMax);
    float3 rightAabbMin, rightAabbMax;
    gLightBVH.getNodeAABB(rightChildOffset, rightAabbMin, rightAabbMax);

    const float3 aabbMin = min(leftAabbMin, rightAabbMin);
    const float3 aabbMax = max(leftAabbMax, rightAabbMax);
    gLightBVH.setNodeAABB(nodeOffset, aabbMin, aabbMax);

    float3 coneDirectionSum = gLightBVH.getNodeConeDirection(leftChildOffset) + gLightBVH.getNodeConeDirection(rightChildOffset);
    const float coneDirectionLength = length(coneDirectionSum);
    const float3 coneDirection = coneDirectionSum / coneDirectionLength;
    const float cosLeftConeAngle = gLightBVH.getNodeCosConeAngle(leftChildOffset);
    const float cosRightConeAngle = gLightBVH.getNodeCosConeAngle(rightChildOffset);
    float cosConeAngle = kInvalidCosConeAngle;

    if (coneDirectionLength >= FLT_MIN &&
        cosLeftConeAngle != kInvalidCosConeAngle && cosRightConeAngle != kInvalidCosConeAngle)
    {
        // This code rotates (cosLeftDiffAngle, sinLeftDiffAngle) counterclockwise by the left child's
        // cone spread angle, and similarly for the right child's cone.
        const float cosLeftDiffAngle = dot(coneDirection, gLightBVH.getNodeConeDirection(leftChildOffset));
        const float sinLeftDiffAngle = sinFromCos(cosLeftDiffAngle);

        const float cosRightDiffAngle = dot(coneDirection, gLightBVH.getNodeConeDirection(rightChildOffset));
        const float sinRightDiffAngle = sinFromCos(cosRightDiffAngle);

        const float sinLeftConeAngle = sinFromCos(cosLeftConeAngle);
        const float sinRightConeAngle = sinFromCos(cosRightConeAngle);

        const float sinLeftTotalAngle = sinLeftConeAngle * cosLeftDiffAngle + sinLeftDiffAngle * cosLeftConeAngle;
        const float sinRightTotalAngle = sinRightConeAngle * cosRightDiffAngle + sinRightDiffAngle * cosRightConeAngle;

        // If neither sum of angles is greater than pi, compute the new cosConeAngle.
        // Otherwise, deactivate the orientation cone as useless since it would represent the whole sphere.
        if (sinLeftTotalAngle > 0.0f && sinRightTotalAngle > 0.0f)
        {
            const float cosLeftTotalAngle = cosLeftConeAngle * cosLeftDiffAngle - sinLeftConeAngle * sinLeftDiffAngle;
            const float cosRightTotalAngle = cosRightConeAngle * cosRightDiffAngle - sinRightConeAngle * sinRightDiffAngle;

            cosConeAngle = min(cosLeftTotalAngle, cosRightTotalAngle);
        }
    }

    gLightBVH.setNodeCosConeAngle(nodeOffset, cosConeAngle);
    gLightBVH.setNodeConeDirection(nodeOffset, coneDirection);
}
