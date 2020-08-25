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
#include "Utils/Math/MathConstants.slangh"

import Experimental.Scene.Lights.LightBVH;
import Experimental.Scene.Lights.LightCollection;

cbuffer CB
{
    LightCollection         gLights;            ///< The light sources.
    RWLightBVH              gLightBVH;          ///< The BVH around the light sources.
    StructuredBuffer<uint>  gNodeIndices;       ///< Buffer containing the indices of all the nodes. The indices are sorted by depths and laid out contiguously in memory; the indices for all the leaves are placed in the lowest level.
    uint                    gFirstNodeOffset;   ///< The offset of the first node index in 'gNodeIndices' to be processed.
    uint                    gNodeCount;         ///< Amount of nodes that need to be processed.
};

/** Compute shader for refitting the leaf nodes.
    The code assumes a leaf stores an indexed list of emissive triangles.
*/
[numthreads(256, 1, 1)]
void updateLeafNodes(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gNodeCount) return;

    uint nodeIndex = gNodeIndices[gFirstNodeOffset + DTid.x];
    LeafNode node = gLightBVH.getLeafNode(nodeIndex);

    // Update the node bounding box.
    float3 aabbMin = float3(FLT_MAX);
    float3 aabbMax = float3(-FLT_MAX);
    float3 normalsSum = float3(0.0f);

    for (uint i = 0; i < node.triangleCount; i++)
    {
        uint triangleIndex = gLightBVH.getNodeTriangleIndex(node, i);
        EmissiveTriangle tri = gLights.getTriangle(triangleIndex);
        for (uint vertexIndex = 0u; vertexIndex < 3u; ++vertexIndex)
        {
            aabbMin = min(aabbMin, tri.posW[vertexIndex]);
            aabbMax = max(aabbMax, tri.posW[vertexIndex]);
        }
        normalsSum += tri.normal;
    }

    node.attribs.setAABB(aabbMin, aabbMax);

    // Update the normal bounding cone.
    float coneDirectionLength = length(normalsSum);
    float3 coneDirection = normalsSum / coneDirectionLength;
    float cosConeAngle = kInvalidCosConeAngle;

    if (coneDirectionLength >= FLT_MIN)
    {
        cosConeAngle = 1.0f;
        for (uint i = 0; i < node.triangleCount; i++)
        {
            uint triangleIndex = gLightBVH.getNodeTriangleIndex(node, i);
            float3 normal = gLights.getTriangle(triangleIndex).normal;
            float cosDiffAngle = dot(coneDirection, normal);
            cosConeAngle = min(cosConeAngle, cosDiffAngle);
        }
        cosConeAngle = max(cosConeAngle, -1.f); // Guard against numerical errors
    }

    node.attribs.cosConeAngle = cosConeAngle;
    node.attribs.coneDirection = coneDirection;

    // Store the updated node.
    gLightBVH.setLeafNode(nodeIndex, node);
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

    uint nodeIndex = gNodeIndices[gFirstNodeOffset + DTid.x];
    InternalNode node = gLightBVH.getInternalNode(nodeIndex);

    uint leftChildIndex = nodeIndex + 1; // Left child is stored immediately after.
    uint rightChildIndex = node.rightChildIdx;

    const SharedNodeAttributes leftNode = gLightBVH.nodes[leftChildIndex].getNodeAttributes();
    const SharedNodeAttributes rightNode = gLightBVH.nodes[rightChildIndex].getNodeAttributes();

    // Update the node bounding box.
    float3 leftAabbMin, leftAabbMax;
    float3 rightAabbMin, rightAabbMax;
    leftNode.getAABB(leftAabbMin, leftAabbMax);
    rightNode.getAABB(rightAabbMin, rightAabbMax);

    float3 aabbMin = min(leftAabbMin, rightAabbMin);
    float3 aabbMax = max(leftAabbMax, rightAabbMax);

    node.attribs.setAABB(aabbMin, aabbMax);

    // Update the normal bounding cone.
    float3 coneDirectionSum = leftNode.coneDirection + rightNode.coneDirection;
    float coneDirectionLength = length(coneDirectionSum);
    float3 coneDirection = coneDirectionSum / coneDirectionLength;
    float cosConeAngle = kInvalidCosConeAngle;

    if (coneDirectionLength >= FLT_MIN &&
        leftNode.cosConeAngle != kInvalidCosConeAngle && rightNode.cosConeAngle != kInvalidCosConeAngle)
    {
        // This code rotates (cosLeftDiffAngle, sinLeftDiffAngle) counterclockwise by the left child's
        // cone spread angle, and similarly for the right child's cone.
        float cosLeftDiffAngle = dot(coneDirection, leftNode.coneDirection);
        float sinLeftDiffAngle = sinFromCos(cosLeftDiffAngle);

        float cosRightDiffAngle = dot(coneDirection, rightNode.coneDirection);
        float sinRightDiffAngle = sinFromCos(cosRightDiffAngle);

        float sinLeftConeAngle = sinFromCos(leftNode.cosConeAngle);
        float sinRightConeAngle = sinFromCos(rightNode.cosConeAngle);

        float sinLeftTotalAngle = sinLeftConeAngle * cosLeftDiffAngle + sinLeftDiffAngle * leftNode.cosConeAngle;
        float sinRightTotalAngle = sinRightConeAngle * cosRightDiffAngle + sinRightDiffAngle * rightNode.cosConeAngle;

        // If neither sum of angles is greater than pi, compute the new cosConeAngle.
        // Otherwise, deactivate the orientation cone as useless since it would represent the whole sphere.
        if (sinLeftTotalAngle > 0.0f && sinRightTotalAngle > 0.0f)
        {
            const float cosLeftTotalAngle = leftNode.cosConeAngle * cosLeftDiffAngle - sinLeftConeAngle * sinLeftDiffAngle;
            const float cosRightTotalAngle = rightNode.cosConeAngle * cosRightDiffAngle - sinRightConeAngle * sinRightDiffAngle;

            cosConeAngle = min(cosLeftTotalAngle, cosRightTotalAngle);
            cosConeAngle = max(cosConeAngle, -1.f); // Guard against numerical errors
        }
    }

    node.attribs.cosConeAngle = cosConeAngle;
    node.attribs.coneDirection = coneDirection;

    // Store the updated node.
    gLightBVH.setInternalNode(nodeIndex, node);
}
