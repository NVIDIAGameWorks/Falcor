/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "Utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "Utils/Math/PackedFormats.h"
#else
import Utils.Math.PackedFormats;
#endif

BEGIN_NAMESPACE_FALCOR

// Uncomment this define to use uncompressed nodes for debugging purposes.
// Note that the compression is lossy and may add a slight bias and/or variance.
//#define USE_UNCOMPRESSED_NODES

static const float kInvalidCosConeAngle = -1.f;

/** Unpacked attributes shared between leaf and internal nodes.
*/
struct SharedNodeAttributes
{
    float3 origin;                  ///< Center of the node bounding box.
    float3 extent;                  ///< Extent of the node bounding box. The min/max corners are at origin +- extent.
    float flux = 0.f;               ///< Total emitted flux.
    float cosConeAngle = kInvalidCosConeAngle;  ///< Normal bounding cone cosine spread angle in [-1,1]. If cosConeAngle == kInvalidCosConeAngle, the cone should not be used.
    float3 coneDirection = { 0.f, 0.f, 0.f };   ///< Normal bounding cone direction.

#ifdef HOST_CODE
    void getAABB(float3& aabbMin, float3& aabbMax)
#else
    void getAABB(out float3 aabbMin, out float3 aabbMax)
#endif
    {
        aabbMin = origin - extent;
        aabbMax = origin + extent;
    }

    SETTER_DECL void setAABB(float3 aabbMin, float3 aabbMax)
    {
        origin = (aabbMax + aabbMin) * 0.5f;
        extent = (aabbMax - aabbMin) * 0.5f;
    }
};

/** Unpacked internal node.
*/
struct InternalNode
{
    SharedNodeAttributes attribs;   ///< Shared node attributes (origin/extent, bounding cone, flux).
    uint rightChildIdx;             ///< Index of the right child node. The left child is stored immediately after the current node.
};

/** Unpacked leaf node.
*/
struct LeafNode
{
    SharedNodeAttributes attribs;   ///< Shared node attributes (origin/extent, bounding cone, flux).
    uint triangleCount = 0;         ///< Number of triangles stored in leaf node.
    uint triangleOffset = 0;        ///< Offset into the triangle index list where this node's triangle indices are stored.
};

/** Light BVH node packed into 32B.

    There are two types of nodes: internal and leaf nodes.
    They share all node attributes, but differ in what data they reference.

    This struct has methods for packing/unpacking nodes to/from the packed format.
    The code is a bit cluttered because it is made C++ compatible for use on the
    host when building the BVH.
*/
struct PackedNode
{
#ifdef USE_UNCOMPRESSED_NODES
    uint4 data[3];
#else
    uint4 data[2];
#endif

    // The MSB bit of the first dword denotes the node type: 0=internal, 1=leaf node.
    // The remaining bits store the triangle count/offset for leaf nodes, and the right node index for internal nodes.
    static const uint kTriangleCountBits = 4;
    static const uint kTriangleOffsetBits = 31 - kTriangleCountBits;

    bool isLeaf() CONST_FUNCTION
    {
        return (data[0].x >> 31) != 0;
    }

    /** Unpack an internal node. The result is only valid if isLeaf() == false.
    */
    InternalNode getInternalNode() CONST_FUNCTION
    {
        InternalNode node;
        node.rightChildIdx = data[0].x; // Note MSB is 0 for internal nodes, so the value is safe to use directly.
        node.attribs = getNodeAttributes();
        return node;
    }

    /** Unpack an leaf node. The result is only valid if isLeaf() == true.
    */
    LeafNode getLeafNode() CONST_FUNCTION
    {
        LeafNode node;
        node.triangleCount = (data[0].x >> kTriangleOffsetBits) & ((1 << kTriangleCountBits) - 1);
        node.triangleOffset = data[0].x & ((1 << kTriangleOffsetBits) - 1);
        node.attribs = getNodeAttributes();
        return node;
    }

    /** Unpacks the shared node attributes.
    */
    SharedNodeAttributes getNodeAttributes() CONST_FUNCTION
    {
        SharedNodeAttributes attribs;
        attribs.origin = { asfloat(data[0].y), asfloat(data[0].z), asfloat(data[0].w) }; // = asfloat(data[0].yzw)
#ifdef USE_UNCOMPRESSED_NODES
        attribs.extent = { asfloat(data[1].x), asfloat(data[1].y), asfloat(data[1].z) };
        attribs.flux = asfloat(data[1].w);
        attribs.coneDirection = { asfloat(data[2].x), asfloat(data[2].y), asfloat(data[2].z) };
        attribs.cosConeAngle = asfloat(data[2].w);
#else
        uint4 packed = { data[1].x & 0xffff, data[1].x >> 16, data[1].y & 0xffff, data[1].y >> 16 };
        attribs.extent = { f16tof32(packed.x), f16tof32(packed.y), f16tof32(packed.z) }; // = f16tof32(packed.xyz)
        attribs.cosConeAngle = packed.w * (1.f / 32767.f) - 1.f;
        attribs.coneDirection = decodeNormal2x16(data[1].z);
        attribs.flux = asfloat(data[1].w);
#endif
        return attribs;
    }

    /** Packs an internal node.
    */
    SETTER_DECL void setInternalNode(const InternalNode node)
    {
        data[0].x = node.rightChildIdx;
        setNodeAttributes(node.attribs);
    }

    /** Packs a leaf node.
    */
    SETTER_DECL void setLeafNode(const LeafNode node)
    {
        data[0].x = (1 << 31) | (node.triangleCount << kTriangleOffsetBits) | node.triangleOffset;
        setNodeAttributes(node.attribs);
    }

    /** Packs the shared node attributes.
    */
    SETTER_DECL void setNodeAttributes(const SharedNodeAttributes attribs)
    {
        data[0].y = asuint(attribs.origin.x);
        data[0].z = asuint(attribs.origin.y);
        data[0].w = asuint(attribs.origin.z);
#ifdef USE_UNCOMPRESSED_NODES
        data[1].x = asuint(attribs.extent.x);
        data[1].y = asuint(attribs.extent.y);
        data[1].z = asuint(attribs.extent.z);
        data[1].w = asuint(attribs.flux);
        data[2].x = asuint(attribs.coneDirection.x);
        data[2].y = asuint(attribs.coneDirection.y);
        data[2].z = asuint(attribs.coneDirection.z);
        data[2].w = asuint(attribs.cosConeAngle);
#else
        uint packedAngle = (uint)((attribs.cosConeAngle + 1.f) * 32767.f); // Note: Round towards -inf so that the quantized cone angle is always equal or larger.
        data[1].x = f32tof16(attribs.extent.x) | (f32tof16(attribs.extent.y) << 16); // TODO: Round towards + inf to get conservative bounding boxes.
        data[1].y = f32tof16(attribs.extent.z) | (packedAngle << 16);
        data[1].z = encodeNormal2x16(attribs.coneDirection);
        data[1].w = asuint(attribs.flux);
#endif
    }
};

END_NAMESPACE_FALCOR
