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
__exported import LightBVHStaticParams;

struct LightBVH
{
    ByteAddressBuffer nodes;                        ///< Buffer containing all the nodes from the BVH, with the root node located at offset 0.
    StructuredBuffer<uint2> triangleBitmasks;       ///< Buffer containing for each emissive triangle, a bit mask of the traversal to follow in order to reach that triangle. Size: lights.triangleCount * sizeof(uint64_t).

    // IMPORTANT: these functions must be kept in sync with the definitions of
    // InternalNode and LeafNode in Experimental/Scene/Lights/LightBVH.h.
    uint getNodeType(uint offset)
    {
        return nodes.Load(offset + kNodeTypeOffset);
    }

    float getNodeCosConeAngle(uint offset)
    {
        return asfloat(nodes.Load(offset + kNodeCosConeAngleOffset));
    }

    float3 getNodeConeDirection(uint offset)
    {
        return asfloat(nodes.Load3(offset + kNodeConeDirectionOffset));
    }

    float getNodeFlux(uint offset)
    {
        return asfloat(nodes.Load(offset + kNodeFluxOffset));
    }

    void getNodeAABB(uint offset, out float3 aabbMin, out float3 aabbMax)
    {
        aabbMin = asfloat(nodes.Load3(offset + kNodeAABBMinOffset));
        aabbMax = asfloat(nodes.Load3(offset + kNodeAABBMaxOffset));
    }

    uint getNodeLeftByteOffset(uint offset)
    {
        return nodes.Load(offset + kNodeLeftByteOffsetOffset);
    }

    uint getNodeTriangleCount(uint offset)
    {
        return nodes.Load(offset + kNodeTriangleCountOffset);
    }

    uint getNodeRightByteOffset(uint offset)
    {
        return nodes.Load(offset + kNodeRightByteOffsetOffset);
    }

    uint getNodeTriangleIndex(uint offset, uint tri)
    {
        return nodes.Load(offset + kNodeTriangleIndicesOffset + kNodeTriangleIndexByteSize * tri);
    }
};

struct RWLightBVH
{
    RWByteAddressBuffer nodes;                      ///< Buffer containing all the nodes from the BVH, with the root node located at offset 0.
    StructuredBuffer<uint2> triangleBitmasks;       ///< Buffer containing for each emissive triangle, a bit mask of the traversal to follow in order to reach that triangle. Size: lights.triangleCount * sizeof(uint64_t).

    // IMPORTANT: these functions must be kept in sync with the definitions of
    // InternalNode and LeafNode in Experimental/Scene/Lights/LightBVH.h.
    uint getNodeType(uint offset)
    {
        return nodes.Load(offset + kNodeTypeOffset);
    }

    float getNodeCosConeAngle(uint offset)
    {
        return asfloat(nodes.Load(offset + kNodeCosConeAngleOffset));
    }

    void setNodeCosConeAngle(uint offset, float cosCone)
    {
        nodes.Store(offset + kNodeCosConeAngleOffset, asuint(cosCone));
    }

    float3 getNodeConeDirection(uint offset)
    {
        return asfloat(nodes.Load3(offset + kNodeConeDirectionOffset));
    }

    void setNodeConeDirection(uint offset, float3 dir)
    {
        nodes.Store3(offset + kNodeConeDirectionOffset, asuint(dir));
    }

    float getNodeFlux(uint offset)
    {
        return asfloat(nodes.Load(offset + kNodeFluxOffset));
    }

    void getNodeAABB(uint offset, out float3 aabbMin, out float3 aabbMax)
    {
        aabbMin = asfloat(nodes.Load3(offset + kNodeAABBMinOffset));
        aabbMax = asfloat(nodes.Load3(offset + kNodeAABBMaxOffset));
    }

    void setNodeAABB(uint offset, float3 aabbMin, float3 aabbMax)
    {
        nodes.Store3(offset + kNodeAABBMinOffset, asuint(aabbMin));
        nodes.Store3(offset + kNodeAABBMaxOffset, asuint(aabbMax));
    }

    uint getNodeLeftByteOffset(uint offset)
    {
        return nodes.Load(offset + kNodeLeftByteOffsetOffset);
    }

    uint getNodeTriangleCount(uint offset)
    {
        return nodes.Load(offset + kNodeTriangleCountOffset);
    }

    uint getNodeRightByteOffset(uint offset)
    {
        return nodes.Load(offset + kNodeRightByteOffsetOffset);
    }

    uint getNodeTriangleIndex(uint offset, uint tri)
    {
        return nodes.Load(offset + kNodeTriangleIndicesOffset + kNodeTriangleIndexByteSize * tri);
    }
};
