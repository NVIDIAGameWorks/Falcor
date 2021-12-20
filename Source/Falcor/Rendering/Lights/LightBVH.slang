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
import Utils.Attributes;
__exported import LightBVHTypes;

/** Light BVH data structure.
*/
struct LightBVH
{
    [root] StructuredBuffer<PackedNode> nodes;      ///< Buffer containing all the nodes from the BVH, with the root node located at index 0.
    StructuredBuffer<uint> triangleIndices;         ///< Buffer containing the indices of all emissive triangles. Each leaf node refers to a contiguous range of indices.
    StructuredBuffer<uint2> triangleBitmasks;       ///< Buffer containing for each emissive triangle, a bit mask of the traversal to follow in order to reach that triangle. Size: lights.triangleCount * sizeof(uint64_t).

    bool isLeaf(uint nodeIndex)
    {
        return nodes[nodeIndex].isLeaf();
    }

    LeafNode getLeafNode(uint nodeIndex)
    {
        return nodes[nodeIndex].getLeafNode();
    }

    InternalNode getInternalNode(uint nodeIndex)
    {
        return nodes[nodeIndex].getInternalNode();
    }

    SharedNodeAttributes getNodeAttributes(uint nodeIndex)
    {
        return nodes[nodeIndex].getNodeAttributes();
    }

    uint getNodeTriangleIndex(const LeafNode node, uint index)
    {
        return triangleIndices[node.triangleOffset + index];
    }
};

/** Variant of the Light BVH data structure where the nodes are writable.
    This is used for BVH refit.
    TODO: Refactor to avoid code duplication using future Slang features.
*/
struct RWLightBVH
{
    [root] RWStructuredBuffer<PackedNode> nodes;    ///< Buffer containing all the nodes from the BVH, with the root node located at index 0.
    StructuredBuffer<uint> triangleIndices;         ///< Buffer containing the indices of all emissive triangles. Each leaf node refers to a contiguous range of indices.
    StructuredBuffer<uint2> triangleBitmasks;       ///< Buffer containing for each emissive triangle, a bit mask of the traversal to follow in order to reach that triangle. Size: lights.triangleCount * sizeof(uint64_t).

    bool isLeaf(uint nodeIndex)
    {
        return nodes[nodeIndex].isLeaf();
    }

    LeafNode getLeafNode(uint nodeIndex)
    {
        return nodes[nodeIndex].getLeafNode();
    }

    InternalNode getInternalNode(uint nodeIndex)
    {
        return nodes[nodeIndex].getInternalNode();
    }

    uint getNodeTriangleIndex(const LeafNode node, uint index)
    {
        return triangleIndices[node.triangleOffset + index];
    }

    // Setters

    void setLeafNode(uint nodeIndex, LeafNode node)
    {
        nodes[nodeIndex].setLeafNode(node);
    }

    void setInternalNode(uint nodeIndex, InternalNode node)
    {
        nodes[nodeIndex].setInternalNode(node);
    }
};
