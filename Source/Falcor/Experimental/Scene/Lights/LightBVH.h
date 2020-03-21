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
#pragma once
#include "LightBVHStaticParams.slang"
#include "LightCollection.h"

#include "Utils/AlignedAllocator.h"
#include "Utils/Math/BBox.h"
#include "Utils/Math/Vector.h"
#include "Utils/UI/Gui.h"

#include <limits>
#include <vector>

namespace Falcor
{
    class LightBVHBuilder;

    /** Utility class representing a light sampling BVH.

        This is binary BVH over all emissive triangles as described by Moreau and Clarberg,
        "Importance Sampling of Many Lights on the GPU", Ray Tracing Gems, Ch. 18, 2019.

        Before being used, the BVH needs to have been built using |LightBVHBuilder::build()|.
        The data can be both used on the CPU (using |traverseBVH()| or the getters like |getInternalNode()|)
        or on the GPU (1. declare a variable of type |LightBVH| in your shader, 2. call |setShaderData()|
        or |setShaderData()| to bind the BVH resources to that variable).

        TODO: Rename all things 'triangle' to 'light' as the BVH can be used for other light types.
    */
    class dlldecl LightBVH
    {
    public:
        using SharedPtr = std::shared_ptr<LightBVH>;
        using SharedConstPtr = std::shared_ptr<const LightBVH>;

        static const uint32_t kInvalidOffset = std::numeric_limits<uint32_t>::max();

        enum class NodeType : uint32_t
        {
            Internal = 0u,
            Leaf
        };

        // TODO: Store node bounds as center + extent instead of min/max? The center is needed for sampling, at least now.
        // TODO: Think about making the extent be scalar (bounding sphere) instead, saves bandwidth and makes bounding cone computations easier. But hard to compute and not as tight, so maybe not.
        // IMPORTANT: these structure definitions must be kept in sync with
        // the helper functions in LightBVHSampler.slang.
        struct InternalNode
        {
            NodeType nodeType = NodeType::Internal;
            glm::vec3 aabbMin;

            float luminousFlux = 0.f;
            glm::vec3 aabbMax;

            float cosConeAngle = kInvalidCosConeAngle; // If cosConeAngle == kInvalidCosConeAngle, the cone should not be used.
            glm::vec3 coneDirection = { 0.f, 0.f, 0.f };

            uint32_t leftNodeOffset = kInvalidOffset;
            uint32_t rightNodeOffset = kInvalidOffset;
        };

        struct LeafNode
        {
            NodeType nodeType = NodeType::Leaf;
            glm::vec3 aabbMin;

            float luminousFlux = 0.f;
            glm::vec3 aabbMax;

            float cosConeAngle = kInvalidCosConeAngle;  // If cosConeAngle == kInvalidCosConeAngle, the cone should not be used.
            glm::vec3 coneDirection = { 0.f, 0.f, 0.f };

            uint32_t triangleCount = 0;
            // The allocator allocates extra space after LeafNodes as needed to allow a larger
            // |triangleIndices| array.  (Thus, this must stay as the last member variable!)
            uint32_t triangleIndices[1];        ///< Array of global triangle indices.

            // Assignment is unsafe given how |triangleIndices| is allocated.
            LeafNode() = default;
            LeafNode(const LeafNode &) = delete;
            LeafNode &operator=(const LeafNode &) = delete;
        };

        struct NodeLocation
        {
            uint32_t byteOffset;
            uint32_t depth;

            NodeLocation() : byteOffset(0), depth(0) {}
            NodeLocation(uint32_t _byteOffset, uint32_t _depth) : byteOffset(_byteOffset), depth(_depth) {}
        };

        /** Function called on each node.
            \param[in] location The location of the current node, that is its byte offset from the beginning of the buffer, and its depth.
            \param[in] pInternalNode A pointer to the current node if the current node is an internal node, otherwise null
            \param[in] pLeaflNode A pointer to the current node if the current node is a leaf node, otherwise null
            \return True if the traversal should continue, false otherwise.
        */
        using TraversalEvalFunction = std::function<bool(const NodeLocation& location, const InternalNode* pInternalNode, const LeafNode* pLeafNode)>;

        /** Creates an empty LightBVH object. Use a LightBVHBuilder to build the BVH.
            \param[in] pLightCollection The light collection around which the BVH will be built.
        */
        static SharedPtr create(const LightCollection::SharedConstPtr& pLightCollection);

        /** Refit all the BVH nodes to the underlying geometry, without changing the hierarchy.
            The BVH needs to have been built before trying to refit it.
            \param[in] pRenderContext The render context.
        */
        void refit(RenderContext* pRenderContext);

        /** Return the type of the specified node.
            \return the type of the specified node.
        */
        NodeType getNodeType(const uint32_t nodeOffset) const;

        /** Return a typed InternalNode pointer to the specified node.
            \return an InternalNode pointer if the specified node is an internal node, nullptr otherwise.
        */
        const InternalNode* getInternalNode(const uint32_t nodeOffset) const;

        /** Return a typed LeafNode pointer to the specified node.
            \return an LeafNode pointer if the specified node is a leaf node, nullptr otherwise.
        */
        const LeafNode* getLeafNode(const uint32_t nodeOffset) const;

        /** Perform a depth-first traversal of the BVH and run a function on each node.
            \param[in] evalNode Function called on each node; see TraversalEvalFunction for more details.
            \param[in] rootNodeByteOffset The byte offset of the node to start traversing.
        */
        void traverseBVH(const TraversalEvalFunction& evalNode, uint32_t rootNodeByteOffset = 0);

        struct BVHStats
        {
            std::vector<uint32_t> nodeCountPerLevel;         ///< For each level in the tree, how many nodes are there.
            std::vector<uint32_t> leafCountPerTriangleCount; ///< For each amount of triangles, how many leaf nodes contain that many triangles.

            uint32_t treeHeight = 0;                         ///< Number of edges on the longest path between the root node and a leaf.
            uint32_t minDepth = 0;                           ///< Number of edges on the shortest path between the root node and a leaf.
            uint32_t byteSize = 0;                           ///< Number of bytes occupied by the BVH.
            uint32_t internalNodeCount = 0;                  ///< Number of internal nodes inside the BVH.
            uint32_t leafNodeCount = 0;                      ///< Number of leaf nodes inside the BVH.
            uint32_t triangleCount = 0;                      ///< Number of triangles inside the BVH.
        };

        /** Returns stats.
        */
        const BVHStats& getStats() const { return mBVHStats; }

        /** Is the BVH valid.
            \return true if the BVH is ready for use.
        */
        virtual bool isValid() const { return mIsValid; }

        /** Render the UI. This default implementation just shows the stats.
        */
        virtual void renderUI(Gui::Widgets& widget);

        /** Bind the light BVH into a shader variable.
            Note that prepareProgram() must have been called before this function.

            \param[in] var The shader variable to set the data into.
            \return True if successful, false otherwise.
        */
        virtual bool setShaderData(ShaderVar const& var) const;

    protected:
        LightBVH(const LightCollection::SharedConstPtr& pLightCollection);

        void computeStats();
        void updateNodeOffsets();
        void renderStats(Gui::Widgets& widget, const BVHStats& stats) const;

        void uploadCPUBuffers(const std::vector<uint64_t>& triangleBitmasks);
        void syncDataToCPU() const;

        /** Allocate enough memory for processing the given number of triangles.
            If the BVH's current capacity can already handle that amount of triangles, no allocation is done.
        */
        void reserve(std::size_t triangleCount);

        /** Return the current size.
            \return the current amount of memory in bytes used by the BVH.
        */
        std::size_t getSize() const { return mAlignedAllocator.getSize(); }

        /** Invalidate the BVH.
        */
        virtual void clear();

        static void verifyStaticParams();

        struct RefitEntryInfo
        {
            uint32_t offset = 0u;
            uint32_t count = 0u;
        };

        // Internal state
        const LightCollection::SharedConstPtr mpLightCollection;

        ComputePass::SharedPtr                mLeafUpdater;             ///< Compute pass for refitting the leaf nodes.
        ComputePass::SharedPtr                mInternalUpdater;         ///< Compute pass for refitting internal nodes.

        // CPU resources
        mutable AlignedAllocator              mAlignedAllocator;        ///< Utility class for the CPU-side node buffer.
        std::vector<uint32_t>                 mNodeOffsets;
        std::vector<RefitEntryInfo>           mPerDepthRefitEntryInfo;  ///< Array containing for each level the number of internal nodes as well as the corresponding offset in mpNodeOffsetsBuffer; the very last entry contains the same data, but for all leaf nodes instead.
        uint32_t                              mMaxTriangleCountPerLeaf = 0u; ///< After the BVH is built, this contains the maximum light count per leaf node.
        BVHStats                              mBVHStats;
        bool                                  mIsValid = false;         ///< True when the BVH has been built.
        mutable bool                          mIsCpuDataValid = false;  ///< Indicates whether the CPU-side data matches the GPU buffers.

        // GPU resources
        Buffer::SharedPtr                     mpBVHNodesBuffer;         ///< Buffer holding all BVH nodes.
        Buffer::SharedPtr                     mpTriangleBitmasksBuffer; ///< Array containing the per triangle bit pattern retracing the tree traversal to reach the triangle: 0=left child, 1=right child.
        Buffer::SharedPtr                     mpNodeOffsetsBuffer;

        friend LightBVHBuilder;
    };
}
