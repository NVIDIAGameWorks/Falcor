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
#include "stdafx.h"
#include "LightBVH.h"

namespace
{
    const char kShaderFile[] = "Experimental/Scene/Lights/LightBVHRefit.cs.slang";
}

namespace Falcor
{
    LightBVH::SharedPtr LightBVH::create(const LightCollection::SharedConstPtr& pLightCollection)
    {
        return SharedPtr(new LightBVH(pLightCollection));
    }

    // TODO: Only update the ones that moved.
    void LightBVH::refit(RenderContext* pRenderContext)
    {
        PROFILE("LightBVH::refit()");

        assert(mIsValid);

        const ComputeVars::SharedPtr& pLeafUpdaterVars = mLeafUpdater->getVars();
        mpLightCollection->setShaderData(pLeafUpdaterVars["CB"]["gLights"]);
        pLeafUpdaterVars["CB"]["gLightBVH"]["nodes"] = mpBVHNodesBuffer;
        pLeafUpdaterVars["gNodeOffsets"] = mpNodeOffsetsBuffer;

        const ComputeVars::SharedPtr& pInternalUpdaterVars = mInternalUpdater->getVars();
        mpLightCollection->setShaderData(pInternalUpdaterVars["CB"]["gLights"]);
        pInternalUpdaterVars["CB"]["gLightBVH"]["nodes"] = mpBVHNodesBuffer;
        pInternalUpdaterVars["gNodeOffsets"] = mpNodeOffsetsBuffer;

        // Update all leaf nodes.
        {
            const uint32_t nodeCount = mPerDepthRefitEntryInfo.back().count;
            assert(nodeCount > 0);
            pLeafUpdaterVars["CB"]["gFirstNodeIndex"] = mPerDepthRefitEntryInfo.back().offset;
            pLeafUpdaterVars["CB"]["gNodeCount"] = nodeCount;

            mLeafUpdater->execute(pRenderContext, nodeCount, 1u, 1u);
            pRenderContext->uavBarrier(mpBVHNodesBuffer.get());
        }

        // Update all internal nodes.
        // Note that mBVHStats.treeHeight may be 0, in which case there is a single leaf and no internal nodes.
        for (int depth = (int)mBVHStats.treeHeight - 1; depth >= 0; --depth)
        {
            const uint32_t nodeCount = mPerDepthRefitEntryInfo[depth].count;
            assert(nodeCount > 0);
            pInternalUpdaterVars["CB"]["gFirstNodeIndex"] = mPerDepthRefitEntryInfo[depth].offset;
            pInternalUpdaterVars["CB"]["gNodeCount"] = nodeCount;

            mInternalUpdater->execute(pRenderContext, nodeCount, 1u, 1u);
            pRenderContext->uavBarrier(mpBVHNodesBuffer.get());
        }

        mIsCpuDataValid = false;
    }

    LightBVH::NodeType LightBVH::getNodeType(const uint32_t nodeOffset) const
    {
        assert(isValid());
        syncDataToCPU();
        const uintptr_t rootNode = reinterpret_cast<uintptr_t>(mAlignedAllocator.getStartPointer());
        return *reinterpret_cast<const NodeType*>(rootNode + nodeOffset);
    }

    const LightBVH::InternalNode* LightBVH::getInternalNode(const uint32_t nodeOffset) const
    {
        assert(isValid());
        syncDataToCPU();
        const uintptr_t rootNode = reinterpret_cast<uintptr_t>(mAlignedAllocator.getStartPointer());

        const NodeType nodeType = getNodeType(nodeOffset);
        return nodeType == NodeType::Internal ? reinterpret_cast<const InternalNode*>(rootNode + nodeOffset) : nullptr;
    }

    const LightBVH::LeafNode* LightBVH::getLeafNode(const uint32_t nodeOffset) const
    {
        assert(isValid());
        syncDataToCPU();
        const uintptr_t rootNode = reinterpret_cast<uintptr_t>(mAlignedAllocator.getStartPointer());

        const NodeType nodeType = getNodeType(nodeOffset);
        return nodeType == NodeType::Leaf ? reinterpret_cast<const LeafNode*>(rootNode + nodeOffset) : nullptr;
    }

    void LightBVH::renderUI(Gui::Widgets& widget)
    {
        // Render the BVH stats.
        renderStats(widget, getStats());
    }

    void LightBVH::renderStats(Gui::Widgets& widget, const BVHStats& stats) const
    {
        const std::string statsStr =
            "  Tree height:         " + std::to_string(stats.treeHeight) + "\n" +
            "  Min depth:           " + std::to_string(stats.minDepth) + "\n" +
            "  Size:                " + std::to_string(stats.byteSize) + " bytes\n" +
            "  Internal node count: " + std::to_string(stats.internalNodeCount) + "\n" +
            "  Leaf node count:     " + std::to_string(stats.leafNodeCount) + "\n" +
            "  Triangle count:      " + std::to_string(stats.triangleCount) + "\n";
        widget.text(statsStr.c_str());

        Gui::Group nodeGroup(widget.gui(), "Node count per level");
        if (nodeGroup.open())
        {
            std::string countStr;
            for (uint32_t level = 0u; level < stats.nodeCountPerLevel.size(); ++level)
            {
                countStr += "  Node count at level " + std::to_string(level) + ": " + std::to_string(stats.nodeCountPerLevel[level]) + "\n";
            }
            if (!countStr.empty()) countStr.pop_back();
            nodeGroup.text(countStr.c_str());

            nodeGroup.release();
        }

        Gui::Group leafGroup(widget.gui(), "Leaf node count histogram for triangle counts");
        if (leafGroup.open())
        {
            std::string countStr;
            for (uint32_t triangleCount = 0u; triangleCount < stats.leafCountPerTriangleCount.size(); ++triangleCount)
            {
                countStr += "  Leaf nodes with " + std::to_string(triangleCount) + " triangles: " + std::to_string(stats.leafCountPerTriangleCount[triangleCount]) + "\n";
            }
            if (!countStr.empty()) countStr.pop_back();
            leafGroup.text(countStr.c_str());

            leafGroup.release();
        }
    }

    void LightBVH::reserve(std::size_t triangleCount)
    {
        const size_t cacheLineSize = 128;
        assert(sizeof(InternalNode) < cacheLineSize && sizeof(LeafNode) < cacheLineSize);
        // To be grossly conservative, assume each triangle requires two
        // nodes and each node fills a cache line.  This is only system RAM
        // and shouldn't be that much, so it's not worth being more careful
        // about it.
        const std::size_t capacityBound = 2u * triangleCount * cacheLineSize;
        mAlignedAllocator.reserve(capacityBound);
    }

    void LightBVH::clear()
    {
        // Reset all CPU data.
        mAlignedAllocator.reset();
        mNodeOffsets.clear();
        mPerDepthRefitEntryInfo.clear();
        mMaxTriangleCountPerLeaf = 0u;
        mBVHStats = BVHStats();
        mIsValid = false;
        mIsCpuDataValid = false;
    }

    LightBVH::LightBVH(const LightCollection::SharedConstPtr& pLightCollection) : mpLightCollection(pLightCollection)
    {
        verifyStaticParams();
        mAlignedAllocator.setMinimumAlignment(16);
        mAlignedAllocator.setCacheLineSize(0);  // Don't worry about allocations that straddle two cache lines.

        mLeafUpdater = ComputePass::create(kShaderFile, "updateLeafNodes");
        mInternalUpdater = ComputePass::create(kShaderFile, "updateInternalNodes");
    }

    void LightBVH::traverseBVH(const TraversalEvalFunction& evalNode, uint32_t rootNodeByteOffset)
    {
        std::stack<NodeLocation> stack({ NodeLocation{ rootNodeByteOffset, 0u } });
        while (!stack.empty())
        {
            const NodeLocation location = stack.top();
            stack.pop();

            const InternalNode* const pInternalNode = getInternalNode(location.byteOffset);
            const LeafNode* const pLeafNode = getLeafNode(location.byteOffset);

            if (evalNode(location, pInternalNode, pLeafNode) == false) break;

            if (pInternalNode)
            {
                stack.push(NodeLocation{ pInternalNode->rightNodeOffset, location.depth + 1u });
                stack.push(NodeLocation{ pInternalNode->leftNodeOffset, location.depth + 1u });
            }
        }
    }

    void LightBVH::computeStats()
    {
        assert(isValid());
        mBVHStats.nodeCountPerLevel.clear();
        mBVHStats.nodeCountPerLevel.reserve(32);

        assert(mMaxTriangleCountPerLeaf > 0);
        mBVHStats.leafCountPerTriangleCount.clear();
        mBVHStats.leafCountPerTriangleCount.resize(mMaxTriangleCountPerLeaf + 1, 0u);

        mBVHStats.treeHeight = 0u;
        mBVHStats.minDepth = std::numeric_limits<uint32_t>::max();
        mBVHStats.internalNodeCount = 0u;
        mBVHStats.leafNodeCount = 0u;
        mBVHStats.triangleCount = 0u;

        traverseBVH([this](const NodeLocation& location, const InternalNode* pInternalNode, const LeafNode* pLeafNode)
        {
            if (pInternalNode)
            {
                if (mBVHStats.nodeCountPerLevel.size() <= location.depth) mBVHStats.nodeCountPerLevel.push_back(1u);
                else ++mBVHStats.nodeCountPerLevel[location.depth];

                ++mBVHStats.internalNodeCount;
            }
            else // This is a leaf node
            {
                if (mBVHStats.nodeCountPerLevel.size() <= location.depth) mBVHStats.nodeCountPerLevel.push_back(1u);
                else ++mBVHStats.nodeCountPerLevel[location.depth];

                ++mBVHStats.leafCountPerTriangleCount[pLeafNode->triangleCount];

                mBVHStats.treeHeight = std::max(mBVHStats.treeHeight, location.depth);
                mBVHStats.minDepth = std::min(mBVHStats.minDepth, location.depth);
                ++mBVHStats.leafNodeCount;
                mBVHStats.triangleCount += pLeafNode->triangleCount;
            }

            return true;
        });

        mBVHStats.byteSize = (uint32_t)mAlignedAllocator.getSize();
    }

    void LightBVH::updateNodeOffsets()
    {
        // The nodes of the BVH are stored in depth-first order. To simplify the work of the refit kernels,
        // they are first run on all leaf nodes, and then on all internal nodes on a per level basis.
        // In order to do that, we need to compute how many internal nodes are stored at each level.
        assert(isValid());
        mPerDepthRefitEntryInfo.clear();
        mPerDepthRefitEntryInfo.resize(mBVHStats.treeHeight + 1);
        mPerDepthRefitEntryInfo.back().count = mBVHStats.leafNodeCount;

        traverseBVH([this](const NodeLocation& location, const InternalNode* pInternalNode, const LeafNode* pLeafNode)
        {
            if (pInternalNode) ++mPerDepthRefitEntryInfo[location.depth].count;
            return true;
        });

        std::vector<uint32_t> perDepthOffset(mPerDepthRefitEntryInfo.size(), 0u);
        for (std::size_t i = 1; i < mPerDepthRefitEntryInfo.size(); ++i)
        {
            const uint32_t currentOffset = mPerDepthRefitEntryInfo[i - 1].offset + mPerDepthRefitEntryInfo[i - 1].count;
            perDepthOffset[i] = mPerDepthRefitEntryInfo[i].offset = currentOffset;
        }

        // For validation purposes
        {
            uint32_t currentOffset = 0u;
            for (const RefitEntryInfo& info : mPerDepthRefitEntryInfo)
            {
                assert(info.offset == currentOffset);
                currentOffset += info.count;
            }
            assert(currentOffset == (mBVHStats.internalNodeCount + mBVHStats.leafNodeCount));
        }

        // Now that we know how many nodes are stored per level (excluding leaf nodes) and how many leaf nodes there are,
        // we can fill in the buffer with all the offsets. The offsets are stored as follows
        // <-- Offsets to all internal nodes at level 0 --> | ... | <-- Offsets to all internal nodes at level (treeHeight - 1) --> | <-- Offsets to all leaf nodes -->
        mNodeOffsets.clear();
        mNodeOffsets.resize(mBVHStats.internalNodeCount + mBVHStats.leafNodeCount, 0u);

        traverseBVH([this, &perDepthOffset](const NodeLocation& location, const InternalNode* pInternalNode, const LeafNode* pLeafNode)
        {
            if (pInternalNode)
            {
                mNodeOffsets[perDepthOffset[location.depth]++] = location.byteOffset;
            }
            else // This is a leaf node
            {
                mNodeOffsets[perDepthOffset.back()++] = location.byteOffset;
            }

            return true;
        });

        if (!mpNodeOffsetsBuffer || mpNodeOffsetsBuffer->getElementCount() < mNodeOffsets.size())
        {
            mpNodeOffsetsBuffer = Buffer::createTyped<uint32_t>((uint32_t)mNodeOffsets.size(), ResourceBindFlags::ShaderResource);
            mpNodeOffsetsBuffer->setName("LightBVH_NodeOffsetsBuffer");
        }

        mpNodeOffsetsBuffer->setBlob(mNodeOffsets.data(), 0u, mNodeOffsets.size() * sizeof(uint32_t));
    }

    void LightBVH::uploadCPUBuffers(const std::vector<uint64_t>& triangleBitmasks)
    {
        const uint32_t bvhByteSize = static_cast<uint32_t>(mAlignedAllocator.getSize());

        // Reallocate buffers if size requirements have changed.
        if (!mpBVHNodesBuffer || mpBVHNodesBuffer->getSize() < bvhByteSize)
        {
            // TODO: Test perf with Buffer::CpuAccess::Write flag. It'd speed up CPU->GPU copy below,
            mpBVHNodesBuffer = Buffer::create(bvhByteSize, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
            mpBVHNodesBuffer->setName("LightBVH_BVHNodes");
        }
        if (!mpTriangleBitmasksBuffer || mpTriangleBitmasksBuffer->getElementCount() < triangleBitmasks.size())
        {
            mpTriangleBitmasksBuffer = Buffer::createStructured(
                mLeafUpdater->getRootVar()["CB"]["gLightBVH"]["triangleBitmasks"],
                uint32_t(triangleBitmasks.size()), Resource::BindFlags::ShaderResource);
            mpTriangleBitmasksBuffer->setName("LightBVH_TriangleBitmasks");
        }

        // Update our GPU side buffers.
        // TODO: This is slow. We will build the buffer on the GPU in the future.
        assert(mpBVHNodesBuffer->getSize() >= bvhByteSize);
        mpBVHNodesBuffer->setBlob(mAlignedAllocator.getStartPointer(), 0, bvhByteSize);
        assert(mpTriangleBitmasksBuffer->getSize() >= triangleBitmasks.size() * sizeof(triangleBitmasks[0]));
        mpTriangleBitmasksBuffer->setBlob(triangleBitmasks.data(), 0, triangleBitmasks.size() * sizeof(triangleBitmasks[0]));

        mIsCpuDataValid = true;
    }

    void LightBVH::syncDataToCPU() const
    {
        if (!mIsValid || mIsCpuDataValid) return;

        {
            // TODO: This is slow because of the flush. We should copy to a staging buffer
            // after the data is updated on the GPU and map the staging buffer here instead.
            const void* const ptr = mpBVHNodesBuffer->map(Buffer::MapType::Read);
            assert(getSize() <= mAlignedAllocator.getSize());
            assert(getSize() <= mpBVHNodesBuffer->getSize());
            std::memcpy(mAlignedAllocator.getStartPointer(), ptr, getSize());
        }
        mpBVHNodesBuffer->unmap();
        mIsCpuDataValid = true;
    }

    bool LightBVH::setShaderData(const ShaderVar& var) const
    {
        assert(var.isValid());

        if (isValid())
        {
            var["nodes"] = mpBVHNodesBuffer;
            var["triangleBitmasks"] = mpTriangleBitmasksBuffer;
        }

        return true;
    }

    void LightBVH::verifyStaticParams()
    {
        // Check at compile time all the offsets defined in LightBVHStaticParams for InternalNode.
        static_assert(kNodeTypeOffset            == offsetof(InternalNode, nodeType));
        static_assert(kNodeAABBMinOffset         == offsetof(InternalNode, aabbMin));
        static_assert(kNodeAABBMaxOffset         == offsetof(InternalNode, aabbMax));
        static_assert(kNodeCosConeAngleOffset    == offsetof(InternalNode, cosConeAngle));
        static_assert(kNodeConeDirectionOffset   == offsetof(InternalNode, coneDirection));
        static_assert(kNodeFluxOffset            == offsetof(InternalNode, luminousFlux));
        static_assert(kNodeLeftByteOffsetOffset  == offsetof(InternalNode, leftNodeOffset));
        static_assert(kNodeRightByteOffsetOffset == offsetof(InternalNode, rightNodeOffset));

        // Check at compile time all the offsets defined in LightBVHStaticParams for LeafNode.
        static_assert(kNodeTypeOffset            == offsetof(LeafNode, nodeType));
        static_assert(kNodeAABBMinOffset         == offsetof(LeafNode, aabbMin));
        static_assert(kNodeAABBMaxOffset         == offsetof(LeafNode, aabbMax));
        static_assert(kNodeCosConeAngleOffset    == offsetof(LeafNode, cosConeAngle));
        static_assert(kNodeConeDirectionOffset   == offsetof(LeafNode, coneDirection));
        static_assert(kNodeFluxOffset            == offsetof(LeafNode, luminousFlux));
        static_assert(kNodeTriangleCountOffset   == offsetof(LeafNode, triangleCount));
        static_assert(kNodeTriangleIndicesOffset == offsetof(LeafNode, triangleIndices));
        static_assert(kNodeTriangleIndexByteSize == sizeof(LeafNode::triangleIndices[0]));
    }
}
