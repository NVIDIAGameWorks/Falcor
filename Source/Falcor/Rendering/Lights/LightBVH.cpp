/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "LightBVH.h"
#include "Core/Assert.h"
#include "Core/API/RenderContext.h"
#include "Utils/Timing/Profiler.h"

namespace
{
    const char kShaderFile[] = "Rendering/Lights/LightBVHRefit.cs.slang";
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
        FALCOR_PROFILE("LightBVH::refit()");

        FALCOR_ASSERT(mIsValid);

        // Update all leaf nodes.
        {
            auto var = mLeafUpdater->getVars()["CB"];
            mpLightCollection->setShaderData(var["gLights"]);
            setShaderData(var["gLightBVH"]);
            var["gNodeIndices"] = mpNodeIndicesBuffer;

            const uint32_t nodeCount = mPerDepthRefitEntryInfo.back().count;
            FALCOR_ASSERT(nodeCount > 0);
            var["gFirstNodeOffset"] = mPerDepthRefitEntryInfo.back().offset;
            var["gNodeCount"] = nodeCount;

            mLeafUpdater->execute(pRenderContext, nodeCount, 1, 1);
        }

        // Update all internal nodes.
        {
            auto var = mInternalUpdater->getVars()["CB"];
            mpLightCollection->setShaderData(var["gLights"]);
            setShaderData(var["gLightBVH"]);
            var["gNodeIndices"] = mpNodeIndicesBuffer;

            // Note that mBVHStats.treeHeight may be 0, in which case there is a single leaf and no internal nodes.
            for (int depth = (int)mBVHStats.treeHeight - 1; depth >= 0; --depth)
            {
                const uint32_t nodeCount = mPerDepthRefitEntryInfo[depth].count;
                FALCOR_ASSERT(nodeCount > 0);
                var["gFirstNodeOffset"] = mPerDepthRefitEntryInfo[depth].offset;
                var["gNodeCount"] = nodeCount;

                mInternalUpdater->execute(pRenderContext, nodeCount, 1, 1);
            }
        }

        mIsCpuDataValid = false;
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
        widget.text(statsStr);

        if (auto nodeGroup = widget.group("Node count per level"))
        {
            std::string countStr;
            for (uint32_t level = 0; level < stats.nodeCountPerLevel.size(); ++level)
            {
                countStr += "  Node count at level " + std::to_string(level) + ": " + std::to_string(stats.nodeCountPerLevel[level]) + "\n";
            }
            if (!countStr.empty()) countStr.pop_back();
            nodeGroup.text(countStr);
        }

        if (auto leafGroup = widget.group("Leaf node count histogram for triangle counts"))
        {
            std::string countStr;
            for (uint32_t triangleCount = 0; triangleCount < stats.leafCountPerTriangleCount.size(); ++triangleCount)
            {
                countStr += "  Leaf nodes with " + std::to_string(triangleCount) + " triangles: " + std::to_string(stats.leafCountPerTriangleCount[triangleCount]) + "\n";
            }
            if (!countStr.empty()) countStr.pop_back();
            leafGroup.text(countStr);
        }
    }

    void LightBVH::clear()
    {
        // Reset all CPU data.
        mNodes.clear();
        mNodeIndices.clear();
        mPerDepthRefitEntryInfo.clear();
        mMaxTriangleCountPerLeaf = 0;
        mBVHStats = BVHStats();
        mIsValid = false;
        mIsCpuDataValid = false;
    }

    LightBVH::LightBVH(const LightCollection::SharedConstPtr& pLightCollection) : mpLightCollection(pLightCollection)
    {
        mLeafUpdater = ComputePass::create(kShaderFile, "updateLeafNodes");
        mInternalUpdater = ComputePass::create(kShaderFile, "updateInternalNodes");
    }

    void LightBVH::traverseBVH(const NodeFunction& evalInternal, const NodeFunction& evalLeaf, uint32_t rootNodeIndex)
    {
        std::stack<NodeLocation> stack({ NodeLocation{ rootNodeIndex, 0 } });
        while (!stack.empty())
        {
            const NodeLocation location = stack.top();
            stack.pop();

            if (mNodes[location.nodeIndex].isLeaf())
            {
                if (!evalLeaf(location)) break;
            }
            else
            {
                if (!evalInternal(location)) break;

                // Push the children nodes onto the stack.
                auto node = mNodes[location.nodeIndex].getInternalNode();
                stack.push(NodeLocation{ location.nodeIndex + 1, location.depth + 1 });
                stack.push(NodeLocation{ node.rightChildIdx, location.depth + 1 });
            }
        }
    }

    void LightBVH::finalize()
    {
        // This function is called after BVH build has finished.
        computeStats();
        updateNodeIndices();
    }

    void LightBVH::computeStats()
    {
        FALCOR_ASSERT(isValid());
        mBVHStats.nodeCountPerLevel.clear();
        mBVHStats.nodeCountPerLevel.reserve(32);

        FALCOR_ASSERT(mMaxTriangleCountPerLeaf > 0);
        mBVHStats.leafCountPerTriangleCount.clear();
        mBVHStats.leafCountPerTriangleCount.resize(mMaxTriangleCountPerLeaf + 1, 0);

        mBVHStats.treeHeight = 0;
        mBVHStats.minDepth = std::numeric_limits<uint32_t>::max();
        mBVHStats.internalNodeCount = 0;
        mBVHStats.leafNodeCount = 0;
        mBVHStats.triangleCount = 0;

        auto evalInternal = [&](const NodeLocation& location)
        {
            if (mBVHStats.nodeCountPerLevel.size() <= location.depth) mBVHStats.nodeCountPerLevel.push_back(1);
            else ++mBVHStats.nodeCountPerLevel[location.depth];

            ++mBVHStats.internalNodeCount;
            return true;
        };
        auto evalLeaf = [&](const NodeLocation& location)
        {
            const auto node = mNodes[location.nodeIndex].getLeafNode();

            if (mBVHStats.nodeCountPerLevel.size() <= location.depth) mBVHStats.nodeCountPerLevel.push_back(1);
            else ++mBVHStats.nodeCountPerLevel[location.depth];

            ++mBVHStats.leafCountPerTriangleCount[node.triangleCount];
            ++mBVHStats.leafNodeCount;

            mBVHStats.treeHeight = std::max(mBVHStats.treeHeight, location.depth);
            mBVHStats.minDepth = std::min(mBVHStats.minDepth, location.depth);
            mBVHStats.triangleCount += node.triangleCount;
            return true;
        };
        traverseBVH(evalInternal, evalLeaf);

        mBVHStats.byteSize = (uint32_t)(mNodes.size() * sizeof(mNodes[0]));
    }

    void LightBVH::updateNodeIndices()
    {
        // The nodes of the BVH are stored in depth-first order. To simplify the work of the refit kernels,
        // they are first run on all leaf nodes, and then on all internal nodes on a per level basis.
        // In order to do that, we need to compute how many internal nodes are stored at each level.
        FALCOR_ASSERT(isValid());
        mPerDepthRefitEntryInfo.clear();
        mPerDepthRefitEntryInfo.resize(mBVHStats.treeHeight + 1);
        mPerDepthRefitEntryInfo.back().count = mBVHStats.leafNodeCount;

        traverseBVH(
            [&](const NodeLocation& location) { ++mPerDepthRefitEntryInfo[location.depth].count; return true; },
            [](const NodeLocation& location) { return true; }
        );

        std::vector<uint32_t> perDepthOffset(mPerDepthRefitEntryInfo.size(), 0);
        for (std::size_t i = 1; i < mPerDepthRefitEntryInfo.size(); ++i)
        {
            uint32_t currentOffset = mPerDepthRefitEntryInfo[i - 1].offset + mPerDepthRefitEntryInfo[i - 1].count;
            perDepthOffset[i] = mPerDepthRefitEntryInfo[i].offset = currentOffset;
        }

        // For validation purposes
        {
            uint32_t currentOffset = 0;
            for (const RefitEntryInfo& info : mPerDepthRefitEntryInfo)
            {
                FALCOR_ASSERT(info.offset == currentOffset);
                currentOffset += info.count;
            }
            FALCOR_ASSERT(currentOffset == (mBVHStats.internalNodeCount + mBVHStats.leafNodeCount));
        }

        // Now that we know how many nodes are stored per level (excluding leaf nodes) and how many leaf nodes there are,
        // we can fill in the buffer with all the node indices sorted by tree level. The indices are stored as follows
        // <-- Indices to all internal nodes at level 0 --> | ... | <-- Indices to all internal nodes at level (treeHeight - 1) --> | <-- Indices to all leaf nodes -->
        mNodeIndices.clear();
        mNodeIndices.resize(mBVHStats.internalNodeCount + mBVHStats.leafNodeCount, 0);

        traverseBVH(
            [&](const NodeLocation& location) { mNodeIndices[perDepthOffset[location.depth]++] = location.nodeIndex; return true; },
            [&](const NodeLocation& location) { mNodeIndices[perDepthOffset.back()++] = location.nodeIndex; return true; }
        );

        if (!mpNodeIndicesBuffer || mpNodeIndicesBuffer->getElementCount() < mNodeIndices.size())
        {
            mpNodeIndicesBuffer = Buffer::createStructured(sizeof(uint32_t), (uint32_t)mNodeIndices.size(), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpNodeIndicesBuffer->setName("LightBVH::mpNodeIndicesBuffer");
        }

        mpNodeIndicesBuffer->setBlob(mNodeIndices.data(), 0, mNodeIndices.size() * sizeof(uint32_t));
    }

    void LightBVH::uploadCPUBuffers(const std::vector<uint32_t>& triangleIndices, const std::vector<uint64_t>& triangleBitmasks)
    {
        // Reallocate buffers if size requirements have changed.
        auto var = mLeafUpdater->getRootVar()["CB"]["gLightBVH"];
        if (!mpBVHNodesBuffer || mpBVHNodesBuffer->getElementCount() < mNodes.size())
        {
            mpBVHNodesBuffer = Buffer::createStructured(var["nodes"], (uint32_t)mNodes.size(), Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
            mpBVHNodesBuffer->setName("LightBVH::mpBVHNodesBuffer");
        }
        if (!mpTriangleIndicesBuffer || mpTriangleIndicesBuffer->getElementCount() < triangleIndices.size())
        {
            mpTriangleIndicesBuffer = Buffer::createStructured(var["triangleIndices"], (uint32_t)triangleIndices.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpTriangleIndicesBuffer->setName("LightBVH::mpTriangleIndicesBuffer");
        }
        if (!mpTriangleBitmasksBuffer || mpTriangleBitmasksBuffer->getElementCount() < triangleBitmasks.size())
        {
            mpTriangleBitmasksBuffer = Buffer::createStructured(var["triangleBitmasks"], (uint32_t)triangleBitmasks.size(), Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, nullptr, false);
            mpTriangleBitmasksBuffer->setName("LightBVH::mpTriangleBitmasksBuffer");
        }

        // Update our GPU side buffers.
        FALCOR_ASSERT(mpBVHNodesBuffer->getElementCount() >= mNodes.size());
        FALCOR_ASSERT(mpBVHNodesBuffer->getStructSize() == sizeof(mNodes[0]));
        mpBVHNodesBuffer->setBlob(mNodes.data(), 0, mNodes.size() * sizeof(mNodes[0]));

        FALCOR_ASSERT(mpTriangleIndicesBuffer->getSize() >= triangleIndices.size() * sizeof(triangleIndices[0]));
        mpTriangleIndicesBuffer->setBlob(triangleIndices.data(), 0, triangleIndices.size() * sizeof(triangleIndices[0]));

        FALCOR_ASSERT(mpTriangleBitmasksBuffer->getSize() >= triangleBitmasks.size() * sizeof(triangleBitmasks[0]));
        mpTriangleBitmasksBuffer->setBlob(triangleBitmasks.data(), 0, triangleBitmasks.size() * sizeof(triangleBitmasks[0]));

        mIsCpuDataValid = true;
    }

    void LightBVH::syncDataToCPU() const
    {
        if (!mIsValid || mIsCpuDataValid) return;

        // TODO: This is slow because of the flush. We should copy to a staging buffer
        // after the data is updated on the GPU and map the staging buffer here instead.
        const void* const ptr = mpBVHNodesBuffer->map(Buffer::MapType::Read);
        FALCOR_ASSERT(mNodes.size() > 0 && mNodes.size() <= mpBVHNodesBuffer->getElementCount());
        std::memcpy(mNodes.data(), ptr, mNodes.size() * sizeof(mNodes[0]));
        mpBVHNodesBuffer->unmap();
        mIsCpuDataValid = true;
    }

    void LightBVH::setShaderData(const ShaderVar& var) const
    {
        if (isValid())
        {
            FALCOR_ASSERT(var.isValid());
            var["nodes"] = mpBVHNodesBuffer;
            var["triangleIndices"] = mpTriangleIndicesBuffer;
            var["triangleBitmasks"] = mpTriangleBitmasksBuffer;
        }
    }
}
