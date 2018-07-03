/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include <unordered_map>
#include <list>
#include "DirectedGraph.h"

namespace Falcor
{
    enum class GraphTraversalFlags
    {
        None = 0x0,
        Reverse = 0x1,
        IgnoreVisited = 0x2,
    };
    template<typename GraphType>
    class DirectedGraphTraversal
    {
    public:
        using Flags = GraphTraversalFlags;
        DirectedGraphTraversal(const typename GraphType::SharedPtr pGraph, uint32_t rootNode, Flags flags) : mpGraph(pGraph), mFlags(flags)
        {
            reset(rootNode);
        }

        bool reset(uint32_t rootNode)
        {
            if (mpGraph->doesNodeExist(rootNode) == false) return false;
            mNodeStack.push(rootNode);

            if (is_set(mFlags, Flags::IgnoreVisited))
            {
                mVisited.assign(mpGraph->getCurrentNodeId(), false);
            }

            return true;
        }
    protected:
        virtual ~DirectedGraphTraversal() = 0 {}

        typename GraphType::SharedPtr mpGraph;
        Flags mFlags;
        std::vector<bool> mVisited;
        std::stack<uint32_t> mNodeStack;
    };

    enum_class_operators(GraphTraversalFlags);

    template<typename GraphType>
    class DirectedGraphDfsTraversal : public DirectedGraphTraversal<typename GraphType>
    {
    public:
        DirectedGraphDfsTraversal(const typename GraphType::SharedPtr pGraph, uint32_t rootNode, Flags flags = Flags::None) : DirectedGraphTraversal(pGraph, rootNode, flags) {}
        ~DirectedGraphDfsTraversal() = default;

        uint32_t traverse()
        {
            if (mNodeStack.empty())
            {
                logWarning("DFS traversal ended, nowhere new to go");
                return GraphType::kInvalidID;
            }

            uint32_t curNode = mNodeStack.top();
            mNodeStack.pop();

            // Insert all the children
            const GraphType::Node* pNode = mpGraph->getNode(curNode);
            bool reverse = is_set(mFlags, GraphTraversalFlags::Reverse);
            uint32_t edgeCount = reverse ? pNode->getIncomingEdgeCount() : pNode->getOutgoingEdgeCount();

            for (uint32_t i = 0; i < edgeCount; i++)
            {
                uint32_t e = reverse ? pNode->getIncomingEdge(i) : pNode->getOutgoingEdge(i);
                const GraphType::Edge* pEdge = mpGraph->getEdge(e);
                uint32_t child = reverse ? pEdge->getSourceNode() : pEdge->getDestNode();
                mNodeStack.push(child);
            }

            return curNode;
        }
    private:
    };
}