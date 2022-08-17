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
#pragma once
#include "Core/Assert.h"
#include "Utils/Logger.h"
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Falcor
{
    class DirectedGraph
    {
    public:
        using SharedPtr = std::shared_ptr<DirectedGraph>;
        static const uint32_t kInvalidID = (uint32_t)-1;

        class Node;
        class Edge;

        /** Create a new graph.
            \return A new object, or throws an exception if creation failed.
        */
        static SharedPtr create()
        {
            return SharedPtr(new DirectedGraph);
        }

        /** Add a node.
            The returned value is a unique identifier of the node
        */
        uint32_t addNode()
        {
            mNodes[mCurrentNodeId] = Node();
            return mCurrentNodeId++;
        }

        /** Remove a node from the graph. This function will also remove all the incoming and outgoing edges associated with the node
            \return A list of edges that were removed
        */
        std::unordered_set<uint32_t> removeNode(uint32_t id)
        {
            if (mNodes.find(id) == mNodes.end())
            {
                logWarning("Can't remove node from DirectGraph, node ID doesn't exist");
                return {};
            }

            std::unordered_set<uint32_t> removedEdges;
            // Find all the edges we need to remove
            for (auto& edgeId : mNodes[id].mIncomingEdges) findEdgesToRemove<false>(mNodes[mEdges[edgeId].mSrc].mOutgoingEdges, id, removedEdges);
            for (auto& edgeId : mNodes[id].mOutgoingEdges) findEdgesToRemove<true>(mNodes[mEdges[edgeId].mDst].mIncomingEdges, id, removedEdges);

            // Remove them
            for (auto& edgeId : removedEdges) removeEdge(edgeId);

            // Remove the index from the map
            mNodes.erase(id);

            return removedEdges;
        }

        /** Add an edge
        */
        uint32_t addEdge(uint32_t srcNode, uint32_t dstNode)
        {
            if (mNodes.find(srcNode) == mNodes.end())
            {
                logWarning("Can't add an edge to DirectGraph, src node ID doesn't exist");
                return kInvalidID;
            }

            if (mNodes.find(dstNode) == mNodes.end())
            {
                logWarning("Can't add an edge to DirectGraph, src node ID doesn't exist");
                return kInvalidID;
            }

            mNodes[srcNode].mOutgoingEdges.push_back(mCurrentEdgeId);
            mNodes[dstNode].mIncomingEdges.push_back(mCurrentEdgeId);

            mEdges[mCurrentEdgeId] = (Edge(srcNode, dstNode));
            return mCurrentEdgeId++;
        }

        /** Remove an edge
        */
        void removeEdge(uint32_t edgeId)
        {
            if (mEdges.find(edgeId) == mEdges.end())
            {
                logWarning("Can't remove edge from DirectedGraph, edge ID doesn't exist");
                return;
            }

            const auto& edge = mEdges[edgeId];
            removeEdgeFromNode<true>(edgeId, mNodes[edge.getDestNode()]);
            removeEdgeFromNode<false>(edgeId, mNodes[edge.getSourceNode()]);

            mEdges.erase(edgeId);
        }

        class Node
        {
        public:
            Node() = default;
            uint32_t getOutgoingEdgeCount() const { return (uint32_t)mOutgoingEdges.size(); }
            uint32_t getIncomingEdgeCount() const { return (uint32_t)mIncomingEdges.size(); }

            uint32_t getIncomingEdge(uint32_t i) const { return mIncomingEdges[i]; }
            uint32_t getOutgoingEdge(uint32_t i) const { return mOutgoingEdges[i]; }
        private:
            friend DirectedGraph;
            std::vector<uint32_t> mIncomingEdges;
            std::vector<uint32_t> mOutgoingEdges;
        };

        class Edge
        {
        public:
            Edge() = default;
            uint32_t getSourceNode() const { return mSrc; }
            uint32_t getDestNode() const { return mDst; }
        private:
            friend DirectedGraph;
            Edge(uint32_t s, uint32_t d) : mSrc(s), mDst(d) {}
            uint32_t mSrc = kInvalidID;
            uint32_t mDst = kInvalidID;
        };

        /** Check if a node exists
        */
        bool doesNodeExist(uint32_t nodeId) const { return mNodes.find(nodeId) != mNodes.end(); }

        /** Check if an edge exists
        */
        bool doesEdgeExist(uint32_t edgeId) const { return mEdges.find(edgeId) != mEdges.end(); }

        /** Get a node
        */
        const Node* getNode(uint32_t nodeId) const
        {
            if (doesNodeExist(nodeId) == false)
            {
                logWarning("DirectGraph::getNode() - node ID doesn't exist");
                return nullptr;
            }
            return &mNodes.at(nodeId);
        }

        /** Get an edge
        */
        const Edge* getEdge(uint32_t edgeId)
        {
            if (doesEdgeExist(edgeId) == false)
            {
                logWarning("DirectGraph::getEdge() - edge ID doesn't exist");
                return nullptr;
            }
            return &mEdges[edgeId];
        }

        uint32_t getCurrentNodeId() const { return mCurrentNodeId; }
        uint32_t getCurrentEdgeId() const { return mCurrentEdgeId; }
     private:
        DirectedGraph() = default;

        std::unordered_map<uint32_t, Node> mNodes;
        std::unordered_map<uint32_t, Edge> mEdges;
        uint32_t mCurrentNodeId = 0;
        uint32_t mCurrentEdgeId = 0;

        template<bool removeSrc>
        void findEdgesToRemove(std::vector<uint32_t>& edges, uint32_t nodeToRemove, std::unordered_set<uint32_t>& removedEdges)
        {
            for (size_t i = 0; i < edges.size(); i++)
            {
                uint32_t edgeId = edges[i];
                const auto& edge = mEdges[edgeId];
                auto& otherNode = removeSrc ? edge.mSrc : edge.mDst;
                if (otherNode == nodeToRemove)
                {
                    removedEdges.insert(edges[i]);
                }
            }
        }

        template<bool removeInput>
        void removeEdgeFromNode(uint32_t edgeId, Node& node)
        {
            auto& vec = removeInput ? node.mIncomingEdges : node.mOutgoingEdges;
            for (auto e = vec.begin(); e != vec.end(); e++)
            {
                if (*e == edgeId)
                {
                    vec.erase(e);
                    return;
                }
            }
            FALCOR_UNREACHABLE();
        }
    };
}
