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

namespace Falcor
{
    template<typename NodeData, typename EdgeData>
    class DirectedGraph
    {
    public:
        using SharedPtr = std::shared_ptr<DirectedGraph>;

        /** Create a new graph
        */
        static SharedPtr create()
        {
            return SharedPtr(new DirectedGraph);
        }

        /** Add a node.
            The returned value is a unique identifier of the node
        */
        uint32_t addNode(const NodeData& nodeData)
        {
            mNodes[mCurrentNodeId] = Node(nodeData);
            return mCurrentNodeId++;
        }

        /** Remove a node from the graph. This function will also remove all the incoming and outgoing edges associated with the node
        */
        void removeNode(uint32_t id)
        {
            if (mNodes.find(id) == mNodes.end())
            {
                logWarning("Can't remove node from DirectGraph, node ID doesn't exist");
                return;
            }

            // Loop over the edges and erase the edges
            for (auto& edgeId : mNodes[id].mIncomingEdges) removeEdgeByNode<false>(mNodes[mEdges[edgeId].mSrc].mOutgoingEdges, id);
            for (auto& edgeId : mNodes[id].mOutgoingEdges) removeEdgeByNode<true>(mNodes[mEdges[edgeId].mDst].mIncomingEdges, id);

            // Remove the index from the map
            mNodes.erase(id);
        }

        /** Add an edge
        */
        uint32_t addEdge(uint32_t srcNode, uint32_t dstNode, const EdgeData& edgeData)
        {
            if (mNodes.find(srcNode) == mNodes.end())
            {
                logWarning("Can't add an edge to DirectGraph, src node ID doesn't exist");
                return -1;
            }

            if (mNodes.find(dstNode) == mNodes.end())
            {
                logWarning("Can't add an edge to DirectGraph, src node ID doesn't exist");
                return -1;
            }

            mNodes[srcNode].mOutgoingEdges.push_back(mCurrentEdgeId);
            mNodes[dstNode].mIncomingEdges.push_back(mCurrentEdgeId);

            mEdges[mCurrentEdgeId] = (Edge(srcNode, dstNode, edgeData));
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

        /** Get a node's data
        */
        const NodeData& getNodeData(uint32_t nodeId)
        {
            if (mNodes.find(nodeId) == mNodes.end())
            {
                logWarning("DirectGraph::getNodeData() - node ID doesn't exist");
                static const NodeData nd;
                return nd;
            }

            return mNodes[nodeId].mData;
        }

        /** Get an edge's data
        */
        const EdgeData& getEdgeData(uint32_t edgeId)
        {
            if (mEdges.find(edgeId) == mEdges.end())
            {
                logWarning("DirectGraph::getEdgeData() - edge ID doesn't exist");
                static const EdgeData e;
                return e;
            }

            return mEdges[edgeId].mData;
        }

        class Node
        {
        public:
            Node() = default;
            const NodeData& getData() const { return mData; }
            uint32_t getOutgoingEdgeCount() const { return (uint32_t)mOutgoingEdges.size(); }
            uint32_t getIncomingEdgeCount() const { return (uint32_t)mIncomingEdges.size(); }

            uint32_t getIncomingEdge(uint32_t i) const { return mIncomingEdges[i]; }
            uint32_t getOutgoingEdge(uint32_t i) const { return mOutgoingEdges[i]; }
        private:
            friend DirectedGraph;
            Node(const NodeData& data) : mData(data) {}
            NodeData mData;
            std::vector<uint32_t> mIncomingEdges;
            std::vector<uint32_t> mOutgoingEdges;
        };

        class Edge
        {
        public:
            Edge() = default;
            const EdgeData& getData() const { return mData; }
            uint32_t getSourceNode() const { return mSrc; }
            uint32_t getDestNode() const { return mDst; }
        private:
            friend DirectedGraph;
            Edge(uint32_t s, uint32_t d, const EdgeData& data) : mSrc(s), mDst(d), mData(data) {}
            uint32_t mSrc = -1;
            uint32_t mDst = -1;
            EdgeData mData;
        };
        
        const Node& getNode(uint32_t nodeId)
        {
            if (mNodes.find(nodeId) == mNodes.end())
            {
                logWarning("DirectGraph::getNode() - node ID doesn't exist");
                static const Node n;
                return n;
            }
            return mNodes[nodeId];
        }

        const Edge& getEdge(uint32_t edgeId)
        {
            if (mEdges.find(edgeId) == mEdges.end())
            {
                logWarning("DirectGraph::getEdge() - edge ID doesn't exist");
                static const Edge e;
                return e;
            }
            return mEdges[edgeId];
        }
     private:
        DirectedGraph() = default;

        std::unordered_map<uint32_t, Node> mNodes;
        std::unordered_map<uint32_t, Edge> mEdges;
        uint32_t mCurrentNodeId = 0;
        uint32_t mCurrentEdgeId = 0;

        template<bool removeSrc>
        void removeEdgeByNode(std::vector<uint32_t>& edges, uint32_t nodeToRemove)
        {
            for (size_t i = 0; i < edges.size();)
            {
                const auto& edge = mEdges[edges[i]];
                auto& otherNode = removeSrc ? edge.mSrc : edge.mDst;
                if (otherNode == nodeToRemove)
                {
                    removeEdge(edges[i]);
                }
                else i++;
            }
        }

        template<bool removeInput>
        void removeEdgeFromNode(uint32_t edgeId, Node& node)
        {
            auto& vec = removeInput ? node.mIncomingEdges : node.mOutgoingEdges;
            for (auto& e = vec.begin(); e != vec.end(); e++)
            {
                if (*e == edgeId)
                {
                    vec.erase(e);
                    return;
                }
            }
            should_not_get_here();
        }
    };
}