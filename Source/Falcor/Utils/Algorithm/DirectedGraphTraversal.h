/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "DirectedGraph.h"
#include "Core/Macros.h"
#include "Core/Error.h"
#include <queue>
#include <stack>
#include <string>
#include <vector>

namespace Falcor
{
class DirectedGraphTraversal
{
public:
    enum class Flags
    {
        None = 0x0,
        Reverse = 0x1,
        IgnoreVisited = 0x2,
    };

    DirectedGraphTraversal(const DirectedGraph& graph, Flags flags) : mGraph(graph), mFlags(flags) {}

protected:
    virtual ~DirectedGraphTraversal() {}

    const DirectedGraph& mGraph;
    Flags mFlags;
    std::vector<bool> mVisited;

    bool reset(uint32_t rootNode)
    {
        if (mGraph.doesNodeExist(rootNode) == false)
            return false;

        if ((uint32_t)mFlags & (uint32_t)Flags::IgnoreVisited)
        {
            mVisited.assign(mGraph.getCurrentNodeId(), false);
        }

        return true;
    }
};

FALCOR_ENUM_CLASS_OPERATORS(DirectedGraphTraversal::Flags);

template<typename Args>
class DirectedGraphTraversalTemplate : public DirectedGraphTraversal
{
public:
    DirectedGraphTraversalTemplate(const DirectedGraph& graph, uint32_t rootNode, Flags flags = Flags::None)
        : DirectedGraphTraversal(graph, flags)
    {
        reset(rootNode);
    }
    ~DirectedGraphTraversalTemplate() = default;

    uint32_t traverse()
    {
        if (mNodeList.empty())
        {
            return DirectedGraph::kInvalidID;
        }

        uint32_t curNode = Args::getTop(mNodeList);
        if (is_set(mFlags, Flags::IgnoreVisited))
        {
            while (mVisited[curNode])
            {
                mNodeList.pop();
                if (mNodeList.empty())
                {
                    return DirectedGraph::kInvalidID;
                }
                curNode = Args::getTop(mNodeList);
            }

            mVisited[curNode] = true;
        }
        mNodeList.pop();

        // Insert all the children
        const DirectedGraph::Node* pNode = mGraph.getNode(curNode);
        FALCOR_ASSERT(pNode);
        bool reverse = is_set(mFlags, Flags::Reverse);
        uint32_t edgeCount = reverse ? pNode->getIncomingEdgeCount() : pNode->getOutgoingEdgeCount();

        for (uint32_t i = 0; i < edgeCount; i++)
        {
            uint32_t e = reverse ? pNode->getIncomingEdge(i) : pNode->getOutgoingEdge(i);
            const DirectedGraph::Edge* pEdge = mGraph.getEdge(e);
            FALCOR_ASSERT(pEdge);
            uint32_t child = reverse ? pEdge->getSourceNode() : pEdge->getDestNode();
            mNodeList.push(child);
        }

        return curNode;
    }

    bool reset(uint32_t rootNode)
    {
        bool b = DirectedGraphTraversal::reset(rootNode);
        mNodeList = decltype(mNodeList)();
        if (b)
            mNodeList.push(rootNode);
        return b;
    }

private:
    typename Args::Container mNodeList;
};

struct DfsArgs
{
    using Container = std::stack<uint32_t>;
    static const std::string getName() { return "DFS"; }
    static const uint32_t& getTop(const Container& c) { return c.top(); };
};
using DirectedGraphDfsTraversal = DirectedGraphTraversalTemplate<DfsArgs>;

struct BfsArgs
{
    using Container = std::queue<uint32_t>;
    static const std::string getName() { return "BFS"; }
    static const uint32_t& getTop(const Container& c) { return c.front(); };
};
using DirectedGraphBfsTraversal = DirectedGraphTraversalTemplate<BfsArgs>;

class DirectedGraphLoopDetector
{
public:
    static bool hasLoop(const DirectedGraph& graph, uint32_t rootNode)
    {
        DirectedGraphDfsTraversal dfs(graph, rootNode);
        // Skip the first node since it's the root
        uint32_t n = dfs.traverse();
        while (n != DirectedGraph::kInvalidID)
        {
            n = dfs.traverse();
            if (n == rootNode)
                return true;
        }

        return false;
    }
};

class DirectedGraphTopologicalSort
{
public:
    static std::vector<uint32_t> sort(const DirectedGraph& graph)
    {
        DirectedGraphTopologicalSort ts(graph);
        for (uint32_t i = 0; i < ts.mGraph.getCurrentNodeId(); i++)
        {
            if (ts.mVisited[i] == false && ts.mGraph.getNode(i))
            {
                ts.sortInternal(i);
            }
        }

        std::vector<uint32_t> result;
        result.reserve(ts.mStack.size());
        while (ts.mStack.empty() == false)
        {
            result.push_back(ts.mStack.top());
            ts.mStack.pop();
        }
        return result;
    }

private:
    DirectedGraphTopologicalSort(const DirectedGraph& graph) : mGraph(graph), mVisited(graph.getCurrentNodeId(), false) {}
    const DirectedGraph& mGraph;
    std::stack<uint32_t> mStack;
    std::vector<bool> mVisited;

    void sortInternal(uint32_t node)
    {
        mVisited[node] = true;
        const DirectedGraph::Node* pNode = mGraph.getNode(node);
        FALCOR_ASSERT(pNode);
        for (uint32_t e = 0; e < pNode->getOutgoingEdgeCount(); e++)
        {
            uint32_t nextNode = mGraph.getEdge(pNode->getOutgoingEdge(e))->getDestNode();
            if (!mVisited[nextNode])
            {
                sortInternal(nextNode);
            }
        }

        mStack.push(node);
    }
};

namespace DirectedGraphPathDetector
{
inline bool hasPath(const DirectedGraph& graph, uint32_t from, uint32_t to)
{
    DirectedGraphDfsTraversal dfs(graph, from, DirectedGraphDfsTraversal::Flags::IgnoreVisited);
    uint32_t node = dfs.traverse();
    node = dfs.traverse(); // skip the root node
    while (node != DirectedGraph::kInvalidID)
    {
        if (node == to)
            return true;
        node = dfs.traverse();
    }
    return false;
}

inline bool hasCycle(const DirectedGraph& graph, uint32_t root)
{
    return hasPath(graph, root, root);
}
}; // namespace DirectedGraphPathDetector
} // namespace Falcor
