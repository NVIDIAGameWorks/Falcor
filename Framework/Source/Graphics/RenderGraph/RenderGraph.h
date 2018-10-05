/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "RenderPass.h"
#include "Utils/DirectedGraph.h"
#include "ResourceCache.h"

namespace Falcor
{
    class Scene;
    class Texture;
    class Fbo;
    class RenderGraphExporter;
    class RenderPassLibrary;

    class RenderGraph
    {
    public:
        using SharedPtr = std::shared_ptr<RenderGraph>;

        static const uint32_t kInvalidIndex = -1;

        ~RenderGraph();

        /** Create a new object
        */
        static SharedPtr create(const std::string& name = "");

        /** Set a scene
        */
        void setScene(const std::shared_ptr<Scene>& pScene);

        /** Add a render-pass. The name has to be unique, otherwise the call will be ignored
        */
        uint32_t addPass(const RenderPass::SharedPtr& pPass, const std::string& passName);

        /** Get a render-pass
        */
        const RenderPass::SharedPtr& getPass(const std::string& name) const;

        /** Remove a render-pass. You need to make sure the edges are still valid after the node was removed
        */
        void removePass(const std::string& name);

        /** Update dictionary for specified render pass.
        */
        void updatePass(const std::string& passName, const Dictionary& dict);

        /** Insert an edge from a render-pass' output into a different render-pass input.
            The render passes must be different, the graph must be a DAG.
            The src/dst strings have the format `renderPassName.resourceName`, where the `renderPassName` is the name used in `setRenderPass()` and the `resourceName` is the resource-name as described by the render-pass object
        */
        uint32_t addEdge(const std::string& src, const std::string& dst);

        /** Remove an edge
         */
        void removeEdge(const std::string& src, const std::string& dst);

        /** Remove an edge
         */
        void removeEdge(uint32_t edgeID);

        /** Check if the graph is ready for execution (all passes inputs/outputs have been initialized correctly, no loops in the graph)
        */
        bool isValid(std::string& log) const;

        /** Execute the graph
        */
        void execute(RenderContext* pContext);

        /** Update graph based on another graph's topology
        */
        void update(const SharedPtr& pGraph);

        /** Set an input resource. The name has the format `renderPassName.resourceName`.
            This is an alias for `getRenderPass(renderPassName)->setInput(resourceName, pResource)`
        */
        bool setInput(const std::string& name, const std::shared_ptr<Resource>& pResource);
        /** Returns true if a render pass exists by this name in the graph.
         */
        bool doesPassExist(const std::string& name) const { return (mNameToIndex.find(name) != mNameToIndex.end()); }

        /** Return the index of a pass from a name, or kInvalidIndex if the pass doesn't exists
        */
        uint32_t getPassIndex(const std::string& name) const;

        /** Get an output resource. The name has the format `renderPassName.resourceName`.
        This is an alias for `getRenderPass(renderPassName)->getOutput(resourceName)`
        */
        const std::shared_ptr<Resource> getOutput(const std::string& name);

        /** Mark a render-pass output as the graph's output. If the graph has no outputs it is invalid.
            The name has the format `renderPassName.resourceName`. You can also use `renderPassName` which will allocate all the render-pass outputs.
            The graph will automatically allocate the output resource
        */
        void markOutput(const std::string& name);

        /** Unmark a graph output
            The name has the format `renderPassName.resourceName`. You can also use `renderPassName` which will allocate all the render-pass outputs
        */
        void unmarkOutput(const std::string& name);

        /** Call this when the swap-chain was resized
        */
        void onResize(const Fbo* pTargetFbo);

        /** Get the attached scene
        */
        const std::shared_ptr<Scene>& getScene() const { return mpScene; }

        /** Get an graph output name from the graph outputs 
        */
        std::string getOutputName(size_t index) const;

        /** Get the num of outputs from this graph
        */
        size_t getOutputCount() const { return mOutputs.size(); }

        /** Get all output names for the render graph
        */
        std::vector<std::string> getAvailableOutputs() const;

        /** Attempts to auto generate edges for render passes.
            \param[in] executionOrder Optional. Ordered list of node ID's as an override of pass search order to use when generating edges.
        */
        void autoGenEdges(const std::vector<uint32_t>& executionOrder = std::vector<uint32_t>());

        /** Render the UI
        */
        void renderUI(Gui* pGui, const char* uiGroup);

        /** Enable/disable pass profiling
        */
        void profileGraph(bool enabled) { mProfileGraph = enabled; }

        /** Mouse event handler.
            Returns true if the event was handled by the object, false otherwise
        */
        bool onMouseEvent(const MouseEvent& mouseEvent);

        /** Keyboard event handler
        Returns true if the event was handled by the object, false otherwise
        */
        bool onKeyEvent(const KeyboardEvent& keyEvent);

        /** Get the dictionary objects used to communicate app data to the render-passes
        */
        const Dictionary::SharedPtr& getPassesDictionary() const { return mpPassDictionary; }

        /** Get the name
        */
        const std::string& getName() const { return mName; }

        /** Get the name
        */
        void setName(const std::string& name) { mName = name; }
    private:
        friend class RenderGraphUI;
        friend class RenderGraphExporter;
        friend class RenderPassLibrary;

        RenderGraph(const std::string& name);
        std::string mName;

        bool compile(std::string& log);
        bool resolveExecutionOrder();
        bool insertAutoPasses();
        bool resolveResourceTypes();
        
        struct EdgeData
        {
            bool autoGenerated = false;
            std::string srcField;
            std::string dstField;
        };

        struct NodeData
        {
            std::string nodeName;
            RenderPass::SharedPtr pPass;
        };

        uint32_t getEdge(const std::string& src, const std::string& dst);
        void getUnsatisfiedInputs(const NodeData* pNodeData, const RenderPassReflection& passReflection, std::vector<RenderPassReflection::Field>& outList) const;
        void autoConnectPasses(const NodeData* pSrcNode, const RenderPassReflection& srcReflection, const NodeData* pDestNode, std::vector<RenderPassReflection::Field>& unsatisfiedInputs);
        bool canAutoResolve(const RenderPassReflection::Field& src, const RenderPassReflection::Field& dst);
        void restoreCompilationChanges();

        bool mRecompile = true;
        std::shared_ptr<Scene> mpScene;

        std::unordered_map<std::string, uint32_t> mNameToIndex;

        DirectedGraph::SharedPtr mpGraph;
        std::unordered_map<uint32_t, EdgeData> mEdgeData;
        std::unordered_map<uint32_t, NodeData> mNodeData;

        struct GraphOut
        {
            uint32_t nodeId;
            std::string field;

            bool operator==(const GraphOut& other) const 
            {
                if (nodeId != other.nodeId) return false;
                if (field != other.field) return false;
                return true;
            }

            bool operator!=(const GraphOut& other) const { return !(*this == other); }
        };

        std::vector<GraphOut> mOutputs; // GRAPH_TODO should this be an unordered set?

        bool isGraphOutput(const GraphOut& graphOut) const;

        ResourceCache::DefaultProperties mSwapChainData;

        std::vector<uint32_t> mExecutionList;
        ResourceCache::SharedPtr mpResourcesCache;

        // TODO Better way to track history, or avoid changing the original graph altogether?
        struct {
            std::vector<std::string> generatedPasses;
            std::vector<std::pair<std::string, std::string>> removedEdges;
        } mCompilationChanges;

        bool mProfileGraph = true;
        Dictionary::SharedPtr mpPassDictionary;
    };

    dlldecl std::vector<RenderGraph*> gRenderGraphs;
}